"""Schema-flexible CSV ingestion and data quality reporting."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
DATE_COLUMN_PATTERN = re.compile(r"(date|datetime|timestamp)", re.IGNORECASE)
PLAYER_ID_COLUMNS = {"playerid", "personid"}
TEAM_ID_COLUMNS = {
    "teamid",
    "hometeamid",
    "awayteamid",
    "playerteamid",
    "opponentteamid",
}
SCORE_COLUMNS = {
    "homescore",
    "awayscore",
    "teamscore",
    "opponentscore",
    "score",
}


def _column_key(column: Any) -> str:
    """Return a case- and punctuation-insensitive form of a column name."""
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def _clean_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Trim text, normalize date-like columns, and count unparseable dates."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].astype("string").str.strip()
        frame[column] = values.replace("", pd.NA)

    invalid_dates: dict[str, int] = {}
    for column in frame.columns:
        if not DATE_COLUMN_PATTERN.search(str(column)):
            continue
        original = frame[column]
        non_missing = original.notna()
        parsed = pd.to_datetime(original, format="mixed", errors="coerce", utc=True)
        invalid_count = int((non_missing & parsed.isna()).sum())
        if invalid_count:
            invalid_dates[str(column)] = invalid_count
        frame[column] = parsed

    return frame, invalid_dates


def _quality_checks(frame: pd.DataFrame) -> dict[str, Any]:
    """Check only NBA fields present in this input; leave absent fields optional."""
    player_fields: dict[str, int] = {}
    team_fields: dict[str, int] = {}
    score_fields: dict[str, dict[str, int]] = {}

    for column in frame.columns:
        key = _column_key(column)
        if key in PLAYER_ID_COLUMNS:
            player_fields[str(column)] = int(frame[column].isna().sum())
        if key in TEAM_ID_COLUMNS:
            team_fields[str(column)] = int(frame[column].isna().sum())
        if key in SCORE_COLUMNS:
            values = pd.to_numeric(frame[column], errors="coerce")
            score_fields[str(column)] = {
                "missing_or_non_numeric": int(values.isna().sum()),
                "negative": int((values < 0).sum()),
            }

    return {
        "player_id_missing_values": player_fields,
        "team_id_missing_values": team_fields,
        "score_validation": score_fields,
    }


def _dataset_kind(filename: str, columns: list[str]) -> str:
    """Infer a broad dataset category from filename and available columns."""
    keys = {_column_key(column) for column in columns}
    stem = Path(filename).stem.lower()
    if "player" in stem or ({"personid", "playerid"} & keys):
        return "player_statistics" if "gameid" in keys else "players"
    if "teamhistory" in stem or {"seasonfounded", "seasonactivetill"} <= keys:
        return "team_histories"
    if "schedule" in stem or "gameday" in keys:
        return "schedule"
    if "team" in stem and "gameid" in keys:
        return "team_statistics"
    if "gameid" in keys and ({"hometeamid", "awayscore"} & keys):
        return "games"
    if "gameid" in keys and "teamid" in keys:
        return "team_statistics"
    return "unknown"


def ingest_csv_file(source: Path, output_dir: Path) -> dict[str, Any]:
    """Clean one CSV, write its Parquet equivalent, and return quality metadata."""
    frame = pd.read_csv(source, low_memory=False, skipinitialspace=True)
    original_rows = len(frame)
    original_dtypes = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    frame, invalid_dates = _clean_dataframe(frame)
    duplicate_rows = int(frame.duplicated(keep="first").sum())
    frame = frame.drop_duplicates(keep="first").reset_index(drop=True)

    parquet_path = output_dir / f"{source.stem}.parquet"
    temporary_path = parquet_path.with_name(f".{parquet_path.name}.tmp")
    try:
        frame.to_parquet(temporary_path, index=False, engine="pyarrow")
        temporary_path.replace(parquet_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    missing_values = {
        str(column): int(count)
        for column, count in frame.isna().sum().items()
    }
    return {
        "source": source.name,
        "processed_file": parquet_path.name,
        "dataset_kind": _dataset_kind(source.name, list(frame.columns)),
        "source_rows": original_rows,
        "processed_rows": len(frame),
        "duplicate_rows_removed": duplicate_rows,
        "columns": [str(column) for column in frame.columns],
        "source_dtypes": original_dtypes,
        "processed_dtypes": {
            str(column): str(dtype) for column, dtype in frame.dtypes.items()
        },
        "missing_values": missing_values,
        "invalid_dates": invalid_dates,
        "nba_field_checks": _quality_checks(frame),
    }


def ingest_directory(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Ingest every top-level CSV in a directory and persist a quality report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.csv"))
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for source in files:
        try:
            result = ingest_csv_file(source, output_dir)
            results.append(result)
            LOGGER.info(
                "Processed %s: %s rows, removed %s duplicates",
                source.name,
                result["processed_rows"],
                result["duplicate_rows_removed"],
            )
        except (
            OSError,
            UnicodeError,
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as error:
            LOGGER.exception("Could not process %s", source)
            errors.append({"source": source.name, "error": str(error)})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir),
        "processed_directory": str(output_dir),
        "files_discovered": len(files),
        "files_processed": len(results),
        "files_failed": len(errors),
        "datasets": results,
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    """Run CSV ingestion from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data" / "raw",
        help="Directory containing source CSV files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed",
        help="Directory for generated Parquet files (default: data/processed)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "data" / "processed" / "data_quality_report.json",
        help="Path for the generated JSON quality report",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    report = ingest_directory(args.input_dir, args.output_dir, args.report)
    LOGGER.info(
        "Ingestion complete: %s/%s files processed; report: %s",
        report["files_processed"],
        report["files_discovered"],
        args.report,
    )
    return 1 if report["files_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
