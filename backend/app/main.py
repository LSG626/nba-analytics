"""FastAPI application for local COURTSIDE AI dataset management."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import Query
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from backend.app import eda

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EDA_FALLBACK_DIR = PROJECT_ROOT / "Raw"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_EVALUATION_DIR = PROJECT_ROOT / "ml" / "evaluation"
PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"
QUALITY_REPORT_PATH = PROCESSED_DATA_DIR / "data_quality_report.json"

app = FastAPI(
    title="COURTSIDE AI API",
    description="Local NBA dataset ingestion and quality metadata.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _raw_csv_files() -> list[Path]:
    """Return available source CSV files in the configured raw-data directory."""
    return sorted(RAW_DATA_DIR.glob("*.csv")) if RAW_DATA_DIR.is_dir() else []


def _report_datasets() -> dict[str, dict[str, Any]]:
    """Read processed dataset metadata when an ingestion report is available."""
    if not QUALITY_REPORT_PATH.is_file():
        return {}
    try:
        report = json.loads(QUALITY_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        LOGGER.exception("Could not read data quality report")
        raise HTTPException(status_code=500, detail="Data quality report is unreadable") from error
    return {
        item["source"]: item
        for item in report.get("datasets", [])
        if isinstance(item, dict) and isinstance(item.get("source"), str)
    }


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Report that the API process is responsive."""
    return {"status": "ok", "service": "courtside-ai"}


def _eda_directories() -> list[Path]:
    """Search processed, configured raw, and supplied legacy Raw data locations."""
    return [PROCESSED_DATA_DIR, RAW_DATA_DIR, EDA_FALLBACK_DIR]


@app.get("/api/eda/overview")
def eda_overview() -> dict[str, Any]:
    """Summarize schemas, coverage, quality and distributions in available tables."""
    return eda.overview(_eda_directories())


@app.get("/api/eda/games")
def eda_games(
    season: str | None = Query(default=None),
    team: str | None = Query(default=None),
    player: str | None = Query(default=None),
) -> dict[str, Any]:
    """Analyze game outcomes and scoring, reporting unsupported player filters."""
    result = eda.games_analysis(_eda_directories(), season=season, team=team)
    if player:
        result.setdefault("filters", {}).setdefault("requested", {})["player"] = player
        result.setdefault("filters", {}).setdefault("unsupported", []).append("player")
    return result


@app.get("/api/eda/teams")
def eda_teams(
    season: str | None = Query(default=None),
    team: str | None = Query(default=None),
) -> dict[str, Any]:
    """Summarize team performance and season splits where fields are available."""
    return eda.teams_analysis(_eda_directories(), season=season, team=team)


@app.get("/api/eda/players")
def eda_players(
    season: str | None = Query(default=None),
    team: str | None = Query(default=None),
    player: str | None = Query(default=None),
) -> dict[str, Any]:
    """Summarize player performance, leaders and consistency from player box scores."""
    return eda.players_analysis(
        _eda_directories(), season=season, team=team, player=player
    )


@app.get("/api/eda/correlations")
def eda_correlations(
    season: str | None = Query(default=None),
    team: str | None = Query(default=None),
    player: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return correlation matrices, outcome relationships and IQR outlier counts."""
    return eda.correlations_analysis(
        _eda_directories(), season=season, team=team, player=player
    )


@app.get("/api/eda/seasons")
def eda_seasons(
    season: str | None = Query(default=None),
    team: str | None = Query(default=None),
    player: str | None = Query(default=None),
) -> dict[str, Any]:
    """Compare season-level scoring and team/player performance."""
    return eda.seasons_analysis(
        _eda_directories(), season=season, team=team, player=player
    )


def _read_model_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        LOGGER.exception("Could not read model metadata %s", path)
        raise HTTPException(status_code=500, detail="Model metadata is unreadable") from error


@app.get("/api/predictions/games")
def predict_game(
    home_team: str = Query(..., min_length=1),
    away_team: str = Query(..., min_length=1),
    game_date: str | None = Query(default=None),
) -> dict[str, Any]:
    """Predict a requested matchup using only saved local models and history."""
    from ml.predict import predict_matchup

    try:
        parsed_date = pd.to_datetime(game_date, utc=True).to_pydatetime() if game_date else None
        return predict_matchup(
            home_team, away_team, parsed_date,
            model_dir=MODEL_DIR, prediction_dir=PREDICTION_DIR,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/predictions/players")
def predict_player(
    player: str = Query(..., min_length=1),
    opponent: str | None = Query(default=None),
    home: bool = Query(default=True),
    game_date: str | None = Query(default=None),
) -> dict[str, Any]:
    """Forecast supported player statistics without claiming lineup availability."""
    from ml.predict import predict_player as run_player_prediction

    try:
        parsed_date = pd.to_datetime(game_date, utc=True).to_pydatetime() if game_date else None
        return run_player_prediction(
            player, opponent=opponent, home=home, game_date=parsed_date,
            model_dir=MODEL_DIR, prediction_dir=PREDICTION_DIR,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/predictions/teams")
def predict_team_strength(team: str = Query(..., min_length=1)) -> dict[str, Any]:
    """Return latest data-backed Elo and available team rating summaries."""
    from ml.predict import _resolve_team

    payload = _read_model_json(PREDICTION_DIR / "team_strength.json")
    if not payload:
        raise HTTPException(status_code=503, detail="Team-strength data is not available; train models first.")
    try:
        team_id, game_state = _resolve_team(team, payload.get("game_team_state", {}))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    team_stats = payload.get("team_statistics", {}).get("teams", {}).get(team_id, {})
    return {
        "available": True,
        "team": game_state.get("team", team_id),
        "team_id": team_id,
        "as_of": payload.get("as_of"),
        "games": game_state.get("games"),
        "win_percentage": game_state.get("win_percentage"),
        "elo": game_state.get("elo"),
        "average_points_for": game_state.get("average_points_for"),
        "average_points_against": game_state.get("average_points_against"),
        "net_rating_proxy": game_state.get("net_rating_proxy"),
        "supported_ratings": {
            key: value for key, value in team_stats.items()
            if key in {"offensive_rating", "defensive_rating", "net_rating", "pace"}
        },
        "ratings_source": payload.get("team_statistics", {}).get("source"),
        "limitations": [
            "Elo is calculated from historical scores; it is not an official NBA rating.",
            "Rating and pace summaries are historical means when present in the supplied team table.",
        ],
    }


@app.get("/api/predictions/latest")
def latest_predictions() -> dict[str, Any]:
    """Return saved forecasts only for future games found in uploaded schedules."""
    path = PREDICTION_DIR / "upcoming_game_predictions.csv"
    if not path.is_file():
        return {"available": False, "predictions": [], "reason": "No scheduled-game predictions have been generated."}
    try:
        frame = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return {
            "available": False,
            "prediction_count": 0,
            "predictions": [],
            "reason": "The saved schedule-prediction file is empty.",
        }
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        LOGGER.exception("Could not read upcoming predictions")
        raise HTTPException(status_code=500, detail="Upcoming prediction file is unreadable") from error
    predictions = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    return {
        "available": bool(predictions),
        "prediction_count": len(predictions),
        "predictions": predictions,
        "reason": None if predictions else "No future scheduled games matched known historical teams.",
    }


@app.get("/api/predictions/teams/list")
def prediction_teams() -> dict[str, Any]:
    """List historical teams available to the saved prediction models."""
    payload = _read_model_json(PREDICTION_DIR / "team_strength.json")
    teams = payload.get("game_team_state", {})
    if not teams:
        return {"available": False, "teams": [], "reason": "Train game models first."}
    return {
        "available": True,
        "as_of": payload.get("as_of"),
        "teams": sorted(
            [
                {
                    "team_id": str(team_id),
                    "team": state.get("team", str(team_id)),
                    "games": state.get("games"),
                }
                for team_id, state in teams.items()
            ],
            key=lambda item: str(item["team"]).casefold(),
        ),
    }


@app.get("/api/predictions/players/search")
def prediction_player_search(
    q: str = Query(..., min_length=2),
    limit: int = Query(default=12, ge=1, le=50),
) -> dict[str, Any]:
    """Search players with saved prediction state, not the larger box-score table."""
    payload = _read_model_json(PREDICTION_DIR / "player_state.json")
    profiles = payload.get("players", {})
    query = q.strip().casefold()
    if not profiles:
        return {"available": False, "players": [], "reason": "Train player models first."}
    matches = [
        {
            "player_id": str(player_id),
            "player": profile.get("player", str(player_id)),
            "team_id": profile.get("team_id"),
            "prior_games": profile.get("prior_games", 0),
        }
        for player_id, profile in profiles.items()
        if int(profile.get("prior_games", 0)) >= 3
        and (
            query in str(profile.get("player", "")).casefold()
            or query in str(player_id).casefold()
        )
    ]
    matches.sort(key=lambda item: (str(item["player"]).casefold(), item["player_id"]))
    return {"available": True, "players": matches[:limit]}


@app.get("/api/predictions/historical/games")
def historical_game_predictions(
    team: str | None = Query(default=None),
    season: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """Return saved chronological holdout predictions and actual results."""
    winner_path = PREDICTION_DIR / "historical_game_winner_predictions.csv"
    score_path = PREDICTION_DIR / "historical_game_score_predictions.csv"
    if not winner_path.is_file() or not score_path.is_file():
        return {
            "available": False,
            "games": [],
            "reason": "Historical holdout predictions are unavailable; train models first.",
        }
    try:
        winner = pd.read_csv(winner_path, low_memory=False)
        scores = pd.read_csv(score_path, low_memory=False)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        LOGGER.exception("Could not read historical prediction files")
        raise HTTPException(status_code=500, detail="Historical predictions are unreadable") from error
    frame = winner.merge(
        scores,
        on=["date", "game_id", "home_team", "away_team", "home_score", "away_score"],
        how="inner",
        suffixes=("", "_score"),
    )
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    season_start = dates.dt.year - (dates.dt.month < 10).astype(int)
    frame["season"] = season_start.astype("Int64").astype("string") + "-" + (
        (season_start + 1) % 100
    ).astype("Int64").astype("string").str.zfill(2)
    if team:
        normalized = team.strip().casefold()
        state = _read_model_json(PREDICTION_DIR / "team_strength.json")
        requested_team = state.get("game_team_state", {}).get(team.strip(), {}).get("team")
        labels = {normalized}
        if requested_team:
            labels.add(str(requested_team).casefold())
        frame = frame.loc[
            frame["home_team"].astype(str).str.casefold().isin(labels)
            | frame["away_team"].astype(str).str.casefold().isin(labels)
        ]
    if season:
        frame = frame.loc[frame["season"].eq(season)]
    frame = frame.sort_values("date", ascending=False).head(limit)
    columns = [
        "date", "season", "game_id", "home_team", "away_team",
        "home_score", "away_score", "home_win", "predicted_home_probability",
        "predicted_away_probability", "predicted_home_win", "confidence",
        "predicted_home_score", "predicted_away_score", "actual_total_points",
        "predicted_total_points", "actual_winning_margin", "predicted_winning_margin",
    ]
    available_columns = [column for column in columns if column in frame]
    games = frame[available_columns].astype(object).where(
        pd.notna(frame[available_columns]), None
    ).to_dict(orient="records")
    return {
        "available": True,
        "source": "chronological_test_holdout",
        "filters": {"team": team, "season": season},
        "count": len(games),
        "games": games,
    }


@app.get("/api/models/performance")
def model_performance() -> dict[str, Any]:
    """Return chronological validation and untouched holdout evaluation metrics."""
    payload = _read_model_json(MODEL_EVALUATION_DIR / "model_performance.json")
    if not payload:
        return {"available": False, "reason": "No model evaluation report exists; train models first."}
    return {"available": True, **payload}


@app.get("/api/models/comparison")
def model_comparison() -> dict[str, Any]:
    """Compare candidate algorithms on validation and selected holdout metrics."""
    payload = _read_model_json(MODEL_EVALUATION_DIR / "model_performance.json")
    if not payload:
        return {"available": False, "reason": "No model comparison exists; train models first."}
    winner = payload.get("winner", {})
    scores = payload.get("scores", {})
    return {
        "available": True,
        "selection_rule": "Best validation log loss for winner; lowest validation MAE per score target; test metrics were not used for selection.",
        "winner_models": winner.get("selected_by_validation_log_loss"),
        "winner_candidates": winner.get("validation_candidates", {}),
        "winner_holdout_metrics": winner.get("test_metrics"),
        "score_models": scores.get("selected_models", {}),
        "score_candidates": scores.get("validation_candidates", {}),
        "score_holdout_metrics": scores.get("test_metrics", {}),
    }


@app.get("/api/backtesting/results")
def backtesting_results() -> dict[str, Any]:
    """Return rolling-origin evaluation across multiple historical periods."""
    payload = _read_model_json(MODEL_EVALUATION_DIR / "backtesting_results.json")
    if not payload:
        return {"available": False, "reason": "No rolling-origin backtest is available; train models first."}
    return {"available": True, **payload}


@app.get("/api/datasets/status")
def dataset_status() -> dict[str, Any]:
    """Summarize locally available raw and processed datasets."""
    raw_files = _raw_csv_files()
    processed_files = (
        list(PROCESSED_DATA_DIR.glob("*.parquet"))
        if PROCESSED_DATA_DIR.is_dir()
        else []
    )
    return {
        "raw_csv_count": len(raw_files),
        "processed_parquet_count": len(processed_files),
        "raw_directory_exists": RAW_DATA_DIR.is_dir(),
        "quality_report_available": QUALITY_REPORT_PATH.is_file(),
    }


@app.get("/api/datasets")
def list_datasets() -> dict[str, Any]:
    """List local CSV inputs and their available processing metadata."""
    report_items = _report_datasets()
    datasets = []
    for path in _raw_csv_files():
        item: dict[str, Any] = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "processed": (PROCESSED_DATA_DIR / f"{path.stem}.parquet").is_file(),
        }
        report_item = report_items.get(path.name)
        if report_item:
            item["rows"] = report_item["source_rows"]
            item["columns"] = report_item["columns"]
            item["dataset_kind"] = report_item["dataset_kind"]
        datasets.append(item)
    return {"datasets": datasets}


@app.get("/api/datasets/quality")
def data_quality() -> dict[str, Any]:
    """Return the latest ingestion quality report, if one has been generated."""
    if not QUALITY_REPORT_PATH.is_file():
        return {"available": False, "report": None}
    report_items = _report_datasets()
    return {"available": True, "report": report_items}


@app.get("/api/datasets/{dataset_name}/metadata")
def dataset_metadata(dataset_name: str) -> dict[str, Any]:
    """Return schema and row metadata for a named raw CSV file."""
    if Path(dataset_name).name != dataset_name or not dataset_name.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Provide a CSV filename")
    path = RAW_DATA_DIR / dataset_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Dataset not found")

    report_item = _report_datasets().get(dataset_name)
    if report_item:
        return report_item
    try:
        sample = pd.read_csv(
            path,
            nrows=1000,
            low_memory=False,
            skipinitialspace=True,
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        LOGGER.exception("Could not inspect dataset %s", path.name)
        raise HTTPException(status_code=422, detail="CSV could not be inspected") from error
    return {
        "source": path.name,
        "processed": False,
        "rows": None,
        "columns": list(sample.columns),
        "sample_dtypes": {str(column): str(dtype) for column, dtype in sample.dtypes.items()},
        "sample_rows": len(sample),
    }


@app.post("/api/datasets/upload", status_code=201)
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Save an uploaded CSV into the local raw-data directory."""
    filename = Path(file.filename or "").name
    if not filename or not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DATA_DIR / filename
    if destination.exists():
        raise HTTPException(status_code=409, detail="A file with that name already exists")

    try:
        header = (await file.read(65536)).decode("utf-8-sig")
        if not header.strip():
            raise HTTPException(status_code=400, detail="CSV file is empty")
        first_record = next(csv.reader([header.splitlines()[0]]), [])
        if not first_record or not any(field.strip() for field in first_record):
            raise HTTPException(status_code=400, detail="CSV header is missing")
        await file.seek(0)
        with destination.open("xb") as target:
            while chunk := await file.read(1024 * 1024):
                target.write(chunk)
    except HTTPException:
        raise
    except (OSError, UnicodeError) as error:
        destination.unlink(missing_ok=True)
        LOGGER.exception("Could not save uploaded CSV %s", filename)
        raise HTTPException(status_code=500, detail="CSV upload could not be saved") from error
    finally:
        await file.close()

    LOGGER.info("Uploaded dataset %s (%s bytes)", filename, destination.stat().st_size)
    return {"name": filename, "size_bytes": destination.stat().st_size, "uploaded": True}
