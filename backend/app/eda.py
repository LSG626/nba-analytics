"""Schema-aware exploratory analysis for locally supplied NBA datasets."""

from __future__ import annotations

from functools import lru_cache
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

KEY = re.compile(r"[^a-z0-9]")

METRICS = {
    "points": ("points", "teamscore", "homescore", "awayscore"),
    "rebounds": ("reboundstotal", "rebounds"),
    "assists": ("assists",),
    "steals": ("steals",),
    "blocks": ("blocks",),
    "minutes": ("numminutes", "minutes"),
    "turnovers": ("turnovers",),
    "field_goal_percentage": ("fieldgoalspercentage", "fgpct"),
    "three_point_percentage": ("threepointerspercentage", "3ppct"),
    "free_throw_percentage": ("freethrowspercentage", "ftpct"),
    "win": ("win", "winner"),
}
COMMON_KEYS = {
    "gameid", "personid", "playerid", "teamid", "hometeamid", "awayteamid",
    "playerteamid", "opponentteamid", "gamedatetimeest", "gamedate", "date",
    "season", "seasonid", "seasonyear", "win", "home", "winner",
    "firstname", "lastname", "playername", "teamname", "teamcity",
    "hometeamname", "awayteamname", "hometeamcity", "awayteamcity",
    "playerteamname", "playerteamcity", "opponentteamname",
    "opponentscore", "pointsagainst", "otallpoints",
}


def _key(value: Any) -> str:
    return KEY.sub("", str(value).lower())


def _find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    columns = {_key(column): str(column) for column in frame.columns}
    for alias in aliases:
        if _key(alias) in columns:
            return columns[_key(alias)]
    return None


def _find_metric(frame: pd.DataFrame, metric: str) -> str | None:
    aliases = METRICS.get(metric, (metric,))
    return _find_column(frame, aliases)


def _source_files(directories: list[Path]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {".csv", ".parquet"}:
                continue
            sources.setdefault(path.stem.lower(), path)
    return sources


def _find_source(sources: dict[str, Path], stems: tuple[str, ...]) -> Path | None:
    return next((sources[stem.lower()] for stem in stems if stem.lower() in sources), None)


def _selected_columns(path: Path) -> list[str] | None:
    if path.suffix.lower() != ".csv":
        return None
    header = pd.read_csv(path, nrows=0, low_memory=False)
    columns = [str(column) for column in header.columns]
    wanted = set(COMMON_KEYS)
    for aliases in METRICS.values():
        wanted.update(_key(alias) for alias in aliases)
    # Include numeric-looking feature columns for exploratory correlations.
    return [
        column for column in columns
        if _key(column) in wanted
        or any(token in _key(column) for token in (
            "percentage", "rating", "pace", "possession", "q1points", "q2points",
            "q3points", "q4points", "ot", "fieldgoals", "threepointers",
            "freethrows", "rebounds", "points", "turnovers", "assists",
            "steals", "blocks", "minutes",
        ))
    ]


@lru_cache(maxsize=12)
def _read_cached(path_text: str, _modified_ns: int, _size: int) -> pd.DataFrame:
    path = Path(path_text)
    stat = path.stat()
    if stat.st_mtime_ns != _modified_ns or stat.st_size != _size:
        raise RuntimeError(f"Dataset changed while it was being read: {path.name}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    selected = _selected_columns(path)
    return pd.read_csv(path, usecols=selected, low_memory=False)


def _read(path: Path) -> pd.DataFrame:
    stat = path.stat()
    return _read_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size).copy(deep=False)


def _season_series(frame: pd.DataFrame) -> pd.Series:
    explicit = _find_column(frame, ("season", "seasonId", "seasonYear"))
    if explicit:
        return frame[explicit].astype("string")
    date_column = _find_column(frame, ("gameDate", "gameDateTimeEst", "date", "timestamp"))
    if not date_column:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
    start_year = dates.dt.year.where(dates.dt.month >= 10, dates.dt.year - 1)
    end_year = start_year + 1
    return start_year.astype("Int64").astype("string") + "-" + (
        end_year.mod(100).astype("Int64").astype("string").str.zfill(2)
    )


def _date_series(frame: pd.DataFrame) -> pd.Series | None:
    column = _find_column(frame, ("gameDate", "gameDateTimeEst", "date", "timestamp"))
    return pd.to_datetime(frame[column], errors="coerce", utc=True) if column else None


def _matches(series: pd.Series, value: str | None) -> pd.Series:
    if not value:
        return pd.Series(True, index=series.index)
    return series.astype("string").str.contains(re.escape(value), case=False, na=False)


def _filter_frame(
    frame: pd.DataFrame,
    *,
    season: str | None = None,
    team: str | None = None,
    player: str | None = None,
    kind: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = pd.Series(True, index=frame.index)
    applied: list[str] = []
    unsupported: list[str] = []
    if season:
        values = _season_series(frame)
        if values.notna().any():
            mask &= values.str.contains(re.escape(season), case=False, na=False)
            applied.append("season")
        else:
            unsupported.append("season")
    if team:
        aliases = (
            ("teamId", "homeTeamId", "awayTeamId", "teamName", "teamCity",
             "homeTeamName", "awayTeamName", "playerTeamId", "opponentTeamId",
             "playerTeamName", "playerTeamCity")
            if kind in {"games", "teams", "players"} else ()
        )
        columns = [_find_column(frame, (alias,)) for alias in aliases]
        columns = [column for column in columns if column]
        if columns:
            matches = pd.Series(False, index=frame.index)
            for column in columns:
                matches |= _matches(frame[column], team)
            if kind == "players":
                matches |= _matches(_team_labels(frame), team)
            mask &= matches
            applied.append("team")
        else:
            unsupported.append("team")
    if player:
        id_column = _find_column(frame, ("personId", "playerId"))
        name_columns = [
            column for column in (
                _find_column(frame, ("playerName",)),
                _find_column(frame, ("firstName",)),
                _find_column(frame, ("lastName",)),
            ) if column
        ]
        if id_column or name_columns:
            matches = _matches(frame[id_column], player) if id_column else pd.Series(
                False, index=frame.index
            )
            for column in name_columns:
                matches |= _matches(frame[column], player)
            if name_columns:
                matches |= _matches(_player_labels(frame), player)
            mask &= matches
            applied.append("player")
        else:
            unsupported.append("player")
    return frame.loc[mask].copy(), {
        "requested": {"season": season, "team": team, "player": player},
        "applied": applied,
        "unsupported": unsupported,
    }


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _stats_frame(
    sources: dict[str, Path], stems: tuple[str, ...]
) -> tuple[pd.DataFrame | None, Path | None]:
    path = _find_source(sources, stems)
    return (_read(path), path) if path else (None, None)


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _metric_scale(metric: str, series: pd.Series) -> float:
    if metric not in {
        "field_goal_percentage", "three_point_percentage", "free_throw_percentage"
    }:
        return 1.0
    values = pd.to_numeric(series, errors="coerce").dropna()
    return 100.0 if not values.empty and values.abs().max() <= 1 else 1.0


def _dataset_metadata(sources: dict[str, Path]) -> list[dict[str, Any]]:
    result = []
    for path in sorted(sources.values()):
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            columns = parquet.schema.names
            rows = parquet.metadata.num_rows
        else:
            columns = [str(column) for column in pd.read_csv(path, nrows=0).columns]
            rows = None
        result.append({
            "name": path.name,
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
            "row_count": rows,
            "row_count_status": "exact" if rows is not None else "not_scanned",
            "column_count": len(columns),
            "columns": columns,
        })
    return result


def _team_labels(frame: pd.DataFrame) -> pd.Series:
    name = _find_column(frame, ("teamName", "playerTeamName", "hometeamName"))
    city = _find_column(frame, ("teamCity", "playerteamCity", "hometeamCity"))
    identifier = _find_column(frame, ("teamId", "playerTeamId", "homeTeamId"))
    if name and city:
        return frame[city].astype("string").str.cat(frame[name].astype("string"), sep=" ")
    if name:
        return frame[name].astype("string")
    return frame[identifier].astype("string") if identifier else pd.Series("Unknown", index=frame.index)


def _player_labels(frame: pd.DataFrame) -> pd.Series:
    first = _find_column(frame, ("firstName",))
    last = _find_column(frame, ("lastName",))
    name = _find_column(frame, ("playerName",))
    if first and last:
        return (frame[first].astype("string") + " " + frame[last].astype("string")).str.strip()
    if name:
        return frame[name].astype("string")
    identifier = _find_column(frame, ("personId", "playerId"))
    return frame[identifier].astype("string") if identifier else pd.Series("Unknown", index=frame.index)


def overview(directories: list[Path]) -> dict[str, Any]:
    sources = _source_files(directories)
    games, games_path = _stats_frame(sources, ("games",))
    teams, teams_path = _stats_frame(sources, ("teamstatisticsextended", "teamstatistics"))
    players, players_path = _stats_frame(sources, ("playerstatisticsextended", "playerstatistics"))
    roster, roster_path = _stats_frame(sources, ("players",))
    game_seasons: list[str] = []
    dates: list[pd.Series] = []
    game_count = 0
    team_count: int | None = None
    player_count: int | None = None
    player_statistics_count: int | None = None
    if games is not None:
        game_count = len(games)
        game_seasons = sorted(str(value) for value in _season_series(games).dropna().unique())
        date_values = _date_series(games)
        if date_values is not None:
            dates.append(date_values)
        home_id = _find_column(games, ("homeTeamId",))
        away_id = _find_column(games, ("awayTeamId",))
        if home_id and away_id:
            team_count = int(pd.concat([games[home_id], games[away_id]]).dropna().nunique())
    if teams is not None and team_count is None:
        team_id = _find_column(teams, ("teamId",))
        if team_id:
            team_count = int(teams[team_id].nunique(dropna=True))
    if roster is not None:
        player_id = _find_column(roster, ("personId", "playerId"))
        player_count = int(roster[player_id].nunique(dropna=True)) if player_id else len(roster)
    if players is not None:
        player_id = _find_column(players, ("personId", "playerId"))
        player_statistics_count = int(players[player_id].nunique(dropna=True)) if player_id else None
        if player_count is None:
            player_count = player_statistics_count
    if players is not None:
        player_seasons = _season_series(players).dropna().astype(str).unique()
        game_seasons = sorted(set(game_seasons) | set(player_seasons))
        date_values = _date_series(players)
        if date_values is not None:
            dates.append(date_values)

    table_summaries: dict[str, Any] = {}
    for label, frame in (("games", games), ("teams", teams), ("players", players), ("roster", roster)):
        if frame is None:
            table_summaries[label] = {"available": False}
            continue
        missing = frame.isna().sum()
        numeric = frame.select_dtypes(include="number")
        table_summaries[label] = {
            "available": True,
            "rows": len(frame),
            "columns": len(frame.columns),
            "column_scope": "selected analysis fields",
            "missing_values": {str(column): int(count) for column, count in missing.items() if count},
            "missing_cells": int(missing.sum()),
            "completeness_percent": (
                100.0 * (frame.size - int(missing.sum())) / frame.size if frame.size else None
            ),
            "duplicate_rows": int(frame.duplicated().sum()),
            "numeric_distributions": {
                str(column): _numeric_summary(numeric[column])
                for column in numeric.columns[:20]
            },
        }
    selected_paths = {
        path.name: table
        for path, table in (
            (games_path, "games"), (teams_path, "teams"),
            (players_path, "players"), (roster_path, "roster"),
        ) if path is not None
    }
    dataset_metadata = _dataset_metadata(sources)
    for dataset in dataset_metadata:
        table_name = selected_paths.get(dataset["name"])
        if table_name and table_summaries[table_name].get("available"):
            dataset["row_count"] = table_summaries[table_name]["rows"]
            dataset["row_count_status"] = "exact"
            dataset["duplicate_rows"] = table_summaries[table_name]["duplicate_rows"]
    all_dates = pd.concat(dates).dropna() if dates else pd.Series(dtype="datetime64[ns, UTC]")
    return _clean({
        "datasets": dataset_metadata,
        "counts": {
            "games": game_count if games is not None else None,
            "seasons": len(game_seasons) if game_seasons else None,
            "season_labels": game_seasons,
            "teams": team_count,
            "players": player_count,
            "players_in_statistics": player_statistics_count,
        },
        "date_coverage": {
            "start": all_dates.min() if not all_dates.empty else None,
            "end": all_dates.max() if not all_dates.empty else None,
        },
        "tables": table_summaries,
        "limitations": [
            "Exact row counts, missingness, duplicate rows and distributions describe the loaded analysis tables only.",
            "Additional discovered files expose schema and size; unscanned row counts and data-quality metrics are explicitly unavailable.",
            "The season label is derived from game dates (October starts a season) only when no season field exists.",
        ],
    })


def games_analysis(
    directories: list[Path], season: str | None = None, team: str | None = None
) -> dict[str, Any]:
    sources = _source_files(directories)
    frame, path = _stats_frame(sources, ("games",))
    if frame is None:
        return {"available": False, "reason": "No Games dataset was found."}
    frame, filters = _filter_frame(frame, season=season, team=team, kind="games")
    home_score = _find_column(frame, ("homeScore",))
    away_score = _find_column(frame, ("awayScore",))
    home_id = _find_column(frame, ("homeTeamId",))
    away_id = _find_column(frame, ("awayTeamId",))
    winner = _find_column(frame, ("winner",))
    result: dict[str, Any] = {
        "available": True,
        "source": path.name if path else None,
        "filters": filters,
        "game_count": len(frame),
        "scored_game_count": 0,
        "supported_metrics": [],
        "unavailable_analyses": [],
    }
    if home_score and away_score:
        home = pd.to_numeric(frame[home_score], errors="coerce")
        away = pd.to_numeric(frame[away_score], errors="coerce")
        valid_scores = home.gt(0) & away.gt(0)
        valid = frame.loc[valid_scores].copy()
        home = home.loc[valid_scores]
        away = away.loc[valid_scores]
        total = home + away
        margin = home - away
        result["scored_game_count"] = len(valid)
        result["invalid_or_missing_score_rows"] = int((~valid_scores).sum())
        total_values = total.dropna()
        histogram_counts, histogram_edges = np.histogram(
            total_values, bins=min(20, max(1, int(np.sqrt(len(total_values)))))
        ) if len(total_values) else (np.array([]), np.array([]))
        result.update({
            "supported_metrics": ["average_points_per_team", "winning_margin", "highest_scoring_games", "lowest_scoring_games", "home_away_win_rate"],
            "average_points_per_team": float(pd.concat([home, away]).mean()) if len(valid) else None,
            "average_total_points": float(total.mean()),
            "average_winning_margin": float(margin.abs().mean()),
            "home_win_percentage": float((home > away).mean() * 100) if len(valid) else None,
            "home_away_win_distribution": {
                "home_wins": int((home > away).sum()),
                "away_wins": int((away > home).sum()),
                "ties": int((away == home).sum()),
            },
            "score_distribution": {
                "bin_centers": ((histogram_edges[:-1] + histogram_edges[1:]) / 2).tolist(),
                "counts": histogram_counts.tolist(),
            },
            "score_boxplot": {
                "home_points": home.dropna().iloc[::max(1, (len(home) + 1999) // 2000)].tolist(),
                "away_points": away.dropna().iloc[::max(1, (len(away) + 1999) // 2000)].tolist(),
            },
        })
        valid["_total"] = total.loc[valid.index]
        valid["_margin"] = margin.loc[valid.index].abs()
        fields = [
            col for col in (
                _find_column(valid, ("gameId",)),
                _find_column(valid, ("gameDate", "gameDateTimeEst")),
                _find_column(valid, ("homeTeamName", "hometeamName")),
                _find_column(valid, ("awayTeamName", "awayteamName")),
                home_score, away_score,
            ) if col
        ]
        result["highest_scoring_games"] = valid.nlargest(10, "_total")[fields + ["_total"]].rename(
            columns={"_total": "total_points"}
        ).to_dict(orient="records")
        result["lowest_scoring_games"] = valid.nsmallest(10, "_total")[fields + ["_total"]].rename(
            columns={"_total": "total_points"}
        ).to_dict(orient="records")
        result["scoring_by_season"] = (
            pd.DataFrame({
                "season": _season_series(valid),
                "points": pd.concat([home, away], axis=1).mean(axis=1),
            }).groupby("season", dropna=True)["points"].mean()
            .reset_index().to_dict(orient="records")
        )
        season_col = _find_column(frame, ("season", "seasonId", "seasonYear"))
        if not season_col:
            result["scoring_by_season"] = [
                {"season": str(row["season"]), "average_points_per_team": row["points"]}
                for row in result["scoring_by_season"]
            ]
        if winner and home_id and away_id:
            result["winner_field_available"] = True
        else:
            result["winner_field_available"] = False
        if not len(valid):
            result["unavailable_analyses"].append(
                "No valid completed games with positive home and away scores were found."
            )
    else:
        result["unavailable_analyses"].append("Scoring and game outcomes require both homeScore and awayScore.")
    team_stats, team_path = _stats_frame(sources, ("teamstatisticsextended", "teamstatistics"))
    if team_stats is not None:
        team_stats, _ = _filter_frame(team_stats, season=season, team=team, kind="teams")
        ot_columns = [column for column in team_stats.columns if _key(column).startswith("ot") and "points" in _key(column)]
        if ot_columns:
            overtime_rows = team_stats[ot_columns].apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0).sum(axis=1) > 0
            game_id = _find_column(team_stats, ("gameId",))
            overtime = (
                team_stats.assign(_overtime=overtime_rows)
                .groupby(game_id, dropna=False)["_overtime"].any()
                if game_id else overtime_rows
            )
            result["overtime"] = {
                "source": team_path.name,
                "games_with_overtime": int(overtime.sum()),
                "games_observed": len(overtime),
                "frequency_percent": float(overtime.mean() * 100) if len(overtime) else None,
            }
        else:
            result["unavailable_analyses"].append("Overtime frequency is unavailable: no overtime point columns were found.")
    else:
        result["unavailable_analyses"].append("Overtime frequency requires a team statistics table with overtime fields.")
    return _clean(result)


def teams_analysis(
    directories: list[Path], season: str | None = None, team: str | None = None
) -> dict[str, Any]:
    sources = _source_files(directories)
    frame, path = _stats_frame(sources, ("teamstatisticsextended", "teamstatistics"))
    if frame is None:
        return {"available": False, "reason": "No team statistics dataset was found."}
    frame, filters = _filter_frame(frame, season=season, team=team, kind="teams")
    if frame.empty:
        return {"available": True, "source": path.name if path else None, "filters": filters, "teams": [], "game_count": 0}
    frame["_team_name"] = _team_labels(frame)
    team_id_col = _find_column(frame, ("teamId", "playerTeamId"))
    frame["_team"] = (
        frame[team_id_col].astype("string").fillna(frame["_team_name"])
        if team_id_col else frame["_team_name"]
    )
    date_values = _date_series(frame)
    labels = frame.groupby("_team", dropna=True)["_team_name"].agg(
        lambda values: values.dropna().value_counts().index[0]
        if values.notna().any() else None
    )
    labels = labels.where(labels.notna(), labels.index.to_series(index=labels.index))
    if date_values is not None:
        frame["_date"] = date_values
        latest_labels = (
            frame.dropna(subset=["_team_name"])
            .sort_values("_date")
            .drop_duplicates("_team", keep="last")
            .set_index("_team")["_team_name"]
        )
        labels.update(latest_labels)
    win_col = _find_column(frame, ("win",))
    score_col = _find_column(frame, ("teamScore", "points"))
    against_col = _find_column(frame, ("opponentScore", "pointsAgainst"))
    home_col = _find_column(frame, ("home",))
    frame["_season"] = _season_series(frame)
    if win_col:
        win_text = frame[win_col].astype("string").str.lower()
        win_values = win_text.map({
            "true": 1.0, "1": 1.0, "1.0": 1.0, "yes": 1.0, "w": 1.0,
            "false": 0.0, "0": 0.0, "0.0": 0.0, "no": 0.0, "l": 0.0,
        })
        numeric_win = pd.to_numeric(frame[win_col], errors="coerce")
        frame["_win"] = win_values.where(
            win_values.notna(), numeric_win.where(numeric_win.isin([0, 1]))
        )
    elif score_col and against_col:
        scored = pd.to_numeric(frame[score_col], errors="coerce")
        conceded = pd.to_numeric(frame[against_col], errors="coerce")
        frame["_win"] = (
            (scored > conceded).astype(float).where(scored.notna() & conceded.notna())
        )
    else:
        frame["_win"] = np.nan
    if date_values is not None:
        frame["_date"] = date_values
    metrics: dict[str, str] = {}
    for metric in ("points", "assists", "rebounds", "steals", "blocks", "turnovers"):
        column = _find_metric(frame, metric)
        if column:
            metrics[metric] = column
    groups = []
    for identity, group in frame.groupby("_team", dropna=True, sort=True):
        label = labels.get(identity, str(identity))
        win_rate = group["_win"].mean() * 100 if group["_win"].notna().any() else None
        item: dict[str, Any] = {
            "team": str(label),
            "games": len(group),
            "wins": int(group["_win"].sum()) if group["_win"].notna().any() else None,
            "win_percentage": win_rate,
            "average_points_scored": float(pd.to_numeric(group[score_col], errors="coerce").mean()) if score_col else None,
            "average_points_conceded": float(pd.to_numeric(group[against_col], errors="coerce").mean()) if against_col else None,
            "point_scoring_consistency_std": float(pd.to_numeric(group[score_col], errors="coerce").std()) if score_col else None,
        }
        if team_id_col and pd.notna(group[team_id_col].iloc[0]):
            item["team_id"] = str(group[team_id_col].iloc[0])
        item["metrics"] = {
            key: _numeric_summary(group[column]) for key, column in metrics.items()
        }
        if home_col:
            home_text = group[home_col].astype("string").str.lower()
            home_values = home_text.isin({"true", "1", "1.0", "yes", "home"})
            away_values = home_text.isin({"false", "0", "0.0", "no", "away"})
            item["home_games"] = int(home_values.sum())
            item["away_games"] = int(away_values.sum())
            item["home_win_percentage"] = (
                float(group.loc[home_values, "_win"].mean() * 100)
                if group.loc[home_values, "_win"].notna().any() else None
            )
            item["away_win_percentage"] = (
                float(group.loc[away_values, "_win"].mean() * 100)
                if group.loc[away_values, "_win"].notna().any() else None
            )
        if "_date" in group:
            dated = group.loc[group["_date"].notna()].sort_values("_date")
            if not dated.empty:
                chronological = dated["_win"].dropna().astype(int).tolist()
                current = longest_win = longest_loss = 0
                prev: int | None = None
                for outcome in chronological:
                    current = current + 1 if outcome == prev else 1
                    if outcome:
                        longest_win = max(longest_win, current)
                    else:
                        longest_loss = max(longest_loss, current)
                    prev = outcome
                item["longest_winning_streak"] = longest_win
                item["longest_losing_streak"] = longest_loss
                item["streaks_available"] = bool(chronological)
                item["streaks_excluded_undated_games"] = int(group["_date"].isna().sum())
            else:
                item["longest_winning_streak"] = None
                item["longest_losing_streak"] = None
                item["streaks_available"] = False
        groups.append(item)
    frame["_team_label"] = frame["_team"].map(labels)
    season_summaries = (
        frame.groupby(["_season", "_team_label"], dropna=True)
        .agg(games=("_win", "size"), win_percentage=("_win", "mean"))
        .reset_index()
    )
    season_summaries["win_percentage"] *= 100
    unavailable = []
    if not win_col and not (score_col and against_col):
        unavailable.append("Win percentage requires a win flag or both team and opponent scores.")
    if not score_col:
        unavailable.append("Offensive points analysis is unavailable: no team score field.")
    if not against_col:
        unavailable.append("Defensive points analysis is unavailable: no opponent score field.")
    if not home_col:
        unavailable.append("Home/away split is unavailable: no home indicator.")
    return _clean({
        "available": True,
        "source": path.name if path else None,
        "filters": filters,
        "team_count": len(groups),
        "game_count": len(frame),
        "teams": sorted(groups, key=lambda row: (row["win_percentage"] is None, -(row["win_percentage"] or 0))),
        "season_performance": season_summaries.rename(
            columns={"_season": "season", "_team_label": "team"}
        ).to_dict(orient="records"),
        "chart_data": {
            "teams": [row["team"] for row in groups],
            "win_percentage": [row["win_percentage"] for row in groups],
            "average_points_scored": [row["average_points_scored"] for row in groups],
            "average_points_conceded": [row["average_points_conceded"] for row in groups],
            "home_win_percentage": [row.get("home_win_percentage") for row in groups],
            "away_win_percentage": [row.get("away_win_percentage") for row in groups],
        },
        "supported_metrics": list(metrics),
        "unavailable_analyses": unavailable,
    })


def players_analysis(
    directories: list[Path],
    season: str | None = None,
    team: str | None = None,
    player: str | None = None,
) -> dict[str, Any]:
    sources = _source_files(directories)
    frame, path = _stats_frame(sources, ("playerstatisticsextended", "playerstatistics"))
    if frame is None:
        return {"available": False, "reason": "No player statistics dataset was found."}
    frame, filters = _filter_frame(frame, season=season, team=team, player=player, kind="players")
    if frame.empty:
        return {"available": True, "source": path.name if path else None, "filters": filters, "players": [], "game_count": 0}
    frame["_player_name"] = _player_labels(frame)
    player_id_col = _find_column(frame, ("personId", "playerId"))
    frame["_player"] = (
        frame[player_id_col].astype("string").fillna(frame["_player_name"])
        if player_id_col else frame["_player_name"]
    )
    player_names = frame.groupby("_player", dropna=True)["_player_name"].agg(
        lambda values: values.dropna().value_counts().index[0]
        if values.notna().any() else None
    )
    player_names = player_names.where(
        player_names.notna(), player_names.index.to_series(index=player_names.index)
    )
    frame["_season"] = _season_series(frame)
    metric_columns = {
        metric: column for metric in (
            "points", "rebounds", "assists", "steals", "blocks", "minutes",
            "field_goal_percentage", "three_point_percentage", "free_throw_percentage",
            "turnovers", "win",
        ) if (column := _find_metric(frame, metric))
    }
    metric_values = pd.DataFrame(index=frame.index)
    for metric, column in metric_columns.items():
        metric_values[metric] = (
            pd.to_numeric(frame[column], errors="coerce")
            * _metric_scale(metric, frame[column])
        )
    player_groups = metric_values.groupby(frame["_player"], sort=True)
    player_means = player_groups.mean()
    player_deviations = player_groups.std()
    player_counts = frame.groupby("_player", sort=True).size()
    team_col = _find_column(frame, ("playerTeamName", "playerTeamCity", "teamName"))
    player_teams = (
        frame.groupby("_player", sort=True)[team_col].agg(
            lambda values: values.dropna().iloc[-1] if values.notna().any() else None
        ) if team_col else None
    )
    per_player = []
    for identity, means in player_means.iterrows():
        deviations = player_deviations.loc[identity]
        item: dict[str, Any] = {
            "player": str(player_names.get(identity, str(identity))),
            "games": int(player_counts.loc[identity]),
            "metrics": {
                metric: float(means[metric]) if pd.notna(means[metric]) else None
                for metric in metric_columns
            },
            "consistency": {
                metric: float(deviations[metric]) if pd.notna(deviations[metric]) else None
                for metric in metric_columns
            },
        }
        if player_id_col:
            item["player_id"] = str(identity)
        if player_teams is not None and pd.notna(player_teams.loc[identity]):
            item["team"] = str(player_teams.loc[identity])
        per_player.append(item)
    leaders = {
        metric: sorted(
            [row for row in per_player if row["metrics"].get(metric) is not None and row["games"] >= 1],
            key=lambda row: row["metrics"][metric],
            reverse=True,
        )[:10]
        for metric in metric_columns if metric != "win"
    }
    season_col = pd.DataFrame({"season": frame["_season"], "player": frame["_player"]})
    for metric, column in metric_columns.items():
        season_col[metric] = (
            pd.to_numeric(frame[column], errors="coerce")
            * _metric_scale(metric, frame[column])
        )
    season_metrics = {
        metric: "mean" for metric in metric_columns if metric != "win"
    }
    if season_metrics:
        season_performance = (
            season_col.groupby(["season", "player"], dropna=True)
            .agg(season_metrics)
            .reset_index()
        )
        season_performance["player"] = season_performance["player"].map(player_names)
    else:
        season_performance = pd.DataFrame(columns=["season", "player"])
    unavailable = [
        f"{label} per game unavailable: corresponding statistic column was not found."
        for label, metric in (
            ("Points", "points"), ("Rebounds", "rebounds"), ("Assists", "assists"),
            ("Steals", "steals"), ("Blocks", "blocks"), ("Minutes", "minutes"),
            ("Field goal percentage", "field_goal_percentage"),
            ("Three-point percentage", "three_point_percentage"),
            ("Free throw percentage", "free_throw_percentage"),
        ) if metric not in metric_columns
    ]
    return _clean({
        "available": True,
        "source": path.name if path else None,
        "filters": filters,
        "game_count": len(frame),
        "player_count": len(per_player),
        "supported_metrics": list(metric_columns),
        "unavailable_analyses": unavailable,
        "players": sorted(per_player, key=lambda row: row["player"].casefold()),
        "leaders": leaders,
        "season_performance": season_performance.rename(
            columns={"season": "season", "player": "player"}
        ).to_dict(orient="records"),
        "chart_data": {
            "players": [
                row["player"] for row in leaders.get("points", [])[:20]
            ],
            "points": [
                row["metrics"]["points"] for row in leaders.get("points", [])[:20]
            ],
            "rebounds": [
                row["metrics"].get("rebounds") for row in leaders.get("points", [])[:20]
            ],
            "assists": [
                row["metrics"].get("assists") for row in leaders.get("points", [])[:20]
            ],
            "points_boxplot": [
                row["metrics"]["points"] for row in per_player
                if row["metrics"].get("points") is not None
            ],
        },
    })


def correlations_analysis(
    directories: list[Path], season: str | None = None, team: str | None = None, player: str | None = None
) -> dict[str, Any]:
    sources = _source_files(directories)
    output: dict[str, Any] = {
        "available": False,
        "filters": {
            "requested": {"season": season, "team": team, "player": player},
            "tables": {},
        },
        "correlations": {},
        "relationships": {},
        "outliers": {},
        "unavailable_analyses": [],
    }
    for label, stems, kind in (
        ("teams", ("teamstatisticsextended", "teamstatistics"), "teams"),
        ("players", ("playerstatisticsextended", "playerstatistics"), "players"),
    ):
        frame, path = _stats_frame(sources, stems)
        if frame is None:
            output["unavailable_analyses"].append(f"{label.title()} correlation analysis unavailable: dataset not found.")
            continue
        frame, filter_info = _filter_frame(
            frame, season=season, team=team, player=player, kind=kind
        )
        output["filters"]["tables"][label] = filter_info
        if frame.empty:
            output["unavailable_analyses"].append(
                f"No {label} records match the requested filters."
            )
            continue
        output["available"] = True
        numeric = frame.select_dtypes(include="number")
        # Convert provider fields such as win flags when pandas inferred text.
        for metric in ("points", "rebounds", "assists", "steals", "blocks", "turnovers", "win"):
            column = _find_metric(frame, metric)
            if column and column not in numeric.columns:
                converted = pd.to_numeric(frame[column], errors="coerce")
                if converted.notna().any():
                    numeric[column] = converted
        selected = [
            column for column in numeric.columns
            if _key(column) not in {"gameid", "personid", "playerid", "teamid", "opponentteamid"}
        ]
        correlation = numeric[selected].corr(min_periods=2)
        output["correlations"][label] = {
            "source": path.name,
            "columns": list(correlation.columns),
            "matrix": correlation.replace([np.inf, -np.inf], np.nan).to_numpy().tolist(),
            "observations": int(len(frame)),
        }
        dependent = _find_metric(frame, "win")
        if dependent:
            win_raw = frame[dependent]
            win = pd.to_numeric(win_raw, errors="coerce")
            if win.notna().sum() == 0:
                win = win_raw.astype("string").str.lower().map(
                    {"true": 1, "yes": 1, "w": 1, "false": 0, "no": 0, "l": 0}
                )
            pairs: dict[str, Any] = {}
            for metric in ("points", "rebounds", "turnovers"):
                column = _find_metric(frame, metric)
                if column:
                    pair = pd.DataFrame({
                        "metric": pd.to_numeric(frame[column], errors="coerce"),
                        "win": win,
                    }).dropna()
                    step = max(1, (len(pair) + 499) // 500)
                    pairs[metric] = {
                        "correlation_with_win": float(pair["metric"].corr(pair["win"]))
                        if len(pair) > 1 else None,
                        "observations": len(pair),
                        "scatter": {
                            "x": pair["metric"].iloc[::step].tolist(),
                            "y": pair["win"].iloc[::step].tolist(),
                        },
                    }
            output["relationships"][label] = pairs
        if label == "players":
            points = _find_metric(frame, "points")
            rebounds = _find_metric(frame, "rebounds")
            assists = _find_metric(frame, "assists")
            for name, right in (("points_rebounds", rebounds), ("points_assists", assists)):
                if points and right:
                    pair = pd.DataFrame({
                        "x": pd.to_numeric(frame[points], errors="coerce"),
                        "y": pd.to_numeric(frame[right], errors="coerce"),
                    }).dropna()
                    step = max(1, (len(pair) + 999) // 1000)
                    output.setdefault("player_relationships", {})[name] = {
                        "x_label": points,
                        "y_label": right,
                        "x": pair["x"].iloc[::step].tolist(),
                        "y": pair["y"].iloc[::step].tolist(),
                        "observations": len(pair),
                    }
        if selected:
            numeric_values = numeric[selected]
            fences = {}
            for column in selected:
                values = numeric_values[column].dropna()
                if values.empty:
                    continue
                q1, q3 = values.quantile([0.25, 0.75])
                spread = q3 - q1
                fences[str(column)] = {
                    "lower": float(q1 - 1.5 * spread),
                    "upper": float(q3 + 1.5 * spread),
                    "outlier_count": int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum()),
                }
            output["outliers"][label] = fences
    return _clean(output)


def seasons_analysis(
    directories: list[Path],
    season: str | None = None,
    team: str | None = None,
    player: str | None = None,
) -> dict[str, Any]:
    games_result = games_analysis(directories, season=season, team=team)
    teams_result = teams_analysis(directories, season=season, team=team)
    player_result = players_analysis(
        directories, season=season, team=team, player=player
    ) if player else None
    return _clean({
        "available": bool(games_result.get("available") or teams_result.get("available")),
        "selected_season": season,
        "filters": {
            "requested": {"season": season, "team": team, "player": player},
            "games": games_result.get("filters"),
            "teams": teams_result.get("filters"),
            "players": player_result.get("filters") if player_result else None,
        },
        "game_scoring_by_season": games_result.get("scoring_by_season", []),
        "team_performance_by_season": teams_result.get("season_performance", []),
        "game_count_in_scope": games_result.get("game_count"),
        "team_count_in_scope": teams_result.get("team_count"),
        "player_performance_by_season": (
            player_result.get("season_performance", []) if player_result else []
        ),
        "unavailable_analyses": [
            *games_result.get("unavailable_analyses", []),
            *teams_result.get("unavailable_analyses", []),
            *(
                player_result.get("unavailable_analyses", [])
                if player_result and not player_result.get("available") else []
            ),
        ],
    })
