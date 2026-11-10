"""Chronological, leakage-safe feature builders for NBA games and players."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

KEY = re.compile(r"[^a-z0-9]")
GAME_FEATURES = [
    "home_is_home",
    "home_prior_games",
    "away_prior_games",
    "home_win_rate_all",
    "away_win_rate_all",
    "home_win_rate_last5",
    "away_win_rate_last5",
    "home_win_rate_last10",
    "away_win_rate_last10",
    "home_points_for_last5",
    "away_points_for_last5",
    "home_points_against_last5",
    "away_points_against_last5",
    "home_points_for_all",
    "away_points_for_all",
    "home_points_against_all",
    "away_points_against_all",
    "home_rest_days",
    "away_rest_days",
    "home_home_win_rate",
    "away_away_win_rate",
    "head_to_head_home_win_rate",
    "head_to_head_games",
    "home_elo",
    "away_elo",
    "elo_difference",
]
PLAYER_TARGETS = {
    "minutes": ("numMinutes", "minutes"),
    "points": ("points",),
    "rebounds": ("reboundsTotal", "rebounds"),
    "assists": ("assists",),
    "steals": ("steals",),
    "blocks": ("blocks",),
    "three_pointers_made": ("threePointersMade", "threePointFieldGoalsMade"),
}
PLAYER_FEATURES = ["home_is_home", "prior_games", "rest_days"]
for _metric in PLAYER_TARGETS:
    PLAYER_FEATURES.extend(
        [f"prior_{_metric}_mean5", f"prior_{_metric}_mean_all", f"prior_{_metric}_std5"]
    )
PLAYER_FEATURES.extend(["prior_matchup_points_mean", "prior_matchup_games"])


def _key(value: Any) -> str:
    return KEY.sub("", str(value).lower())


def _column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    columns = {_key(value): str(value) for value in frame.columns}
    for alias in aliases:
        if _key(alias) in columns:
            return columns[_key(alias)]
    return None


def _source(directories: list[Path], stem: str) -> Path | None:
    for directory in directories:
        if not directory.is_dir():
            continue
        for suffix in (".parquet", ".csv"):
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if columns is not None:
        return pd.read_csv(path, usecols=columns, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _column_names(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema.names
    return pd.read_csv(path, nrows=0, low_memory=False).columns.tolist()


def _team_key(identifier: Any, label: Any) -> str | None:
    if pd.notna(identifier) and str(identifier).strip():
        return str(identifier).strip()
    if pd.notna(label) and str(label).strip():
        return str(label).strip().casefold()
    return None


def build_game_frame(directories: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one row per valid completed game using only strictly prior games."""
    path = _source(directories, "Games")
    if path is None:
        return pd.DataFrame(), {"available": False, "reason": "Games table was not found."}
    header = _column_names(path)
    aliases = {
        "game_id": ("gameId",),
        "date": ("gameDateTimeEst", "gameDate", "date"),
        "home_team_id": ("hometeamId", "homeTeamId"),
        "away_team_id": ("awayteamId", "awayTeamId"),
        "home_team": ("hometeamName", "homeTeamName"),
        "away_team": ("awayteamName", "awayTeamName"),
        "home_score": ("homeScore",),
        "away_score": ("awayScore",),
        "game_type": ("gameType",),
    }
    selected: dict[str, str] = {}
    for output, names in aliases.items():
        actual = _column(pd.DataFrame(columns=header), names)
        if actual:
            selected[output] = actual
    required = {"date", "home_score", "away_score"}
    if not required.issubset(selected):
        return pd.DataFrame(), {
            "available": False,
            "reason": f"Games table lacks required fields: {sorted(required - set(selected))}",
            "source": path.name,
        }
    source = _read(path, list(dict.fromkeys(selected.values())))
    frame = pd.DataFrame(index=source.index)
    for output, column in selected.items():
        frame[output] = source[column]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    frame = frame.dropna(subset=["date", "home_score", "away_score"])
    frame = frame.loc[
        (frame.home_score > 0)
        & (frame.away_score > 0)
        & (frame.home_score != frame.away_score)
    ].copy()
    frame["home_key"] = [
        _team_key(identifier, label)
        for identifier, label in zip(
            frame.get("home_team_id", pd.Series(pd.NA, index=frame.index)),
            frame.get("home_team", pd.Series(pd.NA, index=frame.index)),
        )
    ]
    frame["away_key"] = [
        _team_key(identifier, label)
        for identifier, label in zip(
            frame.get("away_team_id", pd.Series(pd.NA, index=frame.index)),
            frame.get("away_team", pd.Series(pd.NA, index=frame.index)),
        )
    ]
    frame = frame.dropna(subset=["home_key", "away_key"])
    if frame.empty:
        return frame, {
            "available": False,
            "reason": "No completed games have usable home and away team identifiers.",
            "source": path.name,
        }
    if "home_team" not in frame:
        frame["home_team"] = frame["home_key"]
    else:
        frame["home_team"] = frame["home_team"].fillna(frame["home_key"])
    if "away_team" not in frame:
        frame["away_team"] = frame["away_key"]
    else:
        frame["away_team"] = frame["away_team"].fillna(frame["away_key"])
    if "game_id" not in frame:
        frame["game_id"] = pd.NA
    if "game_id" in frame:
        has_id = frame["game_id"].notna()
        frame = pd.concat(
            [
                frame.loc[~has_id],
                frame.loc[has_id].drop_duplicates("game_id", keep="last"),
            ]
        )
    frame = frame.sort_values(["date", "game_id"] if "game_id" in frame else ["date"]).reset_index(drop=True)

    histories: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "games": 0, "wins": 0, "points_for": [], "points_against": [],
            "outcomes": [], "home_games": 0, "home_wins": 0,
            "away_games": 0, "away_wins": 0, "home_outcomes": [],
            "away_outcomes": [], "points_for_sum": 0.0,
            "points_against_sum": 0.0, "last_date": None, "elo": 1500.0,
            "name": None,
        }
    )
    head_to_head: dict[tuple[str, str], list[int]] = defaultdict(list)
    feature_rows: list[dict[str, float]] = []
    home_winners: list[int] = []
    for game in frame.itertuples(index=False):
        home, away = game.home_key, game.away_key
        home_state, away_state = histories[home], histories[away]
        matchup = head_to_head[(home, away)]

        def rest_days(state: dict[str, Any]) -> float:
            if state["last_date"] is None:
                return np.nan
            return float(max(0, min((game.date - state["last_date"]).total_seconds() / 86400, 30)))

        def avg(values: list[float], count: int) -> float:
            return float(np.mean(values[-count:])) if values else np.nan

        def win_rate(state: dict[str, Any], count: int | None = None) -> float:
            outcomes = state["outcomes"] if count is None else state["outcomes"][-count:]
            if count is None:
                return float(state["wins"] / state["games"]) if state["games"] else np.nan
            return float(np.mean(outcomes)) if outcomes else np.nan

        home_elo, away_elo = home_state["elo"], away_state["elo"]
        row = {
            "home_is_home": 1.0,
            "home_prior_games": float(home_state["games"]),
            "away_prior_games": float(away_state["games"]),
            "home_win_rate_all": win_rate(home_state),
            "away_win_rate_all": win_rate(away_state),
            "home_win_rate_last5": win_rate(home_state, 5),
            "away_win_rate_last5": win_rate(away_state, 5),
            "home_win_rate_last10": win_rate(home_state, 10),
            "away_win_rate_last10": win_rate(away_state, 10),
            "home_points_for_last5": avg(home_state["points_for"], 5),
            "away_points_for_last5": avg(away_state["points_for"], 5),
            "home_points_against_last5": avg(home_state["points_against"], 5),
            "away_points_against_last5": avg(away_state["points_against"], 5),
            "home_points_for_all": (
                home_state["points_for_sum"] / home_state["games"]
                if home_state["games"] else np.nan
            ),
            "away_points_for_all": (
                away_state["points_for_sum"] / away_state["games"]
                if away_state["games"] else np.nan
            ),
            "home_points_against_all": (
                home_state["points_against_sum"] / home_state["games"]
                if home_state["games"] else np.nan
            ),
            "away_points_against_all": (
                away_state["points_against_sum"] / away_state["games"]
                if away_state["games"] else np.nan
            ),
            "home_rest_days": rest_days(home_state),
            "away_rest_days": rest_days(away_state),
            "home_home_win_rate": (
                home_state["home_wins"] / home_state["home_games"]
                if home_state["home_games"] else np.nan
            ),
            "away_away_win_rate": (
                away_state["away_wins"] / away_state["away_games"]
                if away_state["away_games"] else np.nan
            ),
            "head_to_head_home_win_rate": float(np.mean(matchup)) if matchup else np.nan,
            "head_to_head_games": float(len(matchup)),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": home_elo - away_elo,
        }
        feature_rows.append(row)
        home_won = int(game.home_score > game.away_score)
        home_winners.append(home_won)
        for key, scored, conceded, did_win, is_home in (
            (home, game.home_score, game.away_score, home_won, True),
            (away, game.away_score, game.home_score, 1 - home_won, False),
        ):
            state = histories[key]
            state["games"] += 1
            state["wins"] += did_win
            state["points_for"].append(float(scored))
            state["points_against"].append(float(conceded))
            state["outcomes"].append(did_win)
            state["points_for_sum"] += float(scored)
            state["points_against_sum"] += float(conceded)
            state["last_date"] = game.date
            if is_home:
                state["home_games"] += 1
                state["home_wins"] += did_win
                state["home_outcomes"].append(did_win)
            else:
                state["away_games"] += 1
                state["away_wins"] += did_win
                state["away_outcomes"].append(did_win)
            if key == home and "home_team" in selected:
                state["name"] = str(game.home_team)
            if key == away and "away_team" in selected:
                state["name"] = str(game.away_team)
        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - 65.0) / 400.0))
        update = 20.0 * (home_won - expected_home)
        histories[home]["elo"] = home_elo + update
        histories[away]["elo"] = away_elo - update
        head_to_head[(home, away)].append(home_won)
        head_to_head[(away, home)].append(1 - home_won)

    features = pd.DataFrame(feature_rows, columns=GAME_FEATURES)
    targets = frame[["date", "game_id", "home_team", "away_team", "home_key", "away_key", "home_score", "away_score"]].copy()
    targets["home_win"] = home_winners
    result = pd.concat([targets.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)
    team_state = {
        key: {
            "team_id": key,
            "team": state["name"] or key,
            "games": state["games"],
            "wins": state["wins"],
            "win_percentage": state["wins"] / state["games"] if state["games"] else None,
            "elo": state["elo"],
            "average_points_for": float(np.mean(state["points_for"])) if state["points_for"] else None,
            "average_points_against": float(np.mean(state["points_against"])) if state["points_against"] else None,
            "net_rating_proxy": (
                float(np.mean(state["points_for"]) - np.mean(state["points_against"]))
                if state["points_for"] else None
            ),
            "home_games": state["home_games"],
            "home_win_percentage": (
                state["home_wins"] / state["home_games"] if state["home_games"] else None
            ),
            "away_games": state["away_games"],
            "away_win_percentage": (
                state["away_wins"] / state["away_games"] if state["away_games"] else None
            ),
            "recent_wins": state["outcomes"][-10:],
            "recent_points_for": state["points_for"][-10:],
            "recent_points_against": state["points_against"][-10:],
            "last_date": state["last_date"].isoformat() if state["last_date"] is not None else None,
            "home_recent_wins": state["home_outcomes"][-10:],
            "away_recent_wins": state["away_outcomes"][-10:],
        }
        for key, state in histories.items()
    }
    return result, {
        "available": True,
        "source": path.name,
        "games": len(result),
        "date_start": result["date"].min().isoformat(),
        "date_end": result["date"].max().isoformat(),
        "unusable_game_rows_excluded": int(len(source) - len(result)),
        "feature_columns": GAME_FEATURES,
        "target_columns": ["home_win", "home_score", "away_score"],
        "team_state": team_state,
        "head_to_head": {
            f"{home}|{away}": outcomes
            for (home, away), outcomes in head_to_head.items()
        },
    }


def build_player_frame(directories: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build lagged player box-score features with strictly prior histories."""
    path = _source(directories, "PlayerStatisticsExtended") or _source(directories, "PlayerStatistics")
    if path is None:
        return pd.DataFrame(), {"available": False, "reason": "Player statistics table was not found."}
    header = _column_names(path)
    aliases = {
        "person_id": ("personId", "playerId"),
        "date": ("gameDateTimeEst", "gameDate", "date"),
        "game_id": ("gameId",),
        "team_id": ("playerteamId", "teamId"),
        "team_name": ("playerteamName", "teamName"),
        "opponent_id": ("opponentteamId", "opponentTeamId"),
        "home": ("home",),
        "first_name": ("firstName",),
        "last_name": ("lastName",),
    }
    selected = {
        output: column for output, names in aliases.items()
        if (column := _column(pd.DataFrame(columns=header), names))
    }
    actual_targets = {
        metric: _column(pd.DataFrame(columns=header), names)
        for metric, names in PLAYER_TARGETS.items()
    }
    actual_targets = {key: value for key, value in actual_targets.items() if value}
    required = {"person_id", "date"}
    if not required.issubset(selected) or not actual_targets:
        return pd.DataFrame(), {
            "available": False,
            "reason": "Player table lacks a player ID, date, or supported performance fields.",
            "source": path.name,
        }
    columns = list(dict.fromkeys([*selected.values(), *actual_targets.values()]))
    source = _read(path, columns)
    frame = pd.DataFrame(index=source.index)
    for output, column in selected.items():
        frame[output] = source[column]
    for metric, column in actual_targets.items():
        frame[f"target_{metric}"] = pd.to_numeric(source[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["date", "person_id"])
    frame["person_id"] = frame["person_id"].astype(str)
    frame["player"] = (
        frame["first_name"].fillna("").astype(str).str.strip()
        + " " + frame["last_name"].fillna("").astype(str).str.strip()
        if {"first_name", "last_name"}.issubset(frame.columns)
        else frame["person_id"]
    ).str.strip()
    frame["team_id"] = frame.get("team_id", pd.Series(pd.NA, index=frame.index)).astype("string")
    frame["opponent_id"] = frame.get("opponent_id", pd.Series(pd.NA, index=frame.index)).astype("string")
    frame["home_is_home"] = (
        frame.get("home", pd.Series(0, index=frame.index)).astype("string").str.lower()
        .map({"true": 1.0, "1": 1.0, "1.0": 1.0, "home": 1.0, "false": 0.0, "0": 0.0, "0.0": 0.0, "away": 0.0})
    )
    frame = frame.sort_values(["date", "game_id"] if "game_id" in frame else ["date"]).reset_index(drop=True)
    grouped = frame.groupby("person_id", sort=False)
    frame["prior_games"] = grouped.cumcount().astype(float)
    previous_date = grouped["date"].shift()
    frame["rest_days"] = (
        (frame["date"] - previous_date).dt.total_seconds() / 86400
    ).clip(lower=0, upper=30)
    for metric in actual_targets:
        values = frame[f"target_{metric}"]
        lag = values.groupby(frame["person_id"], sort=False).shift()
        frame[f"prior_{metric}_mean5"] = lag.groupby(frame["person_id"], sort=False).transform(
            lambda series: series.rolling(5, min_periods=1).mean()
        )
        frame[f"prior_{metric}_std5"] = lag.groupby(frame["person_id"], sort=False).transform(
            lambda series: series.rolling(5, min_periods=2).std()
        )
        frame[f"prior_{metric}_mean_all"] = lag.groupby(
            frame["person_id"], sort=False
        ).transform(lambda series: series.expanding(min_periods=1).mean())
    if "opponent_id" in frame and "target_points" in frame:
        matchup_group = frame.groupby(["person_id", "opponent_id"], sort=False)["target_points"]
        lagged_matchup = matchup_group.shift()
        frame["prior_matchup_points_mean"] = lagged_matchup.groupby(
            [frame["person_id"], frame["opponent_id"]], sort=False
        ).transform(lambda series: series.expanding(min_periods=1).mean())
        frame["prior_matchup_games"] = frame.groupby(
            ["person_id", "opponent_id"], sort=False
        ).cumcount().astype(float)
    else:
        frame["prior_matchup_points_mean"] = np.nan
        frame["prior_matchup_games"] = 0.0
    features = [column for column in PLAYER_FEATURES if column in frame]
    targets = [f"target_{metric}" for metric in actual_targets]
    metadata = {
        "available": True,
        "source": path.name,
        "rows": len(frame),
        "players": int(frame["person_id"].nunique()),
        "date_start": frame["date"].min().isoformat(),
        "date_end": frame["date"].max().isoformat(),
        "feature_columns": features,
        "target_columns": targets,
        "supported_metrics": list(actual_targets),
    }
    return frame, metadata
