"""Inference helpers for saved local prediction models."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.features.build_features import GAME_FEATURES, _column, _read, _source

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_team(query: str, teams: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    normalized = query.strip().casefold()
    exact = [
        (team_id, state) for team_id, state in teams.items()
        if team_id.casefold() == normalized
        or str(state.get("team", "")).casefold() == normalized
    ]
    matches = exact or [
        (team_id, state) for team_id, state in teams.items()
        if normalized in str(state.get("team", "")).casefold()
    ]
    if not matches:
        raise KeyError(f"Team '{query}' is not in the trained historical team data.")
    if len(matches) > 1:
        raise ValueError(f"Team '{query}' is ambiguous; use a full name or team ID.")
    return matches[0]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def matchup_features(
    home_state: dict[str, Any],
    away_state: dict[str, Any],
    h2h: list[int],
    game_date: datetime,
) -> pd.DataFrame:
    def rest(state: dict[str, Any]) -> float:
        last = state.get("last_date")
        if not last:
            return np.nan
        last_date = pd.to_datetime(last, utc=True)
        current_date = pd.to_datetime(game_date, utc=True)
        return float(np.clip((current_date - last_date).total_seconds() / 86400, 0, 30))

    home_recent = home_state.get("recent_wins", [])
    away_recent = away_state.get("recent_wins", [])
    home_recent_for = home_state.get("recent_points_for", [])
    away_recent_for = away_state.get("recent_points_for", [])
    home_recent_against = home_state.get("recent_points_against", [])
    away_recent_against = away_state.get("recent_points_against", [])
    home_elo = float(home_state.get("elo", 1500))
    away_elo = float(away_state.get("elo", 1500))
    values = {
        "home_is_home": 1.0,
        "home_prior_games": float(home_state.get("games", 0)),
        "away_prior_games": float(away_state.get("games", 0)),
        "home_win_rate_all": home_state.get("win_percentage"),
        "away_win_rate_all": away_state.get("win_percentage"),
        "home_win_rate_last5": _mean(home_recent[-5:]),
        "away_win_rate_last5": _mean(away_recent[-5:]),
        "home_win_rate_last10": _mean(home_recent),
        "away_win_rate_last10": _mean(away_recent),
        "home_points_for_last5": _mean(home_recent_for[-5:]),
        "away_points_for_last5": _mean(away_recent_for[-5:]),
        "home_points_against_last5": _mean(home_recent_against[-5:]),
        "away_points_against_last5": _mean(away_recent_against[-5:]),
        "home_points_for_all": home_state.get("average_points_for"),
        "away_points_for_all": away_state.get("average_points_for"),
        "home_points_against_all": home_state.get("average_points_against"),
        "away_points_against_all": away_state.get("average_points_against"),
        "home_rest_days": rest(home_state),
        "away_rest_days": rest(away_state),
        "home_home_win_rate": home_state.get("home_win_percentage"),
        "away_away_win_rate": away_state.get("away_win_percentage"),
        "head_to_head_home_win_rate": _mean(h2h),
        "head_to_head_games": float(len(h2h)),
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_difference": home_elo - away_elo,
    }
    return pd.DataFrame([{feature: values.get(feature, np.nan) for feature in GAME_FEATURES}])


def predict_matchup(
    home_team: str,
    away_team: str,
    game_date: datetime | None = None,
    *,
    model_dir: Path = MODEL_DIR,
    prediction_dir: Path = PREDICTION_DIR,
) -> dict[str, Any]:
    winner_path = model_dir / "game_winner.joblib"
    score_home_path = model_dir / "game_home_score.joblib"
    score_away_path = model_dir / "game_away_score.joblib"
    state_path = prediction_dir / "team_strength.json"
    required = (winner_path, score_home_path, score_away_path, state_path)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Game models are not trained; run python -m scripts.train_models first.")
    state_payload = _read_json(state_path)
    teams = state_payload.get("game_team_state", {})
    home_id, home_state = _resolve_team(home_team, teams)
    away_id, away_state = _resolve_team(away_team, teams)
    if home_id == away_id:
        raise ValueError("Home and away teams must be different.")
    cutoff = pd.to_datetime(state_payload["as_of"], utc=True)
    requested_date = game_date or datetime.now(timezone.utc)
    requested = pd.to_datetime(requested_date, utc=True)
    if requested <= cutoff:
        raise ValueError("Prediction date precedes the latest training data; historical predictions are evaluation-only.")
    h2h = state_payload.get("head_to_head", {}).get(f"{home_id}|{away_id}", [])
    feature_row = matchup_features(home_state, away_state, h2h, requested_date)
    winner_bundle = joblib.load(winner_path)
    home_score_bundle = joblib.load(score_home_path)
    away_score_bundle = joblib.load(score_away_path)
    winner_features = winner_bundle["features"]
    score_home_features = home_score_bundle["features"]
    score_away_features = away_score_bundle["features"]
    home_probability = float(
        winner_bundle["model"].predict_proba(feature_row[winner_features])[:, 1][0]
    )
    predicted_home_score = float(max(0, home_score_bundle["model"].predict(
        feature_row[score_home_features]
    )[0]))
    predicted_away_score = float(max(0, away_score_bundle["model"].predict(
        feature_row[score_away_features]
    )[0]))
    importance = state_payload.get("feature_importance", [])
    return {
        "available": True,
        "prediction_type": "model_prediction",
        "game_date": requested.isoformat(),
        "knowledge_cutoff": cutoff.isoformat(),
        "home_team": home_state.get("team", home_id),
        "away_team": away_state.get("team", away_id),
        "home_win_probability": home_probability,
        "away_win_probability": 1 - home_probability,
        "predicted_winner": home_state.get("team", home_id) if home_probability >= 0.5 else away_state.get("team", away_id),
        "model_confidence": max(home_probability, 1 - home_probability),
        "winner_model": winner_bundle["model_name"],
        "predicted_home_points": predicted_home_score,
        "predicted_away_points": predicted_away_score,
        "predicted_total_points": predicted_home_score + predicted_away_score,
        "predicted_winning_margin": predicted_home_score - predicted_away_score,
        "score_models": {
            "home": home_score_bundle["model_name"],
            "away": away_score_bundle["model_name"],
        },
        "important_features": importance[:10],
        "lineup_or_availability_status": "unknown; no injury or confirmed lineup data was supplied",
    }


def predict_player(
    player: str,
    *,
    opponent: str | None = None,
    home: bool = True,
    game_date: datetime | None = None,
    model_dir: Path = MODEL_DIR,
    prediction_dir: Path = PREDICTION_DIR,
) -> dict[str, Any]:
    model_path = model_dir / "player_performance.joblib"
    state_path = prediction_dir / "player_state.json"
    if not model_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("Player models are unavailable; train models with the supplied player statistics.")
    state_payload = _read_json(state_path)
    profiles = state_payload.get("players", {})
    query = player.strip().casefold()
    matches = [
        item for key, item in profiles.items()
        if key.casefold() == query or str(item.get("player", "")).casefold() == query
    ]
    if not matches:
        matches = [
            item for item in profiles.values()
            if query in str(item.get("player", "")).casefold()
        ]
    if not matches:
        raise KeyError(f"Player '{player}' is not in the trained historical player data.")
    if len(matches) > 1:
        raise ValueError(f"Player '{player}' is ambiguous; use a full name or player ID.")
    profile = matches[0]
    bundle = joblib.load(model_path)
    if int(profile.get("prior_games", 0)) < int(bundle["minimum_prior_games"]):
        raise ValueError(
            f"Player '{player}' has fewer than {bundle['minimum_prior_games']} prior games."
        )
    cutoff = pd.to_datetime(state_payload["as_of"], utc=True)
    requested = pd.to_datetime(game_date or datetime.now(timezone.utc), utc=True)
    if requested <= cutoff:
        raise ValueError("Prediction date precedes the latest player-statistics data.")
    row = {
        **profile,
        "home_is_home": float(home),
        "rest_days": float(np.clip(
            (requested - pd.to_datetime(profile["last_game_date"], utc=True)).total_seconds() / 86400,
            0, 30,
        )),
        "prior_matchup_games": 0.0,
        "prior_matchup_points_mean": np.nan,
    }
    opponent_id = None
    if opponent:
        teams_payload = _read_json(prediction_dir / "team_strength.json")
        try:
            opponent_id, _ = _resolve_team(
                opponent, teams_payload.get("game_team_state", {})
            )
        except KeyError:
            opponent_id = opponent.strip()
            if opponent_id not in profile.get("matchups", {}):
                raise
        matchup = profile.get("matchups", {}).get(opponent_id, {})
        row["prior_matchup_points_mean"] = matchup.get(
            "prior_matchup_points_mean", np.nan
        )
        row["prior_matchup_games"] = matchup.get("prior_matchup_games", 0)
    features = bundle["features"]
    feature_frame = pd.DataFrame([{feature: row.get(feature, np.nan) for feature in features}])
    model_items = bundle["models"]
    minutes = float(max(0, model_items["minutes"]["model"].predict(feature_frame)[0]))
    predictions: dict[str, Any] = {
        "minutes": minutes,
    }
    intervals: dict[str, Any] = {}
    for metric, item in model_items.items():
        if metric == "minutes":
            residuals = item["residual_quantiles"]
            intervals[metric] = [
                max(0, minutes + residuals[0]),
                max(0, minutes + residuals[1]),
            ]
            continue
        rate = max(0, float(item["model"].predict(feature_frame)[0]))
        value = max(0, rate * minutes)
        residuals = item["residual_quantiles"]
        predictions[metric] = value
        intervals[metric] = [
            max(0, value + residuals[0]),
            max(0, value + residuals[1]),
        ]
    return {
        "available": True,
        "prediction_type": "model_prediction",
        "player": profile["player"],
        "player_id": profile["person_id"],
        "team_id": profile.get("team_id"),
        "opponent": opponent,
        "historical_matchup_games": row["prior_matchup_games"],
        "home": home,
        "game_date": requested.isoformat(),
        "knowledge_cutoff": cutoff.isoformat(),
        "predicted_statistics": predictions,
        "empirical_90_percent_intervals": intervals,
        "interval_method": "chronological validation residual quantiles; coverage is empirical, not guaranteed",
        "availability_status": "unknown; this statistical forecast does not confirm player availability or lineup status",
        "minimum_prior_games": bundle["minimum_prior_games"],
    }


def generate_upcoming_schedule_predictions(
    directories: list[Path],
    model_dir: Path,
    prediction_dir: Path,
    *,
    knowledge_cutoff: pd.Timestamp,
) -> dict[str, Any]:
    """Predict only games present in supplied schedules and later than now."""
    schedule_paths: list[Path] = []
    for directory in directories:
        if directory.is_dir():
            schedule_paths.extend(
                path for path in directory.glob("*.csv")
                if "schedule" in path.stem.casefold()
            )
    if not schedule_paths:
        return {"available": False, "reason": "No schedule files were supplied.", "predictions": 0}
    now = pd.Timestamp.now(tz="UTC")
    schedule_rows: list[pd.DataFrame] = []
    for path in schedule_paths:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        home_id = _column(pd.DataFrame(columns=header), ("homeTeamId", "hometeamId"))
        away_id = _column(pd.DataFrame(columns=header), ("awayTeamId", "awayteamId"))
        date = _column(pd.DataFrame(columns=header), ("gameDateTimeEst", "gameDate", "date"))
        game_id = _column(pd.DataFrame(columns=header), ("gameId",))
        home_name = _column(pd.DataFrame(columns=header), ("homeTeamName", "hometeamName"))
        away_name = _column(pd.DataFrame(columns=header), ("awayTeamName", "awayteamName"))
        if not home_id or not away_id or not date:
            continue
        selected = [field for field in (home_id, away_id, date, game_id, home_name, away_name) if field]
        frame = pd.read_csv(path, usecols=selected, low_memory=False)
        frame = frame.rename(columns={
            home_id: "home_team_id", away_id: "away_team_id", date: "date",
            **({game_id: "game_id"} if game_id else {}),
            **({home_name: "home_team"} if home_name else {}),
            **({away_name: "away_team"} if away_name else {}),
        })
        frame["schedule_source"] = path.name
        schedule_rows.append(frame)
    if not schedule_rows:
        return {"available": False, "reason": "Schedule files did not contain usable team IDs and dates.", "predictions": 0}
    schedules = pd.concat(schedule_rows, ignore_index=True)
    schedules["date"] = pd.to_datetime(schedules["date"], errors="coerce", utc=True)
    schedules = schedules.dropna(subset=["date", "home_team_id", "away_team_id"])
    schedules = schedules.loc[
        (schedules["date"] > now)
        & (schedules["date"] > pd.to_datetime(knowledge_cutoff, utc=True))
    ]
    results = []
    state_payload = _read_json(prediction_dir / "team_strength.json")
    teams = state_payload.get("game_team_state", {})
    winner = joblib.load(model_dir / "game_winner.joblib")
    home_model = joblib.load(model_dir / "game_home_score.joblib")
    away_model = joblib.load(model_dir / "game_away_score.joblib")
    for game in schedules.itertuples(index=False):
        try:
            home_id, home_state = _resolve_team(str(game.home_team_id), teams)
            away_id, away_state = _resolve_team(str(game.away_team_id), teams)
        except (KeyError, ValueError):
            continue
        matchup = state_payload.get("head_to_head", {}).get(f"{home_id}|{away_id}", [])
        row = matchup_features(home_state, away_state, matchup, game.date.to_pydatetime())
        probability = float(winner["model"].predict_proba(row[winner["features"]])[:, 1][0])
        predicted_home = float(max(0, home_model["model"].predict(row[home_model["features"]])[0]))
        predicted_away = float(max(0, away_model["model"].predict(row[away_model["features"]])[0]))
        results.append({
            "game_id": getattr(game, "game_id", None),
            "game_date": game.date.isoformat(),
            "home_team": getattr(game, "home_team", home_state.get("team", home_id)),
            "away_team": getattr(game, "away_team", away_state.get("team", away_id)),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_win_probability": probability,
            "away_win_probability": 1 - probability,
            "predicted_winner": home_state.get("team", home_id) if probability >= 0.5 else away_state.get("team", away_id),
            "predicted_home_points": predicted_home,
            "predicted_away_points": predicted_away,
            "predicted_total_points": predicted_home + predicted_away,
            "predicted_winning_margin": predicted_home - predicted_away,
            "schedule_source": game.schedule_source,
            "prediction_type": "model_prediction_for_scheduled_game",
        })
    output = prediction_dir / "upcoming_game_predictions.csv"
    prediction_columns = [
        "game_id", "game_date", "home_team", "away_team", "home_team_id",
        "away_team_id", "home_win_probability", "away_win_probability",
        "predicted_winner", "predicted_home_points", "predicted_away_points",
        "predicted_total_points", "predicted_winning_margin", "schedule_source",
        "prediction_type",
    ]
    pd.DataFrame(results, columns=prediction_columns).to_csv(output, index=False)
    return {
        "available": bool(results),
        "predictions": len(results),
        "source_schedules": sorted({row["schedule_source"] for row in results}),
        "file": str(output),
        "reason": None if results else "No future scheduled games matched known historical teams.",
    }
