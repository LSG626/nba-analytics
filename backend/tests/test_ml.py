"""Tests for chronological training features and prediction API contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from ml.features.build_features import build_game_frame, build_player_frame
from ml.training import (
    _predictions_for_split_rows,
    _seasonal_holdout_metrics,
    _temporal_split,
)


def test_game_features_use_only_prior_games(tmp_path: Path) -> None:
    games = tmp_path / "Games.csv"
    games.write_text(
        "gameId,gameDate,hometeamName,hometeamId,awayteamName,awayteamId,homeScore,awayScore\n"
        "1,2024-01-01,A,1,B,2,100,90\n"
        "2,2024-01-03,B,2,A,1,80,110\n"
        "3,2024-01-05,A,1,B,2,90,95\n",
        encoding="utf-8",
    )
    frame, metadata = build_game_frame([tmp_path])

    assert metadata["available"] is True
    assert len(frame) == 3
    assert pd.isna(frame.loc[0, "home_win_rate_all"])
    assert frame.loc[1, "home_prior_games"] == 1
    assert frame.loc[1, "away_prior_games"] == 1
    assert frame.loc[1, "away_win_rate_all"] == 1
    assert frame.loc[1, "home_points_for_all"] == 90
    assert frame.loc[2, "head_to_head_games"] == 2
    # This feature uses prior meetings with the same home/away orientation.
    assert frame.loc[2, "head_to_head_home_win_rate"] == 1


def test_game_feature_builder_excludes_unplayed_and_tied_games(tmp_path: Path) -> None:
    (tmp_path / "Games.csv").write_text(
        "gameId,gameDate,homeTeamName,awayTeamName,homeScore,awayScore\n"
        "1,2024-01-01,A,B,100,90\n"
        "2,2024-01-02,A,B,0,0\n"
        "3,2024-01-03,A,B,95,95\n",
        encoding="utf-8",
    )
    frame, _ = build_game_frame([tmp_path])
    assert frame["game_id"].tolist() == [1]


def test_game_rows_without_ids_are_not_collapsed_as_duplicates(tmp_path: Path) -> None:
    (tmp_path / "Games.csv").write_text(
        "gameId,gameDate,homeTeamName,awayTeamName,homeScore,awayScore\n"
        ",2024-01-01,A,B,100,90\n"
        ",2024-01-02,A,B,95,90\n",
        encoding="utf-8",
    )
    frame, _ = build_game_frame([tmp_path])
    assert len(frame) == 2


def test_player_features_are_shifted_before_each_target(tmp_path: Path) -> None:
    (tmp_path / "PlayerStatisticsExtended.csv").write_text(
        "personId,gameDate,gameId,playerTeamId,opponentteamId,firstName,lastName,home,"
        "numMinutes,points,reboundsTotal,assists,steals,blocks,threePointersMade\n"
        "7,2024-01-01,1,1,2,Test,Player,true,30,10,4,3,1,0,2\n"
        "7,2024-01-03,2,1,2,Test,Player,false,32,20,5,4,2,1,3\n",
        encoding="utf-8",
    )
    frame, metadata = build_player_frame([tmp_path])

    assert metadata["available"] is True
    assert pd.isna(frame.loc[0, "prior_points_mean_all"])
    assert frame.loc[1, "prior_points_mean_all"] == 10
    assert frame.loc[1, "prior_matchup_games"] == 1
    assert frame.loc[1, "prior_matchup_points_mean"] == 10


def test_temporal_splits_have_no_date_overlap() -> None:
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=100, freq="D"),
        "value": range(100),
    })
    train, validation, test, summary = _temporal_split(frame)
    assert train.sum() + validation.sum() + test.sum() == len(frame)
    assert not (train & validation).any()
    assert not (validation & test).any()
    assert summary["test_used_for_selection_or_tuning"] is False
    assert frame.loc[train, "date"].max() < frame.loc[validation, "date"].min()
    assert frame.loc[validation, "date"].max() < frame.loc[test, "date"].min()


def test_target_mask_selects_predictions_in_split_order() -> None:
    split = pd.Series([False, True, True, False, True])
    target = pd.Series([True, False, True, True, True])
    predictions = np.asarray([10.0, 20.0, 30.0])
    assert _predictions_for_split_rows(
        predictions, split, target
    ).tolist() == [20.0, 30.0]


def test_seasonal_holdout_metrics_infer_season_without_source_column() -> None:
    classification = pd.DataFrame({
        "date": ["2020-01-01", "2020-02-01", "2020-11-01", "2020-12-01"],
        "home_win": [0, 1, 0, 1],
        "predicted_home_probability": [0.2, 0.8, 0.3, 0.7],
    })
    scores = pd.DataFrame({
        "date": classification["date"],
        "home_score": [90, 100, 91, 101],
        "away_score": [100, 90, 101, 91],
        "predicted_home_score": [91, 99, 92, 100],
        "predicted_away_score": [99, 91, 100, 92],
        "actual_total_points": [190, 190, 192, 192],
        "predicted_total_points": [190, 190, 192, 192],
        "actual_winning_margin": [-10, 10, -10, 10],
        "predicted_winning_margin": [-8, 8, -8, 8],
    })

    result = _seasonal_holdout_metrics(classification, scores)
    assert set(result["holdout_seasons"]) == {"2019-20", "2020-21"}
    assert result["holdout_seasons"]["2020-21"]["games"] == 2


@pytest.fixture
def prediction_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setattr(main, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(main, "MODEL_EVALUATION_DIR", tmp_path / "evaluation")
    monkeypatch.setattr(main, "PREDICTION_DIR", tmp_path / "predictions")
    return TestClient(main.app)


def test_prediction_endpoints_do_not_fabricate_when_models_missing(
    prediction_client: TestClient,
) -> None:
    assert prediction_client.get(
        "/api/predictions/games?home_team=Celtics&away_team=Knicks"
    ).status_code == 503
    assert prediction_client.get("/api/predictions/players?player=Tatum").status_code == 503
    assert prediction_client.get("/api/models/performance").json()["available"] is False
    assert prediction_client.get("/api/backtesting/results").json()["available"] is False


def test_prediction_discovery_and_historical_holdout_endpoints(
    prediction_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "team_strength.json").write_text(json.dumps({
        "game_team_state": {"1": {"team": "Alpha"}, "2": {"team": "Beta"}},
    }), encoding="utf-8")
    (prediction_dir / "player_state.json").write_text(json.dumps({
        "players": {
            "7": {"player": "Alex Guard", "team_id": "1", "prior_games": 20},
            "8": {"player": "Alex Rookie", "team_id": "1", "prior_games": 2},
        },
    }), encoding="utf-8")
    pd.DataFrame([{
        "date": "2022-01-02T00:00:00+00:00",
        "game_id": 11,
        "home_team": "Alpha",
        "away_team": "Beta",
        "home_score": 101,
        "away_score": 98,
        "home_win": 1,
        "predicted_home_probability": 0.62,
        "predicted_away_probability": 0.38,
        "predicted_home_win": 1,
        "confidence": 0.62,
    }]).to_csv(prediction_dir / "historical_game_winner_predictions.csv", index=False)
    pd.DataFrame([{
        "date": "2022-01-02T00:00:00+00:00",
        "game_id": 11,
        "home_team": "Alpha",
        "away_team": "Beta",
        "home_score": 101,
        "away_score": 98,
        "predicted_home_score": 99,
        "predicted_away_score": 96,
        "actual_total_points": 199,
        "predicted_total_points": 195,
        "actual_winning_margin": 3,
        "predicted_winning_margin": 3,
    }]).to_csv(prediction_dir / "historical_game_score_predictions.csv", index=False)
    monkeypatch.setattr(main, "PREDICTION_DIR", prediction_dir)

    teams = prediction_client.get("/api/predictions/teams/list").json()
    assert teams["available"] is True
    assert teams["teams"][0]["team"] == "Alpha"
    players = prediction_client.get(
        "/api/predictions/players/search?q=alex"
    ).json()["players"]
    assert [item["player_id"] for item in players] == ["7"]
    history = prediction_client.get(
        "/api/predictions/historical/games",
        params={"team": "1", "season": "2021-22"},
    ).json()
    assert history["available"] is True
    assert len(history["games"]) == 1
    assert history["games"][0]["home_score"] == 101
