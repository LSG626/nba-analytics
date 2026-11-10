"""Tests for schema-flexible ingestion and dataset API endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.ingestion import ingest_directory


def test_ingestion_cleans_dates_deduplicates_and_reports_missing_ids(
    tmp_path: Path,
) -> None:
    """Preserve valid rows and report data issues without inventing identities."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    source = raw_dir / "Games.csv"
    source.write_text(
        "gameId,gameDate,homeScore,awayScore,playerId,teamId\n"
        "1001,2025-01-03,101,99,,1610612738\n"
        "1001,2025-01-03,101,99,,1610612738\n"
        "1002,01/04/2025,88,92,55,\n",
        encoding="utf-8",
    )

    report = ingest_directory(
        raw_dir,
        processed_dir,
        processed_dir / "data_quality_report.json",
    )

    assert report["files_processed"] == 1
    dataset = report["datasets"][0]
    assert dataset["duplicate_rows_removed"] == 1
    assert dataset["processed_rows"] == 2
    assert dataset["nba_field_checks"]["player_id_missing_values"]["playerId"] == 1
    assert dataset["nba_field_checks"]["team_id_missing_values"]["teamId"] == 1
    assert dataset["invalid_dates"] == {}
    loaded = pd.read_parquet(processed_dir / "Games.parquet")
    assert len(loaded) == 2
    assert str(loaded["gameDate"].dtype).startswith("datetime64")


def test_ingestion_supports_unrecognized_schema_and_generates_report(
    tmp_path: Path,
) -> None:
    """Do not reject CSV tables that lack familiar NBA field names."""
    raw_dir = tmp_path / "incoming"
    raw_dir.mkdir()
    (raw_dir / "custom.csv").write_text(
        "code,metric,note\n001,3.5,  \n002,4.0,available\n",
        encoding="utf-8",
    )

    report_path = tmp_path / "reports" / "quality.json"
    report = ingest_directory(raw_dir, tmp_path / "out", report_path)

    assert report["files_failed"] == 0
    assert report["datasets"][0]["dataset_kind"] == "unknown"
    assert report["datasets"][0]["missing_values"]["note"] == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["files_processed"] == 1


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    monkeypatch.setattr(main, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(main, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(main, "EDA_FALLBACK_DIR", tmp_path / "legacy-raw")
    monkeypatch.setattr(
        main,
        "QUALITY_REPORT_PATH",
        processed_dir / "data_quality_report.json",
    )
    return TestClient(main.app)


def test_health_and_empty_dataset_status(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    response = client.get("/api/datasets/status")
    assert response.status_code == 200
    assert response.json()["raw_csv_count"] == 0


def test_upload_and_list_csv(client: TestClient) -> None:
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("sample.csv", b"gameId,homeScore\n1,100\n", "text/csv")},
    )
    assert response.status_code == 201
    assert response.json()["uploaded"] is True
    assert client.get("/api/datasets").json()["datasets"][0]["name"] == "sample.csv"


def test_upload_rejects_non_csv(client: TestClient) -> None:
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("notes.txt", b"not a dataset", "text/plain")},
    )
    assert response.status_code == 400


def test_quality_and_metadata_endpoints(client: TestClient) -> None:
    client.post(
        "/api/datasets/upload",
        files={"file": ("games.csv", b"gameId,homeScore\n1,100\n", "text/csv")},
    )
    assert client.get("/api/datasets/quality").json() == {
        "available": False,
        "report": None,
    }
    metadata = client.get("/api/datasets/games.csv/metadata")
    assert metadata.status_code == 200
    assert metadata.json()["columns"] == ["gameId", "homeScore"]
    assert client.get("/api/datasets/../secrets.csv/metadata").status_code in {
        400,
        404,
    }


def test_eda_endpoints_use_available_fields_and_filters(
    client: TestClient,
) -> None:
    raw_dir = main.RAW_DATA_DIR
    raw_dir.mkdir(parents=True)
    (raw_dir / "Games.csv").write_text(
        "gameId,gameDate,hometeamName,hometeamId,awayteamName,awayteamId,"
        "homeScore,awayScore,winner\n"
        "1,2024-10-22,Celtics,1,Knicks,2,110,100,1\n"
        "2,2025-01-02,Knicks,2,Celtics,1,90,95,1\n",
        encoding="utf-8",
    )
    (raw_dir / "TeamStatistics.csv").write_text(
        "gameId,gameDate,teamName,teamId,opponentTeamName,home,win,"
        "teamScore,opponentScore,reboundsTotal,turnovers,ot1Points\n"
        "1,2024-10-22,Celtics,1,Knicks,1,1,110,100,45,10,0\n"
        "1,2024-10-22,Knicks,2,Celtics,0,0,100,110,40,12,0\n"
        "2,2025-01-02,Knicks,2,Celtics,1,0,90,95,42,11,5\n"
        "2,2025-01-02,Celtics,1,Knicks,0,1,95,90,44,9,5\n",
        encoding="utf-8",
    )
    (raw_dir / "PlayerStatistics.csv").write_text(
        "firstName,lastName,personId,gameId,gameDate,playerteamName,"
        "playerteamId,win,points,reboundsTotal,assists,steals,blocks,"
        "numMinutes,fieldGoalsPercentage,threePointersPercentage,"
        "freeThrowsPercentage,turnovers\n"
        "Jayson,Tatum,10,1,2024-10-22,Celtics,1,1,30,8,4,1,0,35,0.5,0.4,0.8,2\n"
        "Jayson,Tatum,10,2,2025-01-02,Celtics,1,1,25,7,5,2,1,34,0.45,0.3,0.9,3\n"
        "Jalen,Brunson,20,1,2024-10-22,Knicks,2,0,20,3,7,1,0,36,0.4,0.3,0.8,4\n",
        encoding="utf-8",
    )
    (raw_dir / "Players.csv").write_text(
        "personId,firstName,lastName\n10,Jayson,Tatum\n20,Jalen,Brunson\n",
        encoding="utf-8",
    )

    overview = client.get("/api/eda/overview").json()
    assert overview["counts"]["games"] == 2
    assert overview["counts"]["teams"] == 2
    assert overview["counts"]["players"] == 2
    games = client.get("/api/eda/games?team=Celtics").json()
    assert games["game_count"] == 2
    assert games["home_away_win_distribution"]["home_wins"] == 1
    assert games["overtime"]["games_with_overtime"] == 1
    teams = client.get("/api/eda/teams?team=Celtics").json()
    assert teams["team_count"] == 1
    assert teams["teams"][0]["win_percentage"] == 100
    players = client.get("/api/eda/players?player=Tatum").json()
    assert players["player_count"] == 1
    assert players["players"][0]["metrics"]["points"] == 27.5
    assert client.get("/api/eda/correlations").status_code == 200
    assert client.get("/api/eda/seasons?player=Tatum").status_code == 200


def test_eda_reports_missing_datasets_without_inventing_results(
    client: TestClient,
) -> None:
    assert client.get("/api/eda/games").json() == {
        "available": False,
        "reason": "No Games dataset was found.",
    }
    assert client.get("/api/eda/correlations").json()["available"] is False


def test_game_analysis_excludes_unplayed_zero_score_records(
    client: TestClient,
) -> None:
    raw_dir = main.RAW_DATA_DIR
    raw_dir.mkdir(parents=True)
    (raw_dir / "Games.csv").write_text(
        "gameId,gameDate,homeScore,awayScore\n"
        "1,2025-01-01,100,90\n"
        "2,2025-01-02,0,0\n",
        encoding="utf-8",
    )
    result = client.get("/api/eda/games").json()
    assert result["game_count"] == 2
    assert result["scored_game_count"] == 1
    assert result["invalid_or_missing_score_rows"] == 1
    assert result["average_points_per_team"] == 95
    assert result["lowest_scoring_games"][0]["total_points"] == 190
