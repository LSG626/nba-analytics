"""Chronological model training, validation, evaluation, and local artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from ml.features.build_features import (
    GAME_FEATURES,
    PLAYER_FEATURES,
    build_game_frame,
    build_player_frame,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRS = [
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "Raw",
]
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
EVALUATION_DIR = PROJECT_ROOT / "ml" / "evaluation"
PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"
SEED = 42


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or value is None:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _temporal_split(
    frame: pd.DataFrame,
    date_column: str = "date",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    dates = pd.Series(frame[date_column].dropna().sort_values().unique())
    if len(dates) < 10:
        raise ValueError("At least ten unique dates are required for chronological splits.")
    train_end = max(1, int(len(dates) * train_fraction))
    validation_end = max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(dates) - 1)
    train_dates = set(dates.iloc[:train_end])
    validation_dates = set(dates.iloc[train_end:validation_end])
    test_dates = set(dates.iloc[validation_end:])
    train = frame[date_column].isin(train_dates)
    validation = frame[date_column].isin(validation_dates)
    test = frame[date_column].isin(test_dates)
    summary = {
        "method": "chronological unique-date split; no date overlaps",
        "train": {
            "rows": int(train.sum()),
            "date_start": str(min(train_dates)),
            "date_end": str(max(train_dates)),
        },
        "validation": {
            "rows": int(validation.sum()),
            "date_start": str(min(validation_dates)),
            "date_end": str(max(validation_dates)),
        },
        "test": {
            "rows": int(test.sum()),
            "date_start": str(min(test_dates)),
            "date_end": str(max(test_dates)),
        },
        "validation_used_for_selection": True,
        "test_used_for_selection_or_tuning": False,
    }
    return train, validation, test, summary


def _predictions_for_split_rows(
    predictions: np.ndarray,
    split_mask: pd.Series,
    target_mask: pd.Series,
) -> np.ndarray:
    """Select predictions using a full-frame target mask aligned to one split."""
    return predictions[target_mask.loc[split_mask].to_numpy()]


def _classifier_candidates() -> dict[str, Callable[[], Pipeline]]:
    return {
        "LogisticRegression": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]),
        "RandomForest": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestClassifier(
                n_estimators=160, max_depth=16, min_samples_leaf=8,
                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
            )),
        ]),
        "XGBoost": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=160, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                eval_metric="logloss", tree_method="hist", n_jobs=4,
                random_state=SEED,
            )),
        ]),
        "LightGBM": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                n_estimators=160, learning_rate=0.05, num_leaves=24,
                max_depth=-1, min_child_samples=40, subsample=0.85,
                colsample_bytree=0.85, reg_lambda=1.0, verbosity=-1,
                n_jobs=4, random_state=SEED,
            )),
        ]),
    }


def _regressor_candidates() -> dict[str, Callable[[], Pipeline]]:
    return {
        "RandomForest": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestRegressor(
                n_estimators=160, max_depth=16, min_samples_leaf=8,
                n_jobs=-1, random_state=SEED,
            )),
        ]),
        "XGBoost": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=160, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                objective="reg:squarederror", tree_method="hist", n_jobs=4,
                random_state=SEED,
            )),
        ]),
        "LightGBM": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                n_estimators=160, learning_rate=0.05, num_leaves=24,
                min_child_samples=40, subsample=0.85, colsample_bytree=0.85,
                reg_lambda=1.0, verbosity=-1, n_jobs=4, random_state=SEED,
            )),
        ]),
    }


def _class_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    predicted = (probability >= 0.5).astype(int)
    fpr, tpr, thresholds = roc_curve(actual, probability)
    observed, predicted_fraction = calibration_curve(
        actual, probability, n_bins=10, strategy="quantile"
    )
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "roc_auc": roc_auc_score(actual, probability) if len(np.unique(actual)) == 2 else None,
        "log_loss": log_loss(actual, np.column_stack([1 - probability, probability]), labels=[0, 1]),
        "brier_score": brier_score_loss(actual, probability),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "roc_curve": {
            "false_positive_rate": fpr.tolist(),
            "true_positive_rate": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "calibration": {
            "mean_predicted_probability": predicted_fraction.tolist(),
            "observed_home_win_rate": observed.tolist(),
        },
    }


def _reg_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else 0.0,
    }


def _seasonal_holdout_metrics(
    classification_predictions: pd.DataFrame,
    score_predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Compare holdout metrics by inferred NBA season when no season field exists."""
    classifier = classification_predictions.copy()
    scores = score_predictions.copy()
    for frame in (classifier, scores):
        dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        start_year = dates.dt.year - (dates.dt.month < 10).astype(int)
        frame["season"] = start_year.astype("Int64").astype("string") + "-" + (
            (start_year + 1) % 100
        ).astype("Int64").astype("string").str.zfill(2)
    results: dict[str, Any] = {}
    for season, group in classifier.groupby("season", dropna=True, sort=True):
        score_group = scores.loc[scores["season"] == season]
        if group.empty or score_group.empty:
            continue
        classification = _class_metrics(
            group["home_win"].astype(int).to_numpy(),
            group["predicted_home_probability"].astype(float).to_numpy(),
        )
        results[str(season)] = {
            "games": int(len(group)),
            "classification": {
                key: classification[key]
                for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier_score")
            },
            "scores": {
                "home_score": _reg_metrics(
                    score_group["home_score"].to_numpy(),
                    score_group["predicted_home_score"].to_numpy(),
                ),
                "away_score": _reg_metrics(
                    score_group["away_score"].to_numpy(),
                    score_group["predicted_away_score"].to_numpy(),
                ),
                "total_points": _reg_metrics(
                    score_group["actual_total_points"].to_numpy(),
                    score_group["predicted_total_points"].to_numpy(),
                ),
                "winning_margin": _reg_metrics(
                    score_group["actual_winning_margin"].to_numpy(),
                    score_group["predicted_winning_margin"].to_numpy(),
                ),
            },
        }
    return {
        "season_assignment": (
            "Inferred from game timestamp using an October season boundary; "
            "the source Games.csv does not provide a season column."
        ),
        "holdout_seasons": results,
    }


def _importance(model: Pipeline, feature_names: list[str]) -> list[dict[str, Any]]:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.mean(np.abs(estimator.coef_), axis=0)
    else:
        return []
    names = list(feature_names)
    if len(values) > len(names):
        names.extend([f"missing_indicator_{i}" for i in range(len(values) - len(names))])
    return sorted(
        [{"feature": names[index], "importance": float(value)} for index, value in enumerate(values)],
        key=lambda item: item["importance"],
        reverse=True,
    )


def _save_classification_plots(
    metrics: dict[str, Any], test_predictions: pd.DataFrame, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cm = metrics["confusion_matrix"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Confusion matrix", "ROC curve", "Calibration"),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}, {"type": "scatter"}]],
    )
    fig.add_trace(go.Heatmap(z=cm, x=["Away", "Home"], y=["Away", "Home"], colorscale="Blues"), 1, 1)
    roc = metrics["roc_curve"]
    fig.add_trace(go.Scatter(x=roc["false_positive_rate"], y=roc["true_positive_rate"], mode="lines", name="ROC"), 1, 2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance"), 1, 2)
    calibration = metrics["calibration"]
    fig.add_trace(go.Scatter(
        x=calibration["mean_predicted_probability"],
        y=calibration["observed_home_win_rate"],
        mode="lines+markers", name="Observed",
    ), 1, 3)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect"), 1, 3)
    fig.update_layout(title="Game winner — holdout evaluation", height=480)
    fig.write_html(output / "game_winner_diagnostics.html", include_plotlyjs="cdn")
    scatter = go.Figure(go.Scatter(
        x=test_predictions["actual_home_score"],
        y=test_predictions["predicted_home_score"],
        mode="markers", marker={"opacity": 0.35},
    ))
    scatter.add_trace(go.Scatter(
        x=[0, max(test_predictions["actual_home_score"].max(), 1)],
        y=[0, max(test_predictions["actual_home_score"].max(), 1)],
        mode="lines", name="Perfect",
    ))
    scatter.update_layout(title="Home score — actual vs predicted", xaxis_title="Actual", yaxis_title="Predicted")
    scatter.write_html(output / "home_score_actual_vs_predicted.html", include_plotlyjs="cdn")


def _team_strength_source(directories: list[Path]) -> dict[str, Any]:
    from ml.features.build_features import _column, _source, _read, _column_names

    path = _source(directories, "TeamStatisticsExtended") or _source(directories, "TeamStatistics")
    if path is None:
        return {"available": False, "reason": "Team statistics dataset was not found."}
    header = _column_names(path)
    aliases = {
        "team_id": ("teamId", "playerTeamId"),
        "team_name": ("teamName", "playerTeamName"),
        "date": ("gameDateTimeEst", "gameDate", "date"),
        "offensive_rating": ("offensiveRating", "estimatedOffensiveRating"),
        "defensive_rating": ("defensiveRating", "estimatedDefensiveRating"),
        "net_rating": ("netRating", "estimatedNetRating"),
        "pace": ("pace", "estimatedPace"),
    }
    actual = {
        output: column for output, names in aliases.items()
        if (column := _column(pd.DataFrame(columns=header), names))
    }
    metrics = [key for key in ("offensive_rating", "defensive_rating", "net_rating", "pace") if key in actual]
    if "team_id" not in actual or not metrics:
        return {"available": False, "source": path.name, "reason": "No team identifier plus supported ratings/pace fields."}
    source = _read(path, list(actual.values()))
    columns = {value: key for key, value in actual.items()}
    frame = source.rename(columns=columns)
    frame["team_id"] = frame["team_id"].astype("string")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True) if "date" in frame else pd.NaT
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    grouped = frame.groupby("team_id", dropna=True)
    result: dict[str, Any] = {}
    for team_id, group in grouped:
        item: dict[str, Any] = {
            metric: float(group[metric].mean())
            for metric in metrics if group[metric].notna().any()
        }
        name_col = "team_name" if "team_name" in frame else None
        if name_col and group[name_col].notna().any():
            item["team"] = str(group[name_col].dropna().iloc[-1])
        if group["date"].notna().any():
            item["latest_stat_date"] = group["date"].max().isoformat()
        item["games_in_statistics"] = len(group)
        result[str(team_id)] = item
    return {
        "available": True,
        "source": path.name,
        "supported_metrics": metrics,
        "teams": result,
    }


def _write_plotly_feature_importance(
    importance: list[dict[str, Any]], output_dir: Path, title: str
) -> None:
    if not importance:
        return
    frame = pd.DataFrame(importance[:20]).sort_values("importance")
    figure = go.Figure(go.Bar(x=frame["importance"], y=frame["feature"], orientation="h"))
    figure.update_layout(title=title, xaxis_title="Importance", yaxis_title="Feature")
    figure.write_html(output_dir / "game_winner_feature_importance.html", include_plotlyjs="cdn")


def _rolling_origin_backtest(
    frame: pd.DataFrame,
    feature_columns: list[str],
    selected_classifier: str,
) -> dict[str, Any]:
    """Evaluate expanding-window folds strictly before the final test period."""
    dates = pd.Series(frame["date"].dropna().sort_values().unique())
    if len(dates) < 30:
        return {"available": False, "reason": "Not enough distinct dates for rolling-origin folds."}
    folds = []
    for fold_number, origin_fraction in enumerate((0.50, 0.62, 0.74), start=1):
        origin = max(10, int(len(dates) * origin_fraction))
        end = min(len(dates), origin + max(1, int(len(dates) * 0.08)))
        train_dates = set(dates.iloc[:origin])
        evaluation_dates = set(dates.iloc[origin:end])
        train = frame["date"].isin(train_dates)
        evaluation = frame["date"].isin(evaluation_dates)
        if train.sum() < 100 or evaluation.sum() < 20:
            continue
        x_train = frame.loc[train, feature_columns]
        x_eval = frame.loc[evaluation, feature_columns]
        y_train = frame.loc[train, "home_win"].astype(int)
        y_eval = frame.loc[evaluation, "home_win"].astype(int).to_numpy()
        model = _classifier_candidates()[selected_classifier]()
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_eval)[:, 1]
        baseline_probability = float(y_train.mean())
        classifier_model = _class_metrics(y_eval, probability)
        baseline_metrics = _class_metrics(
            y_eval, np.full(len(y_eval), baseline_probability)
        )
        folds.append({
            "fold": fold_number,
            "train_end": max(train_dates),
            "evaluation_start": min(evaluation_dates),
            "evaluation_end": max(evaluation_dates),
            "train_rows": int(train.sum()),
            "evaluation_rows": int(evaluation.sum()),
            "classification": {
                "model": {
                    key: classifier_model[key]
                    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier_score")
                },
                "baseline_historical_home_rate": {
                    key: baseline_metrics[key]
                    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier_score")
                },
            },
        })
    return {
        "available": bool(folds),
        "method": "expanding training window; three rolling chronological evaluation windows within the development period; final holdout excluded",
        "classifier": selected_classifier,
        "folds": folds,
    }


def train_models(
    data_directories: list[Path] | None = None,
    *,
    model_dir: Path = MODEL_DIR,
    evaluation_dir: Path = EVALUATION_DIR,
    prediction_dir: Path = PREDICTION_DIR,
) -> dict[str, Any]:
    """Train model families, select by validation only, then evaluate on test."""
    directories = data_directories or DEFAULT_DATA_DIRS
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    game_frame, game_metadata = build_game_frame(directories)
    if game_frame.empty:
        raise FileNotFoundError(game_metadata.get("reason", "No usable games were found."))
    game_train, game_validation, game_test, game_split = _temporal_split(game_frame)
    features = [column for column in GAME_FEATURES if column in game_frame]
    x_train = game_frame.loc[game_train, features]
    x_validation = game_frame.loc[game_validation, features]
    x_development = game_frame.loc[game_train | game_validation, features]
    x_test = game_frame.loc[game_test, features]
    y = game_frame["home_win"].astype(int)
    candidate_results: dict[str, Any] = {}
    winner_candidates: dict[str, Pipeline] = {}
    validation_scores: dict[str, float] = {}
    for name, factory in _classifier_candidates().items():
        estimator = factory()
        estimator.fit(x_train, y.loc[game_train])
        probability = estimator.predict_proba(x_validation)[:, 1]
        metrics = _class_metrics(y.loc[game_validation].to_numpy(), probability)
        candidate_results[name] = {"validation": metrics}
        winner_candidates[name] = estimator
        validation_scores[name] = metrics["log_loss"]
        LOGGER.info("Winner validation %s log loss %.5f", name, metrics["log_loss"])
    selected_classifier = min(validation_scores, key=validation_scores.get)
    final_classifier = _classifier_candidates()[selected_classifier]()
    final_classifier.fit(x_development, y.loc[game_train | game_validation])
    test_probability = final_classifier.predict_proba(x_test)[:, 1]
    winner_test_metrics = _class_metrics(y.loc[game_test].to_numpy(), test_probability)
    candidate_results[selected_classifier]["test"] = winner_test_metrics
    winner_importance = _importance(final_classifier, features)
    classification_predictions = game_frame.loc[game_test, [
        "date", "game_id", "home_team", "away_team", "home_score", "away_score", "home_win"
    ]].copy()
    classification_predictions["predicted_home_probability"] = test_probability
    classification_predictions["predicted_away_probability"] = 1 - test_probability
    classification_predictions["predicted_home_win"] = (test_probability >= 0.5).astype(int)
    classification_predictions["confidence"] = np.maximum(test_probability, 1 - test_probability)

    regression_candidates: dict[str, Any] = {}
    selected_regressors: dict[str, str] = {}
    regression_test: dict[str, Any] = {}
    regression_predictions: dict[str, np.ndarray] = {}
    regressor_factories = _regressor_candidates()
    for target in ("home_score", "away_score"):
        target_results = {}
        target_models = {}
        target_validation: dict[str, float] = {}
        target_values = pd.to_numeric(game_frame[target], errors="coerce")
        for name, factory in regressor_factories.items():
            estimator = factory()
            estimator.fit(x_train, target_values.loc[game_train])
            validation_prediction = estimator.predict(x_validation)
            metrics = _reg_metrics(target_values.loc[game_validation].to_numpy(), validation_prediction)
            target_results[name] = {"validation": metrics}
            target_models[name] = estimator
            target_validation[name] = metrics["mae"]
        chosen = min(target_validation, key=target_validation.get)
        final_regressor = regressor_factories[chosen]()
        final_regressor.fit(x_development, target_values.loc[game_train | game_validation])
        predicted = np.maximum(final_regressor.predict(x_test), 0)
        regression_predictions[target] = predicted
        regression_test[target] = _reg_metrics(target_values.loc[game_test].to_numpy(), predicted)
        target_results[chosen]["test"] = regression_test[target]
        regression_candidates[target] = target_results
        selected_regressors[target] = chosen
        regression_predictions[target] = predicted
    regression_predictions_frame = game_frame.loc[game_test, [
        "date", "game_id", "home_team", "away_team", "home_score", "away_score"
    ]].copy()
    regression_predictions_frame["predicted_home_score"] = regression_predictions["home_score"]
    regression_predictions_frame["predicted_away_score"] = regression_predictions["away_score"]
    regression_predictions_frame["actual_total_points"] = (
        regression_predictions_frame["home_score"] + regression_predictions_frame["away_score"]
    )
    regression_predictions_frame["predicted_total_points"] = (
        regression_predictions_frame["predicted_home_score"]
        + regression_predictions_frame["predicted_away_score"]
    )
    regression_predictions_frame["actual_winning_margin"] = (
        regression_predictions_frame["home_score"] - regression_predictions_frame["away_score"]
    )
    regression_predictions_frame["predicted_winning_margin"] = (
        regression_predictions_frame["predicted_home_score"]
        - regression_predictions_frame["predicted_away_score"]
    )
    test_margin_metrics = _reg_metrics(
        regression_predictions_frame["actual_winning_margin"].to_numpy(),
        regression_predictions_frame["predicted_winning_margin"].to_numpy(),
    )
    test_total_metrics = _reg_metrics(
        regression_predictions_frame["actual_total_points"].to_numpy(),
        regression_predictions_frame["predicted_total_points"].to_numpy(),
    )
    classification_predictions.to_csv(
        prediction_dir / "historical_game_winner_predictions.csv", index=False
    )
    regression_predictions_frame.to_csv(
        prediction_dir / "historical_game_score_predictions.csv", index=False
    )
    _write_plotly_feature_importance(
        winner_importance, evaluation_dir, f"Game winner feature importance — {selected_classifier}"
    )
    backtest = _rolling_origin_backtest(
        game_frame.loc[game_train | game_validation].reset_index(drop=True),
        features,
        selected_classifier,
    )
    _write_json(evaluation_dir / "backtesting_results.json", backtest)

    player_frame, player_metadata = build_player_frame(directories)
    player_report: dict[str, Any] = {
        "available": player_metadata.get("available", False),
        "source": player_metadata.get("source"),
        "unavailable_reason": player_metadata.get("reason"),
    }
    player_state: dict[str, Any] = {}
    if not player_frame.empty and "minutes" in player_metadata["supported_metrics"]:
        player_history = player_frame
        player_features = [name for name in PLAYER_FEATURES if name in player_frame]
        player_frame = player_frame.loc[player_frame["prior_games"] >= 3].copy()
        player_train, player_validation, player_test, player_split = _temporal_split(player_frame)
        valid_player_features = [
            feature for feature in player_features
            if player_frame.loc[player_train, feature].notna().any()
        ]
        x_player_train = player_frame.loc[player_train, valid_player_features]
        x_player_validation = player_frame.loc[player_validation, valid_player_features]
        x_player_development = player_frame.loc[player_train | player_validation, valid_player_features]
        x_player_test = player_frame.loc[player_test, valid_player_features]
        minutes_target = player_frame["target_minutes"]
        minutes_train_mask = player_train & minutes_target.notna()
        minutes_validation_mask = player_validation & minutes_target.notna()
        minutes_test_mask = player_test & minutes_target.notna()
        minutes_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingRegressor(
                max_iter=100, max_leaf_nodes=15, l2_regularization=1.0,
                early_stopping=False, random_state=SEED,
            )),
        ])
        minutes_model.fit(
            player_frame.loc[minutes_train_mask, valid_player_features],
            minutes_target.loc[minutes_train_mask],
        )
        validation_expected_minutes = np.maximum(
            minutes_model.predict(x_player_validation), 0
        )
        val_minutes_pred = _predictions_for_split_rows(
            validation_expected_minutes, player_validation, minutes_validation_mask
        )
        test_minutes_pred = np.maximum(minutes_model.predict(x_player_test), 0)
        minutes_residual = (
            minutes_target.loc[minutes_validation_mask].to_numpy() - val_minutes_pred
        )
        minutes_model.fit(
            player_frame.loc[minutes_target.notna(), valid_player_features],
            minutes_target.loc[minutes_target.notna()],
        )
        player_bundles: dict[str, Any] = {
            "minutes": {
                "model": minutes_model,
                "target_type": "minutes",
                "residual_quantiles": np.quantile(minutes_residual, [0.05, 0.95]).tolist(),
            }
        }
        player_metrics: dict[str, Any] = {
            "minutes": {
                "validation": _reg_metrics(minutes_target.loc[minutes_validation_mask].to_numpy(), val_minutes_pred),
                "test": _reg_metrics(
                    minutes_target.loc[minutes_test_mask].to_numpy(),
                    _predictions_for_split_rows(
                        test_minutes_pred, player_test, minutes_test_mask
                    ),
                ),
            }
        }
        player_predictions = player_frame.loc[player_test, [
            "date", "game_id", "person_id", "player", "team_id", "opponent_id", "home_is_home"
        ]].copy()
        player_predictions["actual_minutes"] = minutes_target.loc[player_test].to_numpy()
        player_predictions["predicted_minutes"] = test_minutes_pred
        player_predictions["predicted_minutes_lower"] = np.maximum(
            0, test_minutes_pred + player_bundles["minutes"]["residual_quantiles"][0]
        )
        player_predictions["predicted_minutes_upper"] = np.maximum(
            player_predictions["predicted_minutes_lower"],
            test_minutes_pred + player_bundles["minutes"]["residual_quantiles"][1],
        )
        test_expected_minutes = test_minutes_pred
        for metric in player_metadata["supported_metrics"]:
            if metric == "minutes":
                continue
            target_values = player_frame[f"target_{metric}"]
            usable_train = player_train & minutes_target.gt(0) & target_values.notna()
            usable_validation = player_validation & minutes_target.gt(0) & target_values.notna()
            usable_development = (player_train | player_validation) & minutes_target.gt(0) & target_values.notna()
            usable_test = player_test & target_values.notna()
            if int(usable_train.sum()) < 50 or not usable_test.any():
                continue
            rate_train = target_values.loc[usable_train] / minutes_target.loc[usable_train]
            rate_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", HistGradientBoostingRegressor(
                    max_iter=100, max_leaf_nodes=15, l2_regularization=1.0,
                    early_stopping=False, random_state=SEED,
                )),
            ])
            rate_model.fit(
                player_frame.loc[usable_train, valid_player_features],
                rate_train,
            )
            rate_val = np.maximum(
                rate_model.predict(player_frame.loc[usable_validation, valid_player_features]), 0
            )
            val_minutes_for_target = _predictions_for_split_rows(
                validation_expected_minutes, player_validation, usable_validation
            )
            val_predictions = rate_val * val_minutes_for_target
            val_actual = target_values.loc[usable_validation].to_numpy()
            residual = val_actual - val_predictions
            quantiles = np.quantile(residual, [0.05, 0.95]).tolist()

            rate_development = target_values.loc[usable_development] / minutes_target.loc[usable_development]
            rate_model.fit(
                player_frame.loc[usable_development, valid_player_features],
                rate_development,
            )
            rate_test = np.maximum(
                rate_model.predict(player_frame.loc[usable_test, valid_player_features]), 0
            )
            test_minutes_for_target = _predictions_for_split_rows(
                test_expected_minutes, player_test, usable_test
            )
            test_predictions = rate_test * test_minutes_for_target
            test_actual = target_values.loc[usable_test].to_numpy()
            metrics = {
                "validation": _reg_metrics(val_actual, val_predictions),
                "test": _reg_metrics(test_actual, test_predictions),
                "validation_residual_interval_90": quantiles,
                "test_interval_coverage_90": float(np.mean(
                    (test_actual >= np.maximum(0, test_predictions + quantiles[0]))
                    & (test_actual <= np.maximum(0, test_predictions + quantiles[1]))
                )),
                "test_rows": int(len(test_actual)),
            }
            player_metrics[metric] = metrics
            rate_model.fit(
                player_frame.loc[target_values.notna() & minutes_target.gt(0), valid_player_features],
                (
                    target_values.loc[target_values.notna() & minutes_target.gt(0)]
                    / minutes_target.loc[target_values.notna() & minutes_target.gt(0)]
                ),
            )
            player_bundles[metric] = {
                "model": rate_model,
                "target_type": "rate_per_minute",
                "residual_quantiles": quantiles,
            }
            prediction_col = f"predicted_{metric}"
            actual_col = f"actual_{metric}"
            usable_ids = player_frame.loc[usable_test, ["date", "person_id", "game_id"]].index
            player_predictions.loc[usable_ids, actual_col] = test_actual
            player_predictions.loc[usable_ids, prediction_col] = test_predictions
            player_predictions.loc[usable_ids, f"{prediction_col}_lower"] = np.maximum(
                0, test_predictions + quantiles[0]
            )
            player_predictions.loc[usable_ids, f"{prediction_col}_upper"] = np.maximum(
                player_predictions.loc[usable_ids, f"{prediction_col}_lower"],
                test_predictions + quantiles[1],
            )
        joblib.dump({
            "models": player_bundles,
            "features": valid_player_features,
            "supported_metrics": list(player_bundles),
            "knowledge_cutoff": player_frame.loc[player_train | player_validation, "date"].max().isoformat(),
            "availability_note": "Statistical expectation only; player availability, injuries, and lineup status are unknown.",
            "minimum_prior_games": 3,
        }, model_dir / "player_performance.joblib")
        player_predictions.to_csv(
            prediction_dir / "historical_player_predictions.csv", index=False
        )
        player_report = {
            "available": True,
            "source": player_metadata["source"],
            "rows_after_minimum_history": len(player_frame),
            "players": player_metadata["players"],
            "metrics": player_metrics,
            "split": player_split,
            "model": "HistGradientBoostingRegressor; per-minute rates scaled by expected minutes",
            "intervals": "Empirical 5th/95th validation residual quantiles; coverage is measured on the chronological test set, not guaranteed.",
            "availability_note": "Predictions do not assert confirmed player availability or lineup status.",
        }
        # Snapshot the latest observed history; no roster or injury assumptions are added.
        player_state = {}
        for player_id, group in player_history.groupby("person_id", sort=False):
            last = group.iloc[-1]
            record: dict[str, Any] = {
                "person_id": str(player_id),
                "player": str(last["player"]),
                "team_id": str(last["team_id"]) if pd.notna(last["team_id"]) else None,
                "opponent_id": str(last["opponent_id"]) if pd.notna(last["opponent_id"]) else None,
                "last_game_date": last["date"].isoformat(),
                "prior_games": int(len(group)),
                "home_is_home": 1.0,
                "rest_days": 1.0,
            }
            for metric in player_metadata["supported_metrics"]:
                values = pd.to_numeric(group[f"target_{metric}"], errors="coerce").dropna()
                record[f"prior_{metric}_mean5"] = float(values.tail(5).mean()) if not values.empty else None
                record[f"prior_{metric}_mean_all"] = float(values.mean()) if not values.empty else None
                record[f"prior_{metric}_std5"] = float(values.tail(5).std()) if len(values.tail(5)) > 1 else None
            matchup_history: dict[str, dict[str, Any]] = {}
            if "target_points" in group:
                for opponent_id, matchup in group.dropna(subset=["opponent_id"]).groupby(
                    "opponent_id", sort=False
                ):
                    points = pd.to_numeric(matchup["target_points"], errors="coerce").dropna()
                    if not points.empty:
                        matchup_history[str(opponent_id)] = {
                            "prior_matchup_points_mean": float(points.mean()),
                            "prior_matchup_games": int(len(points)),
                        }
            record["matchups"] = matchup_history
            record["prior_matchup_points_mean"] = None
            record["prior_matchup_games"] = 0
            player_state[str(player_id)] = record
    elif not player_frame.empty:
        player_report = {
            "available": False,
            "source": player_metadata.get("source"),
            "unavailable_reason": (
                "Minutes are absent from the player dataset; the configured two-stage "
                "player predictor cannot estimate per-minute statistics."
            ),
            "supported_metrics": player_metadata.get("supported_metrics", []),
        }

    # The untouched holdout metrics above remain the reported estimate. After
    # evaluation, refit the selected deployment estimators on all history.
    final_classifier = _classifier_candidates()[selected_classifier]()
    final_classifier.fit(game_frame.loc[:, features], y)
    game_cutoff = game_frame["date"].max()
    joblib.dump({
        "model": final_classifier,
        "features": features,
        "model_name": selected_classifier,
        "knowledge_cutoff": game_cutoff.isoformat(),
    }, model_dir / "game_winner.joblib")
    for target, selected_name in selected_regressors.items():
        target_values = pd.to_numeric(game_frame[target], errors="coerce")
        production_model = _regressor_candidates()[selected_name]()
        production_model.fit(game_frame.loc[:, features], target_values)
        joblib.dump({
            "model": production_model,
            "features": features,
            "model_name": selected_name,
            "target": target,
            "knowledge_cutoff": game_cutoff.isoformat(),
        }, model_dir / f"game_{target}.joblib")

    team_strength = _team_strength_source(directories)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_sources": {
            "game": game_metadata["source"],
            "player": player_metadata.get("source"),
            "team_strength": team_strength.get("source"),
            "data_directories": [str(path) for path in directories],
        },
        "game_data": {
            key: value for key, value in game_metadata.items()
            if key not in {"team_state", "head_to_head"}
        },
        "game_split": game_split,
        "winner": {
            "selected_by_validation_log_loss": selected_classifier,
            "validation_candidates": {
                name: item["validation"] for name, item in candidate_results.items()
            },
            "test_metrics": winner_test_metrics,
            "feature_importance": winner_importance[:30],
        },
        "scores": {
            "selection_metric": "validation MAE, separately for home and away points",
            "selected_models": selected_regressors,
            "validation_candidates": regression_candidates,
            "test_metrics": {
                "home_score": regression_test["home_score"],
                "away_score": regression_test["away_score"],
                "total_points": test_total_metrics,
                "winning_margin": test_margin_metrics,
            },
            "consistency": "Total points and winning margin are derived from the same predicted home and away scores.",
        },
        "seasonal_holdout_metrics": _seasonal_holdout_metrics(
            classification_predictions, regression_predictions_frame
        ),
        "players": player_report,
        "backtesting": backtest,
        "team_strength": {
            "ratings": team_strength,
            "elo_source": "Chronological game scores, updated only after each completed game.",
            "net_rating_proxy": "Average points scored minus average points conceded when official ratings are absent.",
        },
        "limitations": [
            "Processed Parquet is preferred; when absent the training workflow uses the supplied Raw CSVs without modifying them.",
            "Only completed games with positive home and away scores enter game targets.",
            "No player injury, availability, lineup, or confirmed rotation source is available.",
            "Player prediction intervals are empirical validation residual intervals and do not guarantee future coverage.",
            "Team history includes all competitions represented in the source files; game type is not silently restricted.",
            "SHAP is not required; feature importance uses model-native impurity importance or absolute logistic coefficients.",
            "Selected deployment estimators are refit on all historical rows only after the chronological holdout has been evaluated; holdout metrics are preserved and were not used for model selection.",
        ],
    }
    _write_json(evaluation_dir / "model_performance.json", metadata)
    _write_json(model_dir / "model_metadata.json", {
        "trained_at": metadata["generated_at"],
        "knowledge_cutoff": game_cutoff,
        "selected_game_winner_model": selected_classifier,
        "selected_score_models": selected_regressors,
        "sources": metadata["data_sources"],
        "game_features": features,
    })
    _write_json(prediction_dir / "team_strength.json", {
        "as_of": game_frame["date"].max(),
        "game_team_state": game_metadata["team_state"],
        "head_to_head": game_metadata["head_to_head"],
        "team_statistics": team_strength,
    })
    if player_state:
        _write_json(prediction_dir / "player_state.json", {
            "as_of": player_frame["date"].max(),
            "players": player_state,
        })
    from ml.predict import generate_upcoming_schedule_predictions

    upcoming_predictions = generate_upcoming_schedule_predictions(
        directories, model_dir, prediction_dir, knowledge_cutoff=game_cutoff
    )
    metadata["upcoming_schedule_predictions"] = upcoming_predictions
    _write_plotly_feature_importance(
        winner_importance, evaluation_dir, f"Game winner feature importance — {selected_classifier}"
    )
    _save_classification_plots(winner_test_metrics, regression_predictions_frame.rename(
        columns={"home_score": "actual_home_score"}
    ), evaluation_dir)
    report_path = evaluation_dir / "model_performance.json"
    metadata["artifacts"] = {
        "performance_report": str(report_path),
        "historical_game_predictions": str(prediction_dir / "historical_game_winner_predictions.csv"),
        "historical_score_predictions": str(prediction_dir / "historical_game_score_predictions.csv"),
        "historical_player_predictions": (
            str(prediction_dir / "historical_player_predictions.csv") if player_state else None
        ),
    }
    _write_json(report_path, metadata)
    return metadata
