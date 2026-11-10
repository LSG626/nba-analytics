# COURTSIDE AI prediction methodology

## Data sources and availability

Training discovers `Games` and player/team statistics in this order:
`data/processed/`, `data/raw/`, then the supplied `Raw/` directory. Parquet is
preferred to CSV within each directory. It does not download data or create
synthetic records. A missing or unusable game dataset stops training; missing
player or team-rating data is reported as unavailable.

The supplied historical data is not a real-time feed. Injury, availability,
starting-lineup, rotation, and confirmed roster data are not inferred. Player
forecasts describe statistical expectations only.

## Game feature construction

Game rows require a parseable timestamp, both team identifiers (or names), and
positive, non-tied scores. Tied, unplayed, or incomplete rows are excluded.
Games are ordered chronologically; each feature row is created before that
game's result is added to team history.

Features include pregame all-time and recent team win rates, prior scoring and
points-conceded averages, rest days, home/away records, orientation-specific
head-to-head history, and Elo ratings. Elo is updated after each completed
game. It is an internally derived rating, not an official NBA rating. All-time
averages use historical observations available at that point, not future data.

## Player features

Player box-score rows are ordered by player and game timestamp. Rolling
averages, cumulative averages, consistency, rest, and matchup features are
shifted so the current game's box score cannot enter its own features. Player
records without a minimum history are excluded from fitting and scored only
when targets are present.

The two-stage predictor estimates minutes first, then estimates each supported
per-minute statistic conditional on expected minutes. Supported target fields
are detected in the source; absent metrics are not modeled. Validation-period
residual quantiles are reported as empirical 90% intervals. They are not
guaranteed prediction coverage.

## Split, selection, and evaluation

Unique timestamps are divided chronologically into 70% training, 15%
validation, and 15% test windows (subject to unique timestamp boundaries).
There is no random row split. Validation chooses the game winner classifier by
log loss and each score model by MAE. Final metrics use the later test window,
which is not used for selection or tuning.

After holdout evaluation, selected deployment models are refit on all available
history. Their `knowledge_cutoff` identifies the latest observed timestamp.
Predictions for requested dates at or before that cutoff are rejected.

Winner candidates are Logistic Regression, Random Forest, XGBoost, and LightGBM.
Score models compare Random Forest, XGBoost, and LightGBM regressors for home
and away points independently. Predicted total points and margin are derived
from the same non-negative home and away score estimates, ensuring arithmetic
consistency. Classification reports accuracy, precision, recall, F1, ROC-AUC,
log loss, Brier score, confusion matrix, ROC and calibration data. Regression
reports MAE, RMSE, and R-squared.

Rolling-origin backtests use expanding chronological training windows and
multiple later development-period evaluation windows; the final holdout is
excluded. The rolling-origin implementation currently backtests game-winner
classification only. It also compares against a historical home-win-rate
baseline.

The supplied game table has no season field. For holdout season comparisons,
the report infers an NBA season using an October boundary (October–September)
and labels this as an estimate rather than a source-provided field.

## Artifacts and commands

From the project root:

```powershell
python -m scripts.train_models
```

Repeat `--data-dir` to provide data directories, or use `--model-dir`,
`--evaluation-dir`, and `--prediction-dir` to control local artifact locations.
Training writes Joblib models under `ml/models/`, evaluation JSON and Plotly
diagnostics under `ml/evaluation/`, team/player state and historical/future
predictions under `data/predictions/`. Upcoming forecasts are generated only
when a supplied schedule file contains future games and identifiable teams.

## Limitations

- Mixed competitions or inconsistent data-source coverage may affect team
  history; the pipeline does not silently assume one league/season.
- Statistics unavailable in the detected schema are omitted.
- A schedule prediction is not evidence that a player will participate.
- Model confidence is a predicted class probability, not a guarantee.
- The test window is historical and cannot guarantee performance on future
  seasons, rule changes, roster changes, or unobserved competitions.
- Feature importance is model-native or coefficient-based; SHAP explanations
  are not produced.
