# COURTSIDE AI model evaluation

## Run and source data

The evaluation below was generated from the supplied local data on 2026-11-10.
The source selection was:

- Games: `Games.csv`
- Player box scores: `PlayerStatisticsExtended.csv`
- Team ratings: `TeamStatisticsExtended.csv`

The game pipeline retained **73,482 completed, non-tied games**, covering
**1946-11-26 through 2026-11-07**; four unusable source rows were excluded.
The chronological unique-timestamp split was 57,596 training games
(1946-11-26–2015-01-19), 7,780 validation games
(2015-01-19–2021-01-21), and 8,106 test games
(2021-01-22–2026-11-07). The test window was not used for model selection.

The player source provided **830,948 rows after the three-game prior-history
threshold**, across **3,552 players**. Player test coverage was 130,537
player-game rows from 2022-03-14 through 2026-11-07. The team table supports
offensive rating, defensive rating, net rating, and pace summaries; Elo is
derived chronologically from game scores.

## Game winner

XGBoost was selected by lowest validation log loss. LightGBM had slightly
higher validation accuracy, but its validation log loss was 0.62375 versus
0.62335 for XGBoost.

| Candidate | Validation accuracy | Validation ROC-AUC | Validation log loss | Validation Brier |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.624 | 0.698 | 0.64331 | 0.2260 |
| Random Forest | 0.641 | 0.689 | 0.63387 | 0.2218 |
| XGBoost (selected) | 0.649 | 0.693 | **0.62335** | **0.2170** |
| LightGBM | **0.652** | 0.693 | 0.62375 | 0.2172 |

Selected XGBoost holdout results:

| Accuracy | Precision | Recall | F1 | ROC-AUC | Log loss | Brier |
|---:|---:|---:|---:|---:|---:|---:|
| 0.642 | 0.651 | 0.763 | 0.703 | 0.686 | 0.63352 | 0.2214 |

The holdout confusion matrix, with rows and columns ordered away/home, is
`[[1771, 1836], [1067, 3432]]`. The model identifies more home wins than away
wins, and its probability quality is more informative than a hard-label
accuracy alone. These results are historical and do not guarantee future
season performance.

## Score prediction

LightGBM was selected by validation MAE for both home and away points. Holdout
metrics:

| Target | MAE (points) | RMSE (points) | R² |
|---|---:|---:|---:|
| Home score | 9.585 | 12.075 | 0.120 |
| Away score | 9.578 | 12.102 | 0.126 |
| Total points | 15.279 | 19.434 | 0.115 |
| Winning margin | 11.258 | 14.383 | 0.136 |

Total and margin are derived from the exact same non-negative home and away
score predictions, so those outputs remain arithmetically consistent. The
modest R² values show that the available features explain only a limited share
of game-to-game scoring variance; exact game scoring remains difficult at the
observed point-error scale.

## Player performance

The deployed two-stage HistGradientBoosting approach predicts minutes first,
then predicts supported per-minute box-score rates and scales them by expected
minutes. Test metrics:

| Statistic | MAE | RMSE | R² | Empirical 90% interval coverage |
|---|---:|---:|---:|---:|
| Minutes | 5.198 | 6.716 | 0.616 | Not measured |
| Points | 4.619 | 6.081 | 0.520 | 88.9% |
| Rebounds | 1.900 | 2.508 | 0.453 | 89.5% |
| Assists | 1.319 | 1.812 | 0.505 | 88.9% |
| Steals | 0.693 | 0.902 | 0.117 | 90.2% |
| Blocks | 0.507 | 0.724 | 0.212 | 89.8% |
| Three-pointers made | 0.873 | 1.228 | 0.314 | 87.6% |

Coverage is measured on the chronological test window using residual
quantiles estimated from validation; it is empirical, not a future guarantee.
Minute intervals use the same residual-quantile approach, but minute interval
coverage is not included in the current report.

## Rolling-origin backtest

Three expanding-window evaluation periods were run within the development
history; the final holdout was excluded. The baseline predicts the historical
training home-win rate for every evaluation game.

| Fold | Evaluation dates | Model accuracy | Baseline accuracy | Model log loss | Baseline log loss |
|---|---|---:|---:|---:|---:|
| 1 | 2004-04-10–2007-01-13 | 0.647 | 0.605 | 0.6246 | 0.6737 |
| 2 | 2008-03-17–2010-12-07 | 0.682 | 0.609 | 0.5916 | 0.6706 |
| 3 | 2012-04-14–2014-12-27 | 0.668 | 0.587 | 0.6150 | 0.6824 |

The selected model outperformed this simple baseline on accuracy and log loss
in all three measured windows. The folds are separated historical periods,
not independent seasons, and only game-winner classification is backtested.

## Holdout performance by season

`Games.csv` has no season column, so the holdout results are grouped by an
inferred NBA season using an October boundary (October–September). The final
2026-27 group is partial, ending on 2026-11-07.

| Inferred season | Games | Winner accuracy | Winner log loss | Home-score MAE | Away-score MAE |
|---|---:|---:|---:|---:|---:|
| 2020-21 | 959 | 0.626 | 0.6424 | 9.45 | 9.61 |
| 2021-22 | 1,390 | 0.650 | 0.6410 | 9.35 | 9.51 |
| 2022-23 | 1,384 | 0.624 | 0.6493 | 9.33 | 9.07 |
| 2023-24 | 1,383 | 0.652 | 0.6221 | 9.44 | 9.57 |
| 2024-25 | 1,384 | 0.640 | 0.6265 | 9.59 | 9.65 |
| 2025-26 | 1,400 | 0.659 | 0.6189 | 10.20 | 9.98 |
| 2026-27 (partial) | 206 | 0.612 | 0.6597 | 10.19 | 10.11 |

These are descriptive subsets of the same chronological holdout, not separate
model-tuning runs. The 206-game partial season should not be compared as if it
were a complete season.

## Artifacts and visualizations

The detailed machine-readable metrics and split metadata are in
`ml/evaluation/model_performance.json`; the rolling-origin results are in
`ml/evaluation/backtesting_results.json`. Diagnostic Plotly HTML is generated
in `ml/evaluation/`. Models are stored in `ml/models/`, and holdout forecasts
are stored under `data/predictions/`.

No future schedule row matched a known historical team and a future date during
this run, so **zero upcoming forecasts were generated**. The latest-predictions
endpoint reports that state rather than inventing fixtures or results.

## Interpretation and limitations

- Validation selected the models; reported holdout metrics were not used for
  tuning. The deployed models were refit on all historical rows only after
  holdout evaluation.
- There is no supplied injury, availability, confirmed lineup, or rotation
  feed. Player forecasts must not be interpreted as confirmation that a
  player will play.
- The game table contains a long historical span; data definitions,
  competition coverage, and rule eras may vary. Training does not silently
  filter game types.
- The score model's low R² and approximately 9.6-point per-team MAE limit
  precision for individual games.
- Team rating summaries are historical means of the supplied fields, not
  current official ratings. Elo is an internal score-derived measure.
- Player interval coverage is empirical on this data and can shift in future
  seasons. Minute interval coverage was not calculated.
- SHAP explanations and regression rolling-origin backtests are not currently
  produced.
