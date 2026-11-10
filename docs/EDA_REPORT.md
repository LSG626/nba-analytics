# COURTSIDE AI — Exploratory Data Analysis

## Scope and reproducibility

This report was generated from the local files available on 2026-11-10. The
configured `data/raw/` and `data/processed/` folders contain no source tables
in this workspace; the supplied CSVs are in the legacy `Raw/` directory. The
EDA source resolver checks processed Parquet, configured raw CSV, and then
`Raw/`, so the analysis below uses the uploaded files without copying or
modifying them.

The analysis selects `Games.csv`, `TeamStatisticsExtended.csv`,
`PlayerStatisticsExtended.csv`, and `Players.csv` for its core tables. It
discovers all source files and schemas; tables not selected for analysis are
identified in the API response, but their row counts, missingness and
duplicates are labeled unscanned rather than inferred. Numeric columns are
loaded dynamically for analysis. No expected column is required: unsupported
metrics are returned in `unavailable_analyses`.

## Dataset overview

| Measure | Observed value |
|---|---:|
| Game records in `Games.csv` | 73,486 |
| Date coverage in game/player-stat tables | 1946-11-26 to 2026-11-07 |
| Derived NBA seasons in game/player-stat tables | 81 (1946-47 through 2026-27) |
| Distinct team IDs in game table | 42 |
| Player profiles in `Players.csv` | 6,820 |
| Distinct player IDs in player-statistics table | 4,171 |
| Team-statistics rows | 80,138 |
| Player-statistics rows | 843,946 |

Loaded analysis fields were 99.998% complete in Games, 94.483% in team
statistics, 99.712% in player statistics and 99.985% in the roster. No exact
duplicate rows were found in any of those four loaded tables. The quality
figures apply to the selected analysis columns (not every column in the
provider's extended files). Additional full schemas are returned under
`datasets`; per-column missing-value counts and numeric summaries are in
`tables`.

The player roster is broader than the stat table's distinct player population.
The team-statistics source also contains international club names, so team
statistics should not be interpreted as NBA-only without filtering the
available competition/game-type fields. Historical team names and franchises
can also vary across seasons.

Game and team summaries include every matching row in the selected source by
default; game type is not silently restricted to regular-season NBA play.
Schedules are not merged into completed-game results.

## Game analysis

Score comparisons use only records with positive home and away scores.
Zero-score or missing-score records are retained in the record count but
excluded from scoring averages, margins, home advantage, histograms and
highest/lowest scored-game rankings. This avoids treating schedule placeholders
as completed games. Two of 73,486 game records have invalid/missing scores; the
API reports the number of such rows explicitly.

Scoring averages, absolute winning margins, home/away outcomes, scoring
distribution, seasonal scoring, highest/lowest games and overtime frequency
are calculated from the fields available. `TeamStatisticsExtended.csv` has
overtime point fields; the analysis identifies 2,169 overtime games among
40,069 games with team-stat rows (5.413%). This denominator is the team-stats
source coverage, not the full game table.

Among 73,484 valid scored games, average points per team are 104.29, mean
combined score is 208.59, and mean absolute winning margin is 11.03. The
observed home win rate is 61.55% (45,229 home wins, 28,253 away wins and two
ties).

The highest-scoring record observed was Nuggets–Pistons on 1983-12-13
(184–186, 370 combined points). The lowest valid completed game record is
Lakers–Pistons on 1950-11-22 (18–19, 37 combined points). Zero-score records
previously appeared at the bottom of a naive low-score ranking; they are now
excluded from that ranking.

## Team analysis

For each available team identifier (42 identifiers), the service calculates
games, wins, winning percentage, average scored/conceded points, scoring
consistency, home/away splits, season results and longest chronological
winning/losing streaks where dates and outcomes are present. Team results use
the extended team game-stat table, which contains 80,138 team-game records and
competition coverage beyond NBA-only team names. A small number of records have
team IDs but no team name; they are displayed by ID and can rank highly on very
small samples (the first two such IDs have only two and three rows). Among the
named, long-running franchise groups, the San Antonio Spurs record a 61.26% win
rate over 2,829 team-game rows; the Los Angeles Lakers record 56.38% over 2,829
rows, Miami Heat 55.94% over 2,835 rows, and Boston Celtics 55.77% over 2,826
rows. Team identities are grouped by ID and displayed using the latest
available name, avoiding separate rows for a franchise's historical
city/name variants.

These are descriptive rates across the file's included games, not
schedule-strength-adjusted rankings. Team name changes and non-NBA clubs are
source limitations.

## Player analysis

The extended player-stat file supports per-game averages and within-player
standard deviations for points, rebounds, assists, steals, blocks, minutes,
shooting percentages, turnovers and wins. Shooting percentages are returned in
percentage points when the source encodes them as fractions. Top lists include
the number of source game rows per player and have no minimum-games cutoff.
Therefore small-sample leaders must not be interpreted as season awards or
qualified rankings. For example, the raw points-per-game list starts with a
33-point average based on one game row; Luka Doncic averages 28.85 over 610
rows. Apply an explicit minimum-games threshold in a downstream dashboard if
qualified leaderboards are desired.

Game-level stat records are also available for season comparisons. Fields
absent from the selected source are listed as unavailable; player profile
fields are not substituted for performance records.

## Relationships and advanced analysis

`/api/eda/correlations` returns dynamic numeric correlation matrices,
correlations of points/rebounds/turnovers with the available win outcome, IQR
outlier counts, and sampled chart points for player-stat relationships.
Correlation is association, not causation. The advanced notebook uses
standardized features for exploratory K-Means and PCA; it also explores
random-forest feature importance only when the team-game outcome and sufficient
eligible rows exist. These models are not validated predictions and should not
be used for deployment or causal claims.

The advanced workflow was exercised on the supplied data: K-Means/PCA had
3,606 complete player summaries and 42 team identifiers available; the
exploratory importance fit had 20,000 sampled team-game rows and 65 eligible
features after excluding low-coverage and outcome-leakage fields. No
predictive-performance claim is made.

On the supplied files, team-game-row correlations with the available win
indicator were +0.399 for team points, +0.258 for rebounds and -0.118 for
turnovers (80,138 rows). Player-game-row correlations were +0.065 for points,
+0.046 for rebounds and -0.035 for turnovers (839,994 complete paired rows).
The weaker player-row associations are not directly comparable with team
correlations: individual contributions are nested within team outcomes and
the player rows do not represent independent games. These are unadjusted
Pearson correlations, not causal effects.

## Known limitations

- The game and player season labels are derived from date fields when a source
  season column is absent; the NBA season is treated as beginning in October.
- Scheduled or incomplete games with zero/missing scores are excluded from
  score-based metrics but remain part of the game-record count.
- Source files span different periods and populations. Counts from the game,
  team, player-stat and roster tables are not interchangeable.
- Per-game top performers have no arbitrary eligibility threshold; always
  examine the associated game count.
- The API reads selected analysis columns and reports completeness for those
  columns. It exposes full schemas for other source tables but deliberately
  does not claim unscanned file-level quality statistics.
- `PlayByPlay.parquet` is schema-discovered, not loaded into these first-stage
  summaries. Possession-level analysis is therefore unavailable.
- Schedule CSVs can contain future/unplayed fixtures and are not counted as
  completed games in the game-result analysis.

## Interactive notebooks and endpoints

The five notebooks under `notebooks/` use Plotly for interactive charts:

- `01_data_overview.ipynb`
- `02_game_eda.ipynb`
- `03_team_eda.ipynb`
- `04_player_eda.ipynb`
- `05_advanced_analysis.ipynb`

FastAPI JSON endpoints are `/api/eda/overview`, `/api/eda/games`,
`/api/eda/teams`, `/api/eda/players`, `/api/eda/correlations`, and
`/api/eda/seasons`. Each response is structured for dashboard use and includes
availability, source/filters, chart-ready values and unsupported analyses.
Season, team and player query filters are supported where the selected source
contains those dimensions. Run `python -m pytest backend/tests -q` for the
focused endpoint and analysis tests.
