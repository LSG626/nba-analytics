# Backend

The FastAPI application exposes local dataset status, metadata, EDA, and
trained-model prediction endpoints. It does not use a database or download
data. EDA and model training look in
`data/processed/`, `data/raw/`, and then the supplied `Raw/` directory.

Run the API from the repository root:

```powershell
python -m uvicorn backend.app.main:app --reload
```

Ingest local CSV files and produce Parquet plus a JSON quality report:

```powershell
python -m backend.app.ingestion
```

Pass `--input-dir`, `--output-dir`, and `--report` to override the defaults.

EDA routes:

- `GET /api/eda/overview`
- `GET /api/eda/games?season=&team=`
- `GET /api/eda/teams?season=&team=`
- `GET /api/eda/players?season=&team=&player=`
- `GET /api/eda/correlations?season=&team=&player=`
- `GET /api/eda/seasons?season=&team=&player=`

See [`docs/EDA_REPORT.md`](../docs/EDA_REPORT.md) for measured dataset
findings, coverage and limitations, and `notebooks/` for the Plotly notebooks.
Prediction routes require locally trained artifacts; run
`python -m scripts.train_models` from the project root. See
[`docs/ML_METHODOLOGY.md`](../docs/ML_METHODOLOGY.md) and
[`docs/MODEL_EVALUATION.md`](../docs/MODEL_EVALUATION.md) for details.
The dashboard frontend runs from `frontend/` with `npm install` and
`npm run dev`; see [`frontend/README.md`](../frontend/README.md).

Prediction and model routes:

- `GET /api/predictions/games?home_team=&away_team=&game_date=`
- `GET /api/predictions/players?player=&opponent=&home=&game_date=`
- `GET /api/predictions/teams?team=`
- `GET /api/predictions/teams/list`
- `GET /api/predictions/players/search?q=&limit=`
- `GET /api/predictions/historical/games?team=&season=&limit=`
- `GET /api/predictions/latest`
- `GET /api/models/performance`
- `GET /api/models/comparison`
- `GET /api/backtesting/results`
