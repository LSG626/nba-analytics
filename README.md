# COURTSIDE AI

Local-first NBA analytics platform. Stage 1 provides flexible CSV ingestion,
Parquet outputs, and data-quality reporting. Stage 2 adds schema-aware EDA,
Plotly notebooks, and dashboard-ready FastAPI analysis endpoints. Stage 3 adds
chronological machine-learning training, evaluation, and inference endpoints.
Stage 4 provides a responsive, React Query-powered web dashboard for the local
analytics and model endpoints. It does not fabricate statistics or download
datasets automatically.

## Project structure

```text
courtside-ai/
├── backend/
│   ├── app/
│   │   ├── ingestion.py
│   │   ├── eda.py
│   │   └── main.py
│   └── tests/
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_game_eda.ipynb
│   ├── 03_team_eda.ipynb
│   ├── 04_player_eda.ipynb
│   └── 05_advanced_analysis.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
├── docs/
├── frontend/
│   └── app/
├── ml/
│   ├── evaluation/
│   ├── features/
│   └── models/
├── notebooks/
├── scripts/
├── tests/
├── requirements.txt
└── README.md
```

The workspace may also contain a `Raw/` folder with user-provided files. The
ingestion command defaults to `data/raw`; point `--input-dir` at `Raw` to use
that existing folder without copying large files.

## Add NBA CSV files

1. Download a dataset manually from a source such as [Kaggle NBA Games](https://www.kaggle.com/datasets/nathanlauga/nba-games),
   [Hugging Face NBA Games](https://huggingface.co/datasets/hamzas/nba-games),
   [NBA Data 2010–2024](https://github.com/NocturneBear/NBA-Data-2010-2024), or
   an NBA player-statistics dataset on Kaggle. Follow each provider's terms and
   authentication requirements.
2. Place the CSV files in `data/raw/`. Downloads requiring account access are
   never fetched by this project.
3. In the repository root, run:

   ```powershell
   python -m backend.app.ingestion
   ```

   To ingest the supplied `Raw/` directory instead:

   ```powershell
   python -m backend.app.ingestion --input-dir Raw
   ```

The pipeline handles each CSV independently, preserves its columns, trims
whitespace, treats blank strings as missing, parses date-like fields, removes
exact duplicate rows, and writes one Parquet file per input. The report records
inferred data types, missing values, invalid dates, duplicate counts, and
available player/team/score checks. Missing player or team identifiers remain
missing and are reported; the pipeline does not fill them with invented values.
The EDA service discovers schemas dynamically and uses only available fields.
It checks `data/processed/`, `data/raw/`, and the supplied legacy `Raw/`
directory, in that order. The analysis report and limitations are documented
in [docs/EDA_REPORT.md](docs/EDA_REPORT.md). Interactive Plotly notebooks are
in `notebooks/`.

## Windows setup

Python 3.10 or newer is recommended.

```powershell
cd C:\path\to\courtside-ai
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, run `.\.venv\Scripts\python.exe`
instead of activating the environment.

## Start the backend

From the repository root with the virtual environment active:

```powershell
python -m uvicorn backend.app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive API documentation
is at `http://127.0.0.1:8000/docs`.

Endpoints:

- `GET /api/health`
- `GET /api/datasets/status`
- `GET /api/datasets`
- `GET /api/datasets/{dataset_name}/metadata`
- `GET /api/datasets/quality`
- `POST /api/datasets/upload` (multipart field name: `file`)
- `GET /api/eda/overview`
- `GET /api/eda/games?season=&team=`
- `GET /api/eda/teams?season=&team=`
- `GET /api/eda/players?season=&team=&player=`
- `GET /api/eda/correlations?season=&team=&player=`
- `GET /api/eda/seasons?season=&team=&player=`
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

EDA responses are structured JSON for frontend dashboards and contain
`available`, supported metrics, chart-ready values and unavailable analyses.
Filter parameters are applied only where the dataset contains the requested
dimension.

An upload saves the CSV to `data/raw/`; run ingestion separately to create
processed outputs and refresh the quality report.

## Train prediction models

Run the chronological training/evaluation workflow from the project root:

```powershell
python -m scripts.train_models
```

By default, training searches `data/processed/`, `data/raw/`, and then the
provided `Raw/` folder. It compares winner and score model families using a
chronological validation window, reports a later untouched test window, and
saves model/evaluation/prediction artifacts locally. No predictions are
available until training succeeds. Player predictions do not include injury or
confirmed lineup information. See [docs/ML_METHODOLOGY.md](docs/ML_METHODOLOGY.md)
and [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) for methodology and
measured evaluation results.

## Frontend

The Next.js dashboard reads live EDA, evaluation, and prediction data from
FastAPI. It distinguishes historical holdout predictions from hypothetical
forecasts and reports unavailable datasets rather than substituting sample
values. Start it from the frontend directory:

```powershell
cd C:\path\to\NBA\frontend
npm install
npm run dev
```

Run the backend in another terminal with
`python -m uvicorn backend.app.main:app --reload`; open
`http://localhost:3000`. See [`frontend/README.md`](frontend/README.md) for
configuration and build commands.

## Tests

```powershell
python -m pytest backend/tests -q
```

## Dependencies

Backend/data: FastAPI, Uvicorn, python-multipart, Pandas, NumPy, PyArrow,
Plotly, ipykernel, scikit-learn, XGBoost, LightGBM, and Joblib.
Tests use pytest and httpx.
Frontend: Next.js, React, TypeScript, Tailwind CSS, Recharts, and the shadcn/ui
supporting packages listed in `frontend/package.json`.
