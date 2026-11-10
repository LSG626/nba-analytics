# COURTSIDE AI dashboard

The responsive Next.js dashboard uses TypeScript, Tailwind CSS, React Query,
Recharts, Inter, and reusable shadcn/ui-style components. It reads all
analytics and predictions from the local FastAPI backend. Historical holdout
predictions are distinguished from hypothetical matchup forecasts, and
missing model/data outputs are shown as unavailable.

## Run locally

1. From the project root, activate the Python environment and start FastAPI:

   ```powershell
   python -m uvicorn backend.app.main:app --reload
   ```

2. In another PowerShell window:

   ```powershell
   cd C:\path\to\NBA\frontend
   npm install
   npm run dev
   ```

3. Open `http://localhost:3000`. The API defaults to
   `http://127.0.0.1:8000`; override it with
   `NEXT_PUBLIC_API_BASE_URL` when running the frontend against another local
   API address.

To train prediction models before using the prediction pages, run
`python -m scripts.train_models` from the project root. The dashboard can be
built and checked with `npm run build`, `npm run typecheck`, and `npm run lint`.
The backend permits browser requests from the default Next.js development
origins.
