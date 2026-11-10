"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type {
  Backtesting,
  HistoricalGame,
  ModelPerformance,
} from "@/lib/api";
import { MultiBarChart, TrendChart } from "@/components/charts";
import {
  EmptyState,
  ErrorState,
  formatDate,
  formatNumber,
  formatPercent,
  LoadingState,
  PageHeading,
  Panel,
  PanelHeading,
  SelectField,
} from "@/components/ui";

export function BacktestingPage({
  seasons,
  teams,
}: {
  seasons: string[];
  teams: Array<{ value: string; label: string }>;
}) {
  const [season, setSeason] = useState("");
  const [team, setTeam] = useState("");
  const backtest = useQuery({
    queryKey: ["backtesting"],
    queryFn: ({ signal }) => apiGet<Backtesting>("/api/backtesting/results", signal),
  });
  const performance = useQuery({
    queryKey: ["models", "performance"],
    queryFn: ({ signal }) => apiGet<ModelPerformance>("/api/models/performance", signal),
  });
  const historicalParams = new URLSearchParams({ limit: "150" });
  if (season) historicalParams.set("season", season);
  if (team) historicalParams.set("team", team);
  const historical = useQuery({
    queryKey: ["predictions", "historical", "backtesting", team, season],
    queryFn: ({ signal }) =>
      apiGet<{ available: boolean; games: HistoricalGame[] }>(
        `/api/predictions/historical/games?${historicalParams.toString()}`,
        signal,
      ),
  });

  if (backtest.isPending) return <LoadingState label="Loading rolling-origin backtest" />;
  if (backtest.isError) {
    return <ErrorState message={backtest.error.message} onRetry={() => void backtest.refetch()} />;
  }
  if (!backtest.data.available) {
    return <Panel><EmptyState title="Rolling-origin results unavailable" detail={backtest.data.reason} /></Panel>;
  }

  const foldRows = backtest.data.folds.map((fold) => ({
    fold: `Fold ${fold.fold}`,
    model_accuracy: fold.classification.model.accuracy * 100,
    baseline_accuracy: fold.classification.baseline_historical_home_rate.accuracy * 100,
    model_log_loss: fold.classification.model.log_loss,
    baseline_log_loss: fold.classification.baseline_historical_home_rate.log_loss,
  }));
  const seasonal = performance.data?.seasonal_holdout_metrics?.holdout_seasons ?? {};
  const seasonalRows = Object.entries(seasonal).map(([label, row]) => ({
    season: label,
    accuracy: row.classification.accuracy * 100,
    log_loss: row.classification.log_loss,
  }));
  const games = historical.data?.games ?? [];

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="STAGE 03 / HISTORICAL VALIDATION"
        title="Pressure-test the model."
        description="Review expanding-window backtests and inspect holdout predictions against actual game outcomes."
        action={<span className="badge badge-green">{backtest.data.folds.length} ROLLING WINDOWS</span>}
      />
      <div className="notice notice-neutral">
        <span className="notice-number">01</span>
        <span>{backtest.data.method}. Rolling folds use only data before their evaluation period; the final holdout is separate.</span>
      </div>
      <div className="two-column-grid">
        <TrendChart
          title="Rolling-window accuracy"
          detail="Selected classifier compared with a historical home-win-rate baseline."
          data={foldRows}
          xKey="fold"
          yKey="model_accuracy"
          yLabel="Accuracy (%)"
          color="#b84c29"
        />
        <MultiBarChart
          title="Log loss by window"
          detail="Model probability loss against the historical-rate baseline; lower is better."
          data={foldRows.map((row) => ({
            fold: row.fold,
            model: row.model_log_loss,
            baseline: row.baseline_log_loss,
          }))}
          xKey="fold"
          series={[
            { key: "model", label: "Selected model", color: "#173e35" },
            { key: "baseline", label: "Historical-rate baseline", color: "#b84c29" },
          ]}
        />
      </div>
      <Panel>
        <PanelHeading title="Rolling-origin folds" detail={`Classifier: ${backtest.data.classifier ?? "—"}`} />
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr><th>Fold</th><th>Training through</th><th>Evaluation window</th><th>Games</th><th>Model accuracy</th><th>Baseline accuracy</th><th>Model log loss</th><th>Baseline log loss</th></tr></thead>
            <tbody>{backtest.data.folds.map((fold) => (
              <tr key={fold.fold}>
                <td className="table-primary">0{fold.fold}</td>
                <td>{formatDate(fold.train_end)}</td>
                <td>{formatDate(fold.evaluation_start)} — {formatDate(fold.evaluation_end)}</td>
                <td>{formatNumber(fold.evaluation_rows)}</td>
                <td>{formatPercent(fold.classification.model.accuracy, 100)}</td>
                <td>{formatPercent(fold.classification.baseline_historical_home_rate.accuracy, 100)}</td>
                <td>{formatNumber(fold.classification.model.log_loss, 3)}</td>
                <td>{formatNumber(fold.classification.baseline_historical_home_rate.log_loss, 3)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Panel>
      <div className="two-column-grid">
        <TrendChart
          title="Accuracy across holdout seasons"
          detail="Descriptive subsets of the final test window; not separate tuned models."
          data={seasonalRows}
          xKey="season"
          yKey="accuracy"
          yLabel="Accuracy (%)"
          color="#173e35"
        />
        <TrendChart
          title="Holdout log loss by season"
          detail="Season is inferred from dates because the source has no season field."
          data={seasonalRows}
          xKey="season"
          yKey="log_loss"
          yLabel="Log loss"
          color="#b84c29"
        />
      </div>
      <Panel>
        <PanelHeading title="Prediction error analysis" detail="Saved test-set forecasts with actual results and score/margin errors." />
        <div className="filters-panel compact-filters">
          <div className="filter-controls">
            <SelectField label="Season" value={season} onChange={setSeason} options={[...seasons].reverse().map((value) => ({ value, label: value }))} />
            <SelectField label="Team" value={team} onChange={setTeam} options={teams} />
          </div>
        </div>
        {historical.isPending ? (
          <LoadingState label="Loading holdout predictions" />
        ) : historical.isError ? (
          <ErrorState message={historical.error.message} onRetry={() => void historical.refetch()} />
        ) : games.length ? (
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Date</th><th>Matchup</th><th>Actual score</th><th>Predicted score</th><th>Win pick</th><th>Correct</th><th>Score MAE</th><th>Margin error</th></tr></thead>
              <tbody>{games.map((game, index) => {
                const correct = game.home_win === game.predicted_home_win;
                const homeError = Math.abs(game.home_score - game.predicted_home_score);
                const awayError = Math.abs(game.away_score - game.predicted_away_score);
                return (
                  <tr key={`${game.game_id ?? index}-${game.date}`}>
                    <td>{formatDate(game.date)}<small className="table-sub">{game.season}</small></td>
                    <td><span className="muted">{game.away_team}</span> at <strong>{game.home_team}</strong></td>
                    <td>{formatNumber(game.away_score)}–{formatNumber(game.home_score)}</td>
                    <td>{formatNumber(game.predicted_away_score, 0)}–{formatNumber(game.predicted_home_score, 0)}</td>
                    <td>{game.predicted_home_win ? game.home_team : game.away_team}</td>
                    <td><span className={`result-chip ${correct ? "result-correct" : "result-wrong"}`}>{correct ? "Correct" : "Miss"}</span></td>
                    <td>{formatNumber((homeError + awayError) / 2, 1)}</td>
                    <td>{formatNumber(Math.abs(game.actual_winning_margin - game.predicted_winning_margin), 1)}</td>
                  </tr>
                );
              })}</tbody>
            </table>
          </div>
        ) : (
          <EmptyState title="No historical predictions match the filters" detail="Only the chronological test predictions are listed here." />
        )}
      </Panel>
    </div>
  );
}
