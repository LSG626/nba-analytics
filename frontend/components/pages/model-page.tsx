"use client";

import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type {
  HistoricalGame,
  ModelPerformance,
} from "@/lib/api";
import {
  BarChartPanel,
  ScatterPanel,
  TrendChart,
} from "@/components/charts";
import {
  EmptyState,
  ErrorState,
  formatNumber,
  formatPercent,
  LoadingState,
  PageHeading,
  Panel,
  PanelHeading,
  StatCard,
} from "@/components/ui";

export function ModelPerformancePage() {
  const performance = useQuery({
    queryKey: ["models", "performance"],
    queryFn: ({ signal }) =>
      apiGet<ModelPerformance>("/api/models/performance", signal),
  });
  const historical = useQuery({
    queryKey: ["predictions", "historical", "model-chart"],
    queryFn: ({ signal }) =>
      apiGet<{ available: boolean; games: HistoricalGame[] }>(
        "/api/predictions/historical/games?limit=500",
        signal,
      ),
  });

  if (performance.isPending) return <LoadingState label="Loading model evaluation" />;
  if (performance.isError) {
    return <ErrorState message={performance.error.message} onRetry={() => void performance.refetch()} />;
  }
  const report = performance.data;
  if (!report.available || !report.winner) {
    return <Panel><EmptyState title="Model metrics are unavailable" detail="Train models with the supplied datasets to produce an evaluation report." /></Panel>;
  }
  const winner = report.winner;
  const test = winner.test_metrics;
  const comparison = Object.entries(winner.validation_candidates).map(([name, metrics]) => ({
    model: name,
    log_loss: metrics.log_loss,
  }));
  const accuracyComparison = Object.entries(winner.validation_candidates).map(([name, metrics]) => ({
    model: name,
    accuracy: metrics.accuracy * 100,
  }));
  const roc = test.roc_curve?.false_positive_rate.map((value, index) => ({
    false_positive_rate: value,
    true_positive_rate: test.roc_curve?.true_positive_rate[index] ?? null,
  })) ?? [];
  const calibration = test.calibration?.mean_predicted_probability.map((value, index) => ({
    predicted: value * 100,
    observed: (test.calibration?.observed_home_win_rate[index] ?? 0) * 100,
  })) ?? [];
  const importance = winner.feature_importance ?? [];
  const scoreMetrics = report.scores?.test_metrics ?? {};
  const regressionCandidates = report.scores?.validation_candidates ?? {};
  const actualPredicted = (historical.data?.games ?? [])
    .map((game) => ({ x: game.home_score, y: game.predicted_home_score }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="MODEL EVALUATION"
        title="Performance, not promises."
        description="Candidate comparisons, calibration, holdout diagnostics, and score errors from chronological evaluation."
        action={<span className="badge badge-green">TEST WINDOW UNTOUCHED</span>}
      />
      <div className="stats-grid four-up">
        <StatCard label="SELECTED MODEL" value={winner.selected_by_validation_log_loss} note="Chosen by validation log loss" />
        <StatCard label="TEST ACCURACY" value={formatPercent(test.accuracy, 100)} note={`${formatNumber(report.game_split?.test.rows)} holdout games`} accent />
        <StatCard label="TEST ROC-AUC" value={formatNumber(test.roc_auc, 3)} note="Probability ranking quality" />
        <StatCard label="TEST LOG LOSS" value={formatNumber(test.log_loss, 3)} note={`Brier ${formatNumber(test.brier_score, 3)}`} />
      </div>
      <div className="two-column-grid">
        <BarChartPanel
          title="Winner model comparison"
          detail="Validation log loss used for selection; lower is better."
          data={comparison}
          xKey="model"
          yKey="log_loss"
          color="#173e35"
        />
        <BarChartPanel
          title="Validation accuracy"
          detail="Candidate classifier accuracy before the final test window."
          data={accuracyComparison}
          xKey="model"
          yKey="accuracy"
          color="#b84c29"
        />
      </div>
      <div className="three-column-grid">
        <Panel>
          <PanelHeading title="Confusion matrix" detail="Rows = actual; columns = predicted. Away / home." />
          {test.confusion_matrix ? <ConfusionMatrix matrix={test.confusion_matrix} /> : <EmptyState title="Matrix unavailable" />}
        </Panel>
        <TrendChart
          title="ROC curve"
          detail={`Test ROC-AUC ${formatNumber(test.roc_auc, 3)}`}
          data={roc}
          xKey="false_positive_rate"
          yKey="true_positive_rate"
          yLabel="True positive rate"
          color="#b84c29"
        />
        <TrendChart
          title="Probability calibration"
          detail="Observed home-win frequency by predicted-probability bin."
          data={calibration}
          xKey="predicted"
          yKey="observed"
          yLabel="Observed (%)"
          color="#173e35"
        />
      </div>
      <div className="two-column-grid">
        <BarChartPanel
          title="Feature importance"
          detail="Model-native importance for the selected winner estimator."
          data={importance.slice(0, 12).map((item) => ({
            feature: item.feature.replaceAll("_", " "),
            importance: item.importance,
          }))}
          xKey="feature"
          yKey="importance"
          horizontal
        />
        <ScatterPanel
          title="Actual vs predicted home score"
          detail="Sample of saved chronological test predictions; points near the diagonal are closer."
          data={actualPredicted}
          xLabel="Actual home points"
          yLabel="Predicted home points"
        />
      </div>
      <Panel>
        <PanelHeading title="Score-regression evaluation" detail="Selected model and chronological test metrics by target." />
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr><th>Target</th><th>Selected model</th><th>Test MAE</th><th>Test RMSE</th><th>Test R²</th></tr></thead>
            <tbody>{Object.entries(scoreMetrics).map(([target, metrics]) => (
              <tr key={target}>
                <td className="table-primary">{target.replaceAll("_", " ")}</td>
                <td>{report.scores?.selected_models[target] ?? "Derived from scores"}</td>
                <td>{formatNumber(metrics.mae, 2)}</td>
                <td>{formatNumber(metrics.rmse, 2)}</td>
                <td>{formatNumber(metrics.r2, 3)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Panel>
      {Object.entries(regressionCandidates).length > 0 && (
        <Panel>
          <PanelHeading title="Regression candidate validation" detail="Models selected separately for home and away points, by validation MAE." />
          <div className="candidate-grid">
            {Object.entries(regressionCandidates).map(([target, candidates]) => (
              <div className="candidate-column" key={target}>
                <h3>{target.replaceAll("_", " ")}</h3>
                {Object.entries(candidates).map(([name, result]) => (
                  <div className="candidate-row" key={name}>
                    <span>{name}{report.scores?.selected_models[target] === name && <i>SELECTED</i>}</span>
                    <strong>{formatNumber(result.validation.mae, 2)} MAE</strong>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </Panel>
      )}
      {report.players?.available && (
        <Panel>
          <PanelHeading title="Player model test results" detail="Two-stage minutes and per-minute-rate estimator; held-out player games." />
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Statistic</th><th>Test rows</th><th>MAE</th><th>RMSE</th><th>R²</th><th>90% interval coverage</th></tr></thead>
              <tbody>{Object.entries(report.players.metrics ?? {}).map(([metric, result]) => (
                <tr key={metric}>
                  <td className="table-primary">{metric.replaceAll("_", " ")}</td>
                  <td>{formatNumber(result.test_rows)}</td>
                  <td>{formatNumber(result.test?.mae, 2)}</td>
                  <td>{formatNumber(result.test?.rmse, 2)}</td>
                  <td>{formatNumber(result.test?.r2, 3)}</td>
                  <td>{formatPercent(result.test_interval_coverage_90)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <p className="footnote">{report.players.availability_note} Intervals are empirical and are not guaranteed.</p>
        </Panel>
      )}
      {report.limitations?.length ? (
        <Panel><PanelHeading title="Evaluation limitations" /><ul className="limitation-list">{report.limitations.map((note) => <li key={note}>{note}</li>)}</ul></Panel>
      ) : null}
    </div>
  );
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  return (
    <div className="confusion-wrap">
      <div className="confusion-axis"><span>ACTUAL ↓</span><span>PREDICTED →</span></div>
      <div className="confusion-grid">
        <span className="confusion-label">Away</span>
        <strong>{formatNumber(matrix[0]?.[0])}</strong>
        <strong>{formatNumber(matrix[0]?.[1])}</strong>
        <span className="confusion-label">Home</span>
        <strong>{formatNumber(matrix[1]?.[0])}</strong>
        <strong>{formatNumber(matrix[1]?.[1])}</strong>
      </div>
      <div className="confusion-foot"><span>Predicted away</span><span>Predicted home</span></div>
    </div>
  );
}
