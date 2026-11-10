"use client";

import { useQuery } from "@tanstack/react-query";
import {
  ArrowDownRight,
  ArrowUpRight,
  CircleDot,
  Database,
  Layers3,
  Users,
} from "lucide-react";
import { API_BASE, apiGet } from "@/lib/api";
import type {
  Backtesting,
  HistoricalGame,
  ModelPerformance,
  Overview,
} from "@/lib/api";
import { BarChartPanel, TrendChart } from "@/components/charts";
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
  StatCard,
} from "@/components/ui";

export function OverviewPage() {
  const overview = useQuery({
    queryKey: ["eda", "overview"],
    queryFn: ({ signal }) => apiGet<Overview>("/api/eda/overview", signal),
  });
  const performance = useQuery({
    queryKey: ["models", "performance"],
    queryFn: ({ signal }) =>
      apiGet<ModelPerformance>("/api/models/performance", signal),
  });
  const backtesting = useQuery({
    queryKey: ["backtesting"],
    queryFn: ({ signal }) => apiGet<Backtesting>("/api/backtesting/results", signal),
  });
  const history = useQuery({
    queryKey: ["predictions", "historical", "recent"],
    queryFn: ({ signal }) =>
      apiGet<{ available: boolean; games: HistoricalGame[] }>(
        "/api/predictions/historical/games?limit=8",
        signal,
      ),
  });

  if (overview.isPending) return <LoadingState label="Loading workspace overview" />;
  if (overview.isError) {
    return (
      <ErrorState
        message={overview.error.message || "The backend could not load dataset coverage."}
        onRetry={() => void overview.refetch()}
      />
    );
  }

  const data = overview.data;
  const test = performance.data?.winner?.test_metrics;
  const selectedModel =
    performance.data?.winner?.selected_by_validation_log_loss;
  const seasonal = performance.data?.seasonal_holdout_metrics?.holdout_seasons;
  const seasonTrend = seasonal
    ? Object.entries(seasonal).map(([season, result]) => ({
        season,
        accuracy: result.classification.accuracy * 100,
      }))
    : [];
  const teamRows = performance.data?.scores
    ? [
        {
          target: "Home points",
          mae: performance.data.scores.test_metrics.home_score?.mae,
        },
        {
          target: "Away points",
          mae: performance.data.scores.test_metrics.away_score?.mae,
        },
        {
          target: "Total points",
          mae: performance.data.scores.test_metrics.total_points?.mae,
        },
        {
          target: "Winning margin",
          mae: performance.data.scores.test_metrics.winning_margin?.mae,
        },
      ].filter((row): row is { target: string; mae: number } =>
        Number.isFinite(row.mae),
      )
    : [];

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="COURTSIDE / INTELLIGENCE"
        title="The numbers behind the game."
        description="A live view of the supplied NBA datasets, model evaluations, and saved historical forecasts."
        action={
          <a className="button button-secondary" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
            <Database size={15} /> API docs
          </a>
        }
      />

      <div className="stats-grid">
        <StatCard
          label="GAMES IN SOURCE"
          value={data.counts.games === null ? "—" : formatNumber(data.counts.games)}
          note={`${formatDate(data.date_coverage.start)} — ${formatDate(data.date_coverage.end)}`}
          icon={<CircleDot size={17} />}
        />
        <StatCard
          label="SEASONS COVERED"
          value={data.counts.seasons === null ? "—" : formatNumber(data.counts.seasons)}
          note={data.counts.season_labels.at(-1) ?? "Season not available"}
          icon={<Layers3 size={17} />}
        />
        <StatCard
          label="TEAMS"
          value={data.counts.teams === null ? "—" : formatNumber(data.counts.teams)}
          note="Unique teams detected in supplied records"
          icon={<ArrowUpRight size={17} />}
        />
        <StatCard
          label="PLAYERS"
          value={data.counts.players === null ? "—" : formatNumber(data.counts.players)}
          note={
            data.counts.players_in_statistics === null
              ? "Player count from available roster/stat tables"
              : `${formatNumber(data.counts.players_in_statistics)} with box-score statistics`
          }
          icon={<Users size={17} />}
        />
      </div>

      {performance.isError && (
        <ErrorState
          message="Model evaluation could not be loaded. Dataset summaries remain available."
          onRetry={() => void performance.refetch()}
        />
      )}
      {performance.isPending ? (
        <LoadingState label="Loading model evaluation" />
      ) : performance.data?.available ? (
        <>
          <div className="overview-grid">
            <Panel>
              <PanelHeading
                title="Model snapshot"
                detail={`Evaluated ${formatDate(performance.data.generated_at)} · chronological holdout`}
              />
              <div className="model-snapshot">
                <div className="snapshot-mark">AI</div>
                <div className="snapshot-main">
                  <span className="snapshot-label">SELECTED WINNER MODEL</span>
                  <strong>{selectedModel ?? "Unavailable"}</strong>
                  <span className="snapshot-sub">
                    Validation-selected · {formatNumber(performance.data.game_data?.games)} historical games
                  </span>
                </div>
                <div className="snapshot-metric">
                  <span>HOLDOUT ACCURACY</span>
                  <strong>{formatPercent(test?.accuracy)}</strong>
                  <small>ROC-AUC {formatNumber(test?.roc_auc, 3)}</small>
                </div>
                <div className="snapshot-metric">
                  <span>LOG LOSS</span>
                  <strong>{formatNumber(test?.log_loss, 3)}</strong>
                  <small>Lower is better</small>
                </div>
              </div>
              <div className="data-note">
                <span className="signal-dot" />
                <span>
                  Models use historical observations through{" "}
                  {formatDate(performance.data.game_data?.date_end)}. They do not
                  include confirmed player availability or lineup information.
                </span>
              </div>
            </Panel>
            <Panel className="coverage-panel">
              <PanelHeading title="Data coverage" detail="Observed source window" />
              <div className="coverage-range">
                <div>
                  <span>FIRST GAME</span>
                  <strong>{formatDate(data.date_coverage.start)}</strong>
                </div>
                <div className="coverage-line"><span /></div>
                <div>
                  <span>LATEST GAME</span>
                  <strong>{formatDate(data.date_coverage.end)}</strong>
                </div>
              </div>
              <div className="coverage-sources">
                {(performance.data.data_sources
                  ? Object.entries(performance.data.data_sources)
                      .filter(([key, value]) => key !== "data_directories" && typeof value === "string")
                      .map(([key, value]) => ({ key, value: String(value) }))
                  : []
                ).map((source) => (
                  <div className="source-row" key={source.key}>
                    <span>{source.key.replaceAll("_", " ")}</span>
                    <strong>{source.value}</strong>
                  </div>
                ))}
              </div>
            </Panel>
          </div>

          <div className="two-column-grid">
            <TrendChart
              title="Holdout accuracy by season"
              detail="Season labels inferred from game date; partial seasons are shown as observed."
              data={seasonTrend}
              xKey="season"
              yKey="accuracy"
              yLabel="Accuracy (%)"
              color="#b84c29"
            />
            <BarChartPanel
              title="Score model error"
              detail="Mean absolute error on the chronological holdout, in points."
              data={teamRows}
              xKey="target"
              yKey="mae"
              color="#173e35"
              horizontal
            />
          </div>
        </>
      ) : (
        <Panel>
          <PanelHeading title="Model evaluation" />
          <EmptyState
            title="No trained model report is available"
            detail={performance.data?.reason ?? "Run the local training command to generate measured model results."}
          />
        </Panel>
      )}

      <div className="two-column-grid overview-bottom">
        <Panel>
          <PanelHeading
            title="Recent holdout predictions"
            detail="Saved test-set forecasts paired with actual final scores."
            action={<span className="badge badge-neutral">HISTORICAL</span>}
          />
          {history.isPending ? (
            <LoadingState label="Loading saved prediction history" />
          ) : history.isError ? (
            <ErrorState
              message={history.error.message}
              onRetry={() => void history.refetch()}
            />
          ) : history.data?.games.length ? (
            <div className="table-scroll">
              <table className="data-table compact-table">
                <thead>
                  <tr><th>Date</th><th>Matchup</th><th>Final</th><th>Home chance</th></tr>
                </thead>
                <tbody>
                  {history.data.games.slice(0, 6).map((game, index) => (
                    <tr key={`${game.game_id ?? index}-${game.date}`}>
                      <td>{formatDate(game.date)}</td>
                      <td>{game.away_team} <span className="muted">at</span> {game.home_team}</td>
                      <td>{formatNumber(game.away_score)}–{formatNumber(game.home_score)}</td>
                      <td>{formatPercent(game.predicted_home_probability, 100)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              title="No saved historical prediction rows"
              detail="This view only shows predictions produced by the chronological test evaluation."
            />
          )}
        </Panel>
        <Panel>
          <PanelHeading
            title="Dataset quality"
            detail="Quality statistics are reported only for tables scanned by the backend."
          />
          <div className="quality-list">
            {Object.entries(data.tables)
              .filter(([, table]) => table.available)
              .map(([name, table]) => (
                <div className="quality-row" key={name}>
                  <span className="quality-icon"><Database size={15} /></span>
                  <div className="quality-main">
                    <strong>{name}</strong>
                    <span>{formatNumber(table.rows)} rows · {formatNumber(table.duplicate_rows ?? 0)} duplicates</span>
                  </div>
                  <div className="quality-value">
                    {table.completeness_percent === undefined || table.completeness_percent === null
                      ? "—"
                      : `${table.completeness_percent.toFixed(1)}%`}
                    <small>complete</small>
                  </div>
                </div>
              ))}
          </div>
          {data.limitations?.length ? (
            <p className="footnote">
              {data.limitations[0]}
            </p>
          ) : null}
        </Panel>
      </div>

      {backtesting.data?.available && (
        <div className="callout-row">
          <span className="callout-icon"><ArrowDownRight size={17} /></span>
          <p>
            Rolling-origin evaluation reports {backtesting.data.folds.length} historical
            windows. These development-period checks are separate from the final
            chronological holdout.
          </p>
        </div>
      )}
    </div>
  );
}
