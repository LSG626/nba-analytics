"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BarChart3, CircleDot, GitCompare, Users } from "lucide-react";
import { apiGet } from "@/lib/api";
import type {
  Correlations,
  GameAnalysis,
  Overview,
  PlayerAnalysis,
  TeamAnalysis,
} from "@/lib/api";
import {
  BarChartPanel,
  CorrelationHeatmap,
  MultiTrendChart,
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
  SelectField,
} from "@/components/ui";

type AnalysisTab = "games" | "teams" | "players" | "correlations" | "seasons";

const tabItems: Array<{ id: AnalysisTab; label: string; icon: typeof CircleDot }> = [
  { id: "games", label: "Games", icon: CircleDot },
  { id: "teams", label: "Teams", icon: BarChart3 },
  { id: "players", label: "Players", icon: Users },
  { id: "correlations", label: "Correlations", icon: GitCompare },
  { id: "seasons", label: "Seasons", icon: BarChart3 },
];

export function AnalyticsPage({
  seasons,
  teams,
}: {
  seasons: string[];
  teams: Array<{ value: string; label: string }>;
}) {
  const [tab, setTab] = useState<AnalysisTab>("games");
  const [season, setSeason] = useState("");
  const [team, setTeam] = useState("");
  const [player, setPlayer] = useState("");
  const query = new URLSearchParams();
  if (season) query.set("season", season);
  if (team) query.set("team", team);
  if (player && (tab === "players" || tab === "correlations")) query.set("player", player);
  const suffix = query.size ? `?${query.toString()}` : "";

  const games = useQuery({
    queryKey: ["eda", "games", season, team],
    queryFn: ({ signal }) => apiGet<GameAnalysis>(`/api/eda/games${suffix}`, signal),
    enabled: tab === "games",
  });
  const teamAnalysis = useQuery({
    queryKey: ["eda", "teams", season, team],
    queryFn: ({ signal }) => apiGet<TeamAnalysis>(`/api/eda/teams${suffix}`, signal),
    enabled: tab === "teams",
  });
  const playerAnalysis = useQuery({
    queryKey: ["eda", "players", season, team, player],
    queryFn: ({ signal }) => apiGet<PlayerAnalysis>(`/api/eda/players${suffix}`, signal),
    enabled: tab === "players",
  });
  const correlations = useQuery({
    queryKey: ["eda", "correlations", season, team, player],
    queryFn: ({ signal }) =>
      apiGet<Correlations>(`/api/eda/correlations${suffix}`, signal),
    enabled: tab === "correlations",
  });
  const seasonsQuery = useQuery({
    queryKey: ["eda", "seasons", season, team, player],
    queryFn: ({ signal }) =>
      apiGet<Record<string, unknown>>(`/api/eda/seasons${suffix}`, signal),
    enabled: tab === "seasons",
  });

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="STAGE 02 / EXPLORATORY ANALYSIS"
        title="Read the game in context."
        description="Explore scoring, team and player performance, relationships, and historical trends from the supplied datasets."
      />
      <div className="filters-panel">
        <div className="filter-intro">
          <span className="filter-mark"><BarChart3 size={16} /></span>
          <div><strong>Analysis scope</strong><small>Filters apply where the source fields support them.</small></div>
        </div>
        <div className="filter-controls">
          <SelectField
            label="Season"
            value={season}
            onChange={setSeason}
            options={[...seasons].reverse().map((value) => ({ value, label: value }))}
          />
          <SelectField
            label="Team"
            value={team}
            onChange={setTeam}
            options={teams}
          />
          {(tab === "players" || tab === "correlations") && (
            <label className="field">
              <span>Player filter</span>
              <input
                type="search"
                placeholder="Player name or ID"
                value={player}
                onChange={(event) => setPlayer(event.target.value)}
              />
            </label>
          )}
        </div>
      </div>

      <div className="tabs-row" role="tablist" aria-label="Analysis category">
        {tabItems.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            className={`tab-button${tab === item.id ? " tab-active" : ""}`}
            onClick={() => setTab(item.id)}
          >
            <item.icon size={15} /> {item.label}
          </button>
        ))}
      </div>

      {tab === "games" && (
        <QueryState
          query={games}
          retry={() => void games.refetch()}
          loading="Loading game analysis"
        >
          {(data) => <GameSection data={data} />}
        </QueryState>
      )}
      {tab === "teams" && (
        <QueryState
          query={teamAnalysis}
          retry={() => void teamAnalysis.refetch()}
          loading="Loading team analysis"
        >
          {(data) => <TeamSection data={data} />}
        </QueryState>
      )}
      {tab === "players" && (
        <QueryState
          query={playerAnalysis}
          retry={() => void playerAnalysis.refetch()}
          loading="Loading player analysis"
        >
          {(data) => <PlayerSection data={data} />}
        </QueryState>
      )}
      {tab === "correlations" && (
        <QueryState
          query={correlations}
          retry={() => void correlations.refetch()}
          loading="Loading correlations"
        >
          {(data) => <CorrelationSection data={data} />}
        </QueryState>
      )}
      {tab === "seasons" && (
        <QueryState
          query={seasonsQuery}
          retry={() => void seasonsQuery.refetch()}
          loading="Loading season comparisons"
        >
          {(data) => <SeasonSection data={data} />}
        </QueryState>
      )}
    </div>
  );
}

function QueryState<T>({
  query,
  retry,
  loading,
  children,
}: {
  query: {
    isPending: boolean;
    isError: boolean;
    error: Error | null;
    data: T | undefined;
  };
  retry: () => void;
  loading: string;
  children: (data: T) => React.ReactNode;
}) {
  if (query.isPending) return <LoadingState label={loading} />;
  if (query.isError) {
    return <ErrorState message={query.error?.message ?? "Analysis request failed."} onRetry={retry} />;
  }
  return query.data ? children(query.data) : <EmptyState title="No response from the analysis endpoint" />;
}

function UnsupportedNotes({ notes }: { notes?: string[] }) {
  if (!notes?.length) return null;
  return (
    <div className="unavailable-list">
      {notes.map((note) => <p key={note}>{note}</p>)}
    </div>
  );
}

function GameSection({ data }: { data: GameAnalysis }) {
  if (!data.available) {
    return <Panel><EmptyState title="Game analysis unavailable" detail={data.reason} /></Panel>;
  }
  const distribution = data.score_distribution?.bin_centers.map((center, index) => ({
    total_points: Math.round(center),
    games: data.score_distribution?.counts[index] ?? 0,
  })) ?? [];
  return (
    <div className="page-stack">
      <div className="stats-grid three-up">
        <div className="metric-tile"><span>GAMES IN FILTER</span><strong>{formatNumber(data.game_count)}</strong><small>{formatNumber(data.scored_game_count)} with valid scores</small></div>
        <div className="metric-tile"><span>AVERAGE TOTAL</span><strong>{formatNumber(data.average_total_points, 1)}</strong><small>{formatNumber(data.average_points_per_team, 1)} points per team</small></div>
        <div className="metric-tile"><span>HOME WIN RATE</span><strong>{formatPercent(data.home_win_percentage, 1)}</strong><small>Average winning margin {formatNumber(data.average_winning_margin, 1)}</small></div>
      </div>
      <div className="two-column-grid">
        <TrendChart
          title="Scoring by season"
          detail="Average points per team in games with valid scores."
          data={(data.scoring_by_season ?? []).map((row) => ({
            ...row,
            average_points_per_team: Number(row.average_points_per_team?.toFixed(1)),
          }))}
          xKey="season"
          yKey="average_points_per_team"
          yLabel="Points"
        />
        <BarChartPanel
          title="Game total distribution"
          detail="Histogram from the game table's valid score rows."
          data={distribution}
          xKey="total_points"
          yKey="games"
          color="#173e35"
        />
      </div>
      {data.home_away_win_distribution && (
        <Panel>
          <PanelHeading title="Home versus away" detail="Observed outcome counts from the selected game rows." />
          <div className="outcome-strip">
            {[
              { label: "Home wins", value: data.home_away_win_distribution.home_wins, color: "red" },
              { label: "Away wins", value: data.home_away_win_distribution.away_wins, color: "green" },
              { label: "Ties", value: data.home_away_win_distribution.ties, color: "neutral" },
            ].map((item) => (
              <div className="outcome-item" key={item.label}>
                <span className={`outcome-dot ${item.color}`} />
                <span>{item.label}</span>
                <strong>{formatNumber(item.value)}</strong>
              </div>
            ))}
          </div>
          {data.overtime && (
            <div className="single-stat-row">
              <span>Overtime games</span>
              <strong>{formatNumber(data.overtime.games_with_overtime)} / {formatNumber(data.overtime.games_observed)}</strong>
              <span>{formatPercent(data.overtime.frequency_percent)}</span>
            </div>
          )}
        </Panel>
      )}
      <div className="two-column-grid">
        <GameTable title="Highest-scoring games" rows={data.highest_scoring_games ?? []} />
        <GameTable title="Lowest-scoring games" rows={data.lowest_scoring_games ?? []} />
      </div>
      <UnsupportedNotes notes={data.unavailable_analyses} />
    </div>
  );
}

function GameTable({
  title,
  rows,
}: {
  title: string;
  rows: Array<Record<string, string | number | null>>;
}) {
  if (!rows.length) return <Panel><PanelHeading title={title} /><EmptyState title="No game rows available" /></Panel>;
  const first = rows[0];
  const columns = Object.keys(first).slice(-5);
  return (
    <Panel>
      <PanelHeading title={title} detail="Actual games from the supplied source." />
      <div className="table-scroll">
        <table className="data-table compact-table">
          <thead><tr>{columns.map((column) => <th key={column}>{column.replaceAll("_", " ")}</th>)}</tr></thead>
          <tbody>{rows.slice(0, 6).map((row, index) => (
            <tr key={String(row.gameId ?? row.game_id ?? index)}>
              {columns.map((column) => <td key={column}>{row[column] === null ? "—" : String(row[column])}</td>)}
            </tr>
          ))}</tbody>
        </table>
      </div>
    </Panel>
  );
}

function TeamSection({ data }: { data: TeamAnalysis }) {
  if (!data.available) return <Panel><EmptyState title="Team analysis unavailable" detail={data.reason} /></Panel>;
  const top = [...data.teams]
    .filter((row) => row.win_percentage !== null)
    .sort((left, right) => (right.win_percentage ?? -1) - (left.win_percentage ?? -1))
    .slice(0, 12);
  const bySeason = data.season_performance.filter((row) => !Number.isNaN(row.win_percentage));
  const recentSeasons = [...new Set(bySeason.map((row) => row.season))]
    .sort()
    .slice(-8);
  const seasonTeamData = recentSeasons.map((season) => {
    const row: Record<string, string | number | null> = { season };
    for (const team of top.slice(0, 4)) {
      const value = bySeason.find((item) => item.season === season && item.team === team.team);
      row[team.team] = value?.win_percentage ?? null;
    }
    return row;
  });
  return (
    <div className="page-stack">
      <div className="stats-grid three-up">
        <div className="metric-tile"><span>TEAMS IN VIEW</span><strong>{formatNumber(data.team_count)}</strong><small>Team-stat records: {formatNumber(data.game_count)}</small></div>
        <div className="metric-tile"><span>TOP WIN RATE</span><strong>{formatPercent(top[0]?.win_percentage)}</strong><small>{top[0]?.team ?? "Not available"}</small></div>
        <div className="metric-tile"><span>AVAILABLE TEAM SEASONS</span><strong>{formatNumber(new Set(bySeason.map((row) => row.season)).size)}</strong><small>Based on supported game/team fields</small></div>
      </div>
      <div className="two-column-grid">
        <BarChartPanel
          title="Team win percentage"
          detail="Team-level win rates calculated from available team records."
          data={top.map((row) => ({ team: row.team, win_percentage: row.win_percentage }))}
          xKey="team"
          yKey="win_percentage"
          horizontal
        />
        <MultiTrendChart
          title="Season win-rate trends"
          detail="Top teams in this scope; unsupported team-seasons are left blank."
          data={seasonTeamData}
          xKey="season"
          yLabel="Win rate (%)"
          series={top.slice(0, 4).map((team, index) => ({
            key: team.team,
            label: team.team,
            color: ["#173e35", "#b84c29", "#6f8f83", "#c49335"][index],
          }))}
        />
      </div>
      <Panel>
        <PanelHeading title="Team comparison" detail="Sort by the supported performance measures in this filtered dataset." />
        <TeamTable rows={data.teams} />
      </Panel>
      <UnsupportedNotes notes={data.unavailable_analyses} />
    </div>
  );
}

function TeamTable({ rows }: { rows: TeamAnalysis["teams"] }) {
  const [sortKey, setSortKey] = useState<"win_percentage" | "average_points_scored" | "average_points_conceded">("win_percentage");
  const sorted = [...rows].sort(
    (left, right) => (right[sortKey] ?? -Infinity) - (left[sortKey] ?? -Infinity),
  );
  return (
    <>
      <div className="table-toolbar">
        <span>{formatNumber(rows.length)} team records</span>
        <label className="inline-select">Sort by
          <select value={sortKey} onChange={(event) => setSortKey(event.target.value as typeof sortKey)}>
            <option value="win_percentage">Win percentage</option>
            <option value="average_points_scored">Points scored</option>
            <option value="average_points_conceded">Points conceded</option>
          </select>
        </label>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead><tr><th>Team</th><th>Games</th><th>Wins</th><th>Win rate</th><th>Points for</th><th>Points against</th><th>Home win rate</th><th>Away win rate</th></tr></thead>
          <tbody>{sorted.slice(0, 100).map((row) => (
            <tr key={row.team_id ?? row.team}>
              <td className="table-primary">{row.team}</td>
              <td>{formatNumber(row.games)}</td>
              <td>{formatNumber(row.wins)}</td>
              <td>{formatPercent(row.win_percentage)}</td>
              <td>{formatNumber(row.average_points_scored, 1)}</td>
              <td>{formatNumber(row.average_points_conceded, 1)}</td>
              <td>{formatPercent(row.home_win_percentage)}</td>
              <td>{formatPercent(row.away_win_percentage)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </>
  );
}

function PlayerSection({ data }: { data: PlayerAnalysis }) {
  if (!data.available) return <Panel><EmptyState title="Player analysis unavailable" detail={data.reason} /></Panel>;
  const leaders = data.leaders.points ?? [];
  const pointsRebounds = data.season_performance;
  const players = data.players;
  return (
    <div className="page-stack">
      <div className="stats-grid three-up">
        <div className="metric-tile"><span>PLAYER RECORDS</span><strong>{formatNumber(data.player_count)}</strong><small>{formatNumber(data.game_count)} player-game rows</small></div>
        <div className="metric-tile"><span>SUPPORTED METRICS</span><strong>{formatNumber(data.supported_metrics?.length)}</strong><small>Detected in the source schema</small></div>
        <div className="metric-tile"><span>POINTS LEADER</span><strong className="metric-name">{leaders[0]?.player ?? "—"}</strong><small>{leaders[0] ? `${formatNumber(leaders[0].metrics.points, 1)} points per game` : "Unavailable"}</small></div>
      </div>
      <div className="two-column-grid">
        <BarChartPanel
          title="Points leaders"
          detail="Mean points per recorded player game in this scope."
          data={leaders.slice(0, 12).map((row) => ({ player: row.player, points: row.metrics.points }))}
          xKey="player"
          yKey="points"
          horizontal
        />
        <Panel>
          <PanelHeading title="Player comparison" detail="Compare available box-score averages; sort the table by supported data." />
          <PlayerTable rows={players} />
        </Panel>
      </div>
      {pointsRebounds.length > 0 && (
        <Panel>
          <PanelHeading title="Season-by-season player records" detail="Aggregates are shown only for selected player rows." />
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr>{Object.keys(pointsRebounds[0]).map((key) => <th key={key}>{key.replaceAll("_", " ")}</th>)}</tr></thead>
              <tbody>{pointsRebounds.slice(0, 60).map((row, index) => (
                <tr key={`${String(row.player)}-${String(row.season)}-${index}`}>
                  {Object.entries(row).map(([key, value]) => <td key={key}>{typeof value === "number" ? formatNumber(value, 1) : String(value ?? "—")}</td>)}
                </tr>
              ))}</tbody>
            </table>
          </div>
        </Panel>
      )}
      <UnsupportedNotes notes={data.unavailable_analyses} />
    </div>
  );
}

function PlayerTable({ rows }: { rows: PlayerAnalysis["players"] }) {
  const [sortKey, setSortKey] = useState("points");
  const [direction, setDirection] = useState<"desc" | "asc">("desc");
  const metrics = rows[0] ? Object.keys(rows[0].metrics).filter((key) => key !== "win") : [];
  const sorted = [...rows].sort((left, right) => {
    const a = left.metrics[sortKey] ?? -Infinity;
    const b = right.metrics[sortKey] ?? -Infinity;
    return direction === "desc" ? b - a : a - b;
  });
  return (
    <>
      <div className="table-toolbar">
        <span>{formatNumber(rows.length)} players</span>
        <label className="inline-select">Metric
          <select value={sortKey} onChange={(event) => setSortKey(event.target.value)}>
            {metrics.map((metric) => <option value={metric} key={metric}>{metric.replaceAll("_", " ")}</option>)}
          </select>
        </label>
        <button className="quiet-button" type="button" onClick={() => setDirection(direction === "desc" ? "asc" : "desc")}>
          {direction === "desc" ? "High to low" : "Low to high"}
        </button>
      </div>
      <div className="table-scroll player-table-scroll">
        <table className="data-table">
          <thead><tr><th>Player</th><th>Games</th>{metrics.slice(0, 6).map((metric) => <th key={metric}>{metric.replaceAll("_", " ")}</th>)}</tr></thead>
          <tbody>{sorted.slice(0, 100).map((row) => (
            <tr key={row.player_id ?? row.player}>
              <td className="table-primary">{row.player}</td>
              <td>{formatNumber(row.games)}</td>
              {metrics.slice(0, 6).map((metric) => <td key={metric}>{formatNumber(row.metrics[metric], 1)}</td>)}
            </tr>
          ))}</tbody>
        </table>
      </div>
    </>
  );
}

function CorrelationSection({ data }: { data: Correlations }) {
  if (!data.available) {
    return (
      <Panel>
        <EmptyState title="Correlation analysis unavailable" detail={data.unavailable_analyses.join(" ")} />
      </Panel>
    );
  }
  const playerRelationships = Object.entries(data.player_relationships ?? {});
  const playerCharts = playerRelationships.map(([key, relation]) => (
    <ScatterPanel
      key={key}
      title={key.replaceAll("_", " × ")}
      detail={`${relation.observations.toLocaleString()} paired player records`}
      data={relation.x.map((x, index) => ({ x, y: relation.y[index] })).filter((point) => Number.isFinite(point.y))}
      xLabel={relation.x_label}
      yLabel={relation.y_label}
    />
  ));
  const winRelations = Object.entries(data.relationships.teams ?? {}).filter(
    ([, relation]) => relation.correlation_with_win !== null,
  );
  return (
    <div className="page-stack">
      <div className="two-column-grid">
        {data.correlations.teams && <CorrelationHeatmap title="Team statistic correlations" table={data.correlations.teams} />}
        {data.correlations.players && <CorrelationHeatmap title="Player statistic correlations" table={data.correlations.players} />}
      </div>
      <div className="two-column-grid">
        {playerCharts.length ? playerCharts : (
          <Panel><PanelHeading title="Player performance relationships" /><EmptyState title="No supported player relationship fields" /></Panel>
        )}
        <Panel>
          <PanelHeading title="Team metrics and wins" detail="Correlation values are descriptive, not causal." />
          {winRelations.length ? (
            <div className="relationship-list">
              {winRelations.map(([name, relation]) => (
                <div className="relationship-row" key={name}>
                  <span>{name.replaceAll("_", " ")}</span>
                  <strong className={(relation.correlation_with_win ?? 0) >= 0 ? "positive" : "negative"}>
                    {(relation.correlation_with_win ?? 0) >= 0 ? "+" : ""}
                    {relation.correlation_with_win?.toFixed(3)}
                  </strong>
                  <small>{formatNumber(relation.observations)} paired rows</small>
                </div>
              ))}
            </div>
          ) : <EmptyState title="No team win relationship was measurable" />}
        </Panel>
      </div>
      <UnsupportedNotes notes={data.unavailable_analyses} />
    </div>
  );
}

function SeasonSection({ data }: { data: Record<string, unknown> }) {
  const scoring = Array.isArray(data.game_scoring_by_season)
    ? data.game_scoring_by_season as Array<Record<string, unknown>>
    : [];
  const teamSeasons = Array.isArray(data.team_performance_by_season)
    ? data.team_performance_by_season as Array<Record<string, unknown>>
    : [];
  const unavailable = Array.isArray(data.unavailable_analyses)
    ? data.unavailable_analyses as string[]
    : [];
  const recent = scoring.slice(-30).map((row) => ({
    season: String(row.season),
    points: Number(row.average_points_per_team),
  })).filter((row) => Number.isFinite(row.points));
  const topTeams = [...new Set(teamSeasons.map((row) => String(row.team)))].slice(0, 6);
  const selectedSeasons = [...new Set(teamSeasons.map((row) => String(row.season)))].slice(-12);
  const trends = selectedSeasons.map((season) => {
    const row: Record<string, string | number | null> = { season };
    topTeams.slice(0, 4).forEach((teamName) => {
      const record = teamSeasons.find((item) => String(item.season) === season && String(item.team) === teamName);
      row[teamName] = record && typeof record.win_percentage === "number" ? record.win_percentage : null;
    });
    return row;
  });
  return (
    <div className="page-stack">
      <div className="stats-grid three-up">
        <div className="metric-tile"><span>GAME COUNT IN SCOPE</span><strong>{formatNumber(data.game_count_in_scope as number | null)}</strong><small>From the games source</small></div>
        <div className="metric-tile"><span>TEAMS IN SCOPE</span><strong>{formatNumber(data.team_count_in_scope as number | null)}</strong><small>From team statistics</small></div>
        <div className="metric-tile"><span>SEASONS WITH SCORES</span><strong>{formatNumber(scoring.length)}</strong><small>Date-derived when source labels are absent</small></div>
      </div>
      <div className="two-column-grid">
        <TrendChart
          title="Historical scoring trend"
          detail="Average team points by season in the available game table."
          data={recent}
          xKey="season"
          yKey="points"
          yLabel="Points"
        />
        <MultiTrendChart
          title="Team win-rate trends"
          detail="Selected available teams and recent seasons."
          data={trends}
          xKey="season"
          yLabel="Win rate (%)"
          series={topTeams.slice(0, 4).map((team, index) => ({
            key: team,
            label: team,
            color: ["#173e35", "#b84c29", "#6f8f83", "#c49335"][index],
          }))}
        />
      </div>
      <Panel>
        <PanelHeading title="Season values" detail="Source-backed game scoring aggregates." />
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr><th>Season</th><th>Avg points / team</th></tr></thead>
            <tbody>{scoring.slice(-30).map((row) => (
              <tr key={String(row.season)}><td className="table-primary">{String(row.season)}</td><td>{formatNumber(row.average_points_per_team as number, 1)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </Panel>
      <UnsupportedNotes notes={unavailable} />
    </div>
  );
}

export type SeasonSource = Overview;
