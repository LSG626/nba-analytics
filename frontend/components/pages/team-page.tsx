"use client";

import { useState } from "react";
import { useQuery, useQueries } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import type { TeamAnalysis } from "@/lib/api";
import { BarChartPanel, MultiTrendChart } from "@/components/charts";
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

interface TeamSnapshot {
  available: boolean;
  team: string;
  team_id: string;
  as_of: string;
  games: number;
  win_percentage: number | null;
  elo: number | null;
  average_points_for: number | null;
  average_points_against: number | null;
  net_rating_proxy: number | null;
  supported_ratings: Record<string, number>;
  ratings_source: string | null;
}

export function TeamPage({
  seasons,
  teams,
}: {
  seasons: string[];
  teams: Array<{ value: string; label: string }>;
}) {
  const [season, setSeason] = useState("");
  const [selectedTeam, setSelectedTeam] = useState("");
  const teamQuery = new URLSearchParams();
  if (season) teamQuery.set("season", season);
  if (selectedTeam) {
    const matched = teams.find((team) => team.value === selectedTeam);
    if (matched) teamQuery.set("team", matched.label);
  }
  const teamData = useQuery({
    queryKey: ["eda", "teams", "team-page", season, selectedTeam],
    queryFn: ({ signal }) =>
      apiGet<TeamAnalysis>(
        `/api/eda/teams${teamQuery.size ? `?${teamQuery.toString()}` : ""}`,
        signal,
      ),
  });
  const snapshots = useQueries({
    queries: (selectedTeam ? [selectedTeam] : []).map((teamId) => ({
      queryKey: ["prediction-team", teamId],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        apiGet<TeamSnapshot>(`/api/predictions/teams?team=${encodeURIComponent(teamId)}`, signal),
      enabled: Boolean(teamId),
    })),
  });

  if (teamData.isPending) return <LoadingState label="Loading team performance" />;
  if (teamData.isError) {
    return <ErrorState message={teamData.error.message} onRetry={() => void teamData.refetch()} />;
  }
  const data = teamData.data;
  if (!data.available) {
    return <Panel><EmptyState title="Team analytics unavailable" detail={data.reason} /></Panel>;
  }
  const selectedRecord = selectedTeam
    ? data.teams.find((team) => team.team_id === selectedTeam || team.team === teams.find((item) => item.value === selectedTeam)?.label)
    : undefined;
  const allSnapshots = snapshots
    .map((query) => query.data)
    .filter((team): team is TeamSnapshot => Boolean(team?.available));
  const seasonPerformance = data.season_performance.filter((row) => row.win_percentage !== null);
  const chartSeasons = [...new Set(seasonPerformance.map((row) => row.season))].sort().slice(-10);
  const seriesTeams = selectedRecord
    ? [selectedRecord.team]
    : [...new Set(seasonPerformance.map((row) => row.team))].slice(0, 4);
  const trend = chartSeasons.map((year) => {
    const point: Record<string, string | number | null> = { season: year };
    for (const teamName of seriesTeams) {
      const record = seasonPerformance.find((item) => item.season === year && item.team === teamName);
      point[teamName] = record?.win_percentage ?? null;
    }
    return point;
  });
  const sortedTeams = [...data.teams].sort((a, b) => (b.win_percentage ?? -Infinity) - (a.win_percentage ?? -Infinity));
  const ratingMetrics = [
    { key: "offensive_rating", label: "Offensive rating" },
    { key: "defensive_rating", label: "Defensive rating" },
    { key: "net_rating", label: "Net rating" },
    { key: "pace", label: "Pace" },
  ];

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="TEAM INTELLIGENCE"
        title="Every possession, in context."
        description="Compare team results, scoring, and available rating fields from the supplied team and game tables."
        action={
          <SelectField
            label="Season"
            value={season}
            onChange={setSeason}
            options={[...seasons].reverse().map((value) => ({ value, label: value }))}
          />
        }
      />
      <Panel>
        <div className="team-toolbar">
          <div><p className="eyebrow">TEAM COMPARISON</p><h2>{selectedRecord?.team ?? "League snapshot"}</h2><p>Select a team to inspect its rating snapshot and season record.</p></div>
          <SelectField label="Team" value={selectedTeam} onChange={setSelectedTeam} options={teams} placeholder="All teams" />
        </div>
        <div className="team-rating-grid">
          {(selectedTeam ? allSnapshots.slice(0, 1) : allSnapshots).map((snapshot) => (
            <div className="team-snapshot-card" key={snapshot.team_id}>
              <div className="team-snapshot-top">
                <strong>{snapshot.team}</strong><span>As of {formatDate(snapshot.as_of)}</span>
              </div>
              <div className="team-snapshot-stats">
                <div><span>WIN RATE</span><strong>{formatPercent(snapshot.win_percentage)}</strong></div>
                <div><span>ELO</span><strong>{formatNumber(snapshot.elo, 0)}</strong></div>
                <div><span>PTS FOR</span><strong>{formatNumber(snapshot.average_points_for, 1)}</strong></div>
                <div><span>PTS AGAINST</span><strong>{formatNumber(snapshot.average_points_against, 1)}</strong></div>
                <div><span>NET POINTS</span><strong>{formatNumber(snapshot.net_rating_proxy, 1)}</strong></div>
              </div>
            </div>
          ))}
          {selectedTeam && !allSnapshots.length && (
            <EmptyState title="Team strength snapshot unavailable" detail="No model-derived team record is available for this selection." />
          )}
        </div>
      </Panel>

      <div className="two-column-grid">
        <BarChartPanel
          title="Win percentage"
          detail="Derived from available team-level records; filtered to the selected season when provided."
          data={sortedTeams.slice(0, 12).map((team) => ({ team: team.team, win_percentage: team.win_percentage }))}
          xKey="team"
          yKey="win_percentage"
          horizontal
        />
        <MultiTrendChart
          title="Season win-rate progression"
          detail={selectedTeam ? "Selected team by season." : "Top four available teams by label."}
          data={trend}
          xKey="season"
          yLabel="Win rate (%)"
          series={seriesTeams.slice(0, 4).map((name, index) => ({ key: name, label: name, color: ["#173e35", "#b84c29", "#6f8f83", "#c49335"][index] }))}
        />
      </div>
      <Panel>
        <PanelHeading
          title="Supported team ratings"
          detail={allSnapshots[0]?.ratings_source ?? "Only metrics detected in the team statistics source are shown."}
        />
        {allSnapshots.length ? (
          <div className="rating-grid">
            {ratingMetrics.map((metric) => {
              const found = allSnapshots
                .filter((team) => Number.isFinite(team.supported_ratings[metric.key]))
                .map((team) => ({ team: team.team, value: team.supported_ratings[metric.key] }));
              return (
                <div className="rating-cell" key={metric.key}>
                  <span>{metric.label}</span>
                  <strong>{found.length ? formatNumber(found[0].value, 1) : "—"}</strong>
                  <small>{found.length ? found[0].team : "Not present in source"}</small>
                </div>
              );
            })}
          </div>
        ) : (
          <EmptyState title="Select a team to load its model snapshot" detail="Rating summaries are returned per team by the backend." />
        )}
        <p className="footnote">Elo and net points are derived indicators, not official NBA ratings. Rating fields shown above are historical source summaries.</p>
      </Panel>
      <Panel>
        <PanelHeading title="Team records" detail={`${formatNumber(data.game_count)} team records in this scope`} />
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr><th>Team</th><th>Games</th><th>Win rate</th><th>Points scored</th><th>Points conceded</th><th>Home record</th><th>Away record</th><th>Longest win run</th></tr></thead>
            <tbody>{sortedTeams.slice(0, 80).map((team) => (
              <tr key={team.team_id ?? team.team}>
                <td className="table-primary">{team.team}</td>
                <td>{formatNumber(team.games)}</td>
                <td>{formatPercent(team.win_percentage)}</td>
                <td>{formatNumber(team.average_points_scored, 1)}</td>
                <td>{formatNumber(team.average_points_conceded, 1)}</td>
                <td>{formatPercent(team.home_win_percentage)}</td>
                <td>{formatPercent(team.away_win_percentage)}</td>
                <td>{formatNumber(team.longest_winning_streak)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Panel>
      {data.unavailable_analyses?.length ? (
        <Panel><PanelHeading title="Data limitations" /><ul className="limitation-list">{data.unavailable_analyses.map((note) => <li key={note}>{note}</li>)}</ul></Panel>
      ) : null}
    </div>
  );
}
