"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowLeftRight, CalendarClock, CircleHelp, Search, Sparkles } from "lucide-react";
import { apiGet } from "@/lib/api";
import type {
  GameForecast,
  HistoricalGame,
  PlayerAnalysis,
  PlayerForecast,
  PredictionPlayer,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
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
  SelectField,
} from "@/components/ui";

export function GamePredictionsPage({
  seasons,
  teams,
}: {
  seasons: string[];
  teams: Array<{ value: string; label: string }>;
}) {
  const [mode, setMode] = useState<"forecast" | "history">("forecast");
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [historicTeam, setHistoricTeam] = useState("");
  const [season, setSeason] = useState("");
  const [gameDate, setGameDate] = useState("");
  const [receivedAt, setReceivedAt] = useState<string | null>(null);

  const forecast = useMutation({
    mutationFn: (params: { home: string; away: string; date: string }) => {
      const query = new URLSearchParams({
        home_team: params.home,
        away_team: params.away,
      });
      if (params.date) query.set("game_date", params.date);
      return apiGet<GameForecast>(`/api/predictions/games?${query.toString()}`);
    },
    onSuccess: () => setReceivedAt(new Date().toISOString()),
  });

  const historyQuery = new URLSearchParams({ limit: "100" });
  if (historicTeam) historyQuery.set("team", historicTeam);
  if (season) historyQuery.set("season", season);
  const history = useQuery({
    queryKey: ["predictions", "historical", historicTeam, season],
    queryFn: ({ signal }) =>
      apiGet<{ available: boolean; reason?: string; games: HistoricalGame[] }>(
        `/api/predictions/historical/games?${historyQuery.toString()}`,
        signal,
      ),
    enabled: mode === "history",
  });

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="STAGE 03 / GAME FORECASTS"
        title="Model the matchup."
        description="Inspect saved test-set results or request a clearly labeled hypothetical matchup forecast."
      />
      <div className="mode-switch">
        <button className={mode === "forecast" ? "mode-active" : ""} type="button" onClick={() => setMode("forecast")}>
          <Sparkles size={15} /> Hypothetical forecast
        </button>
        <button className={mode === "history" ? "mode-active" : ""} type="button" onClick={() => setMode("history")}>
          <CalendarClock size={15} /> Historical test results
        </button>
      </div>

      {mode === "forecast" ? (
        <div className="page-stack">
          <Panel className="forecast-builder">
            <PanelHeading
              title="Build a matchup"
              detail="Teams are sourced from the trained model's historical data; no fixture is assumed."
              action={<span className="badge badge-warm">NOT A SCHEDULED GAME</span>}
            />
            {teams.length < 2 ? (
              <EmptyState title="Trained team data unavailable" detail="Train the game models before requesting a forecast." />
            ) : (
              <>
                <div className="matchup-form">
                  <SelectField
                    label="Home team"
                    value={home}
                    onChange={setHome}
                    options={teams.filter((team) => team.value !== away)}
                    placeholder="Select home team"
                  />
                  <span className="versus-mark">VS</span>
                  <SelectField
                    label="Away team"
                    value={away}
                    onChange={setAway}
                    options={teams.filter((team) => team.value !== home)}
                    placeholder="Select away team"
                  />
                  <label className="field">
                    <span>Forecast date (optional)</span>
                    <input type="date" value={gameDate} onChange={(event) => setGameDate(event.target.value)} />
                  </label>
                  <Button
                    variant="primary"
                    type="button"
                    className="forecast-submit"
                    disabled={!home || !away || home === away || forecast.isPending}
                    onClick={() => forecast.mutate({ home, away, date: gameDate })}
                  >
                    {forecast.isPending ? "Calculating…" : "Run forecast"} <ArrowLeftRight size={15} />
                  </Button>
                </div>
                <div className="notice notice-warm">
                  <CircleHelp size={16} />
                  <span>
                    No matching schedule feed is available. This is a hypothetical
                    model forecast based on historical team data, not a live game
    or confirmed fixture.
                  </span>
                </div>
              </>
            )}
          </Panel>
          {forecast.isError && <ErrorState message={forecast.error.message} />}
          {forecast.data && (
            <ForecastResult forecast={forecast.data} receivedAt={receivedAt} />
          )}
        </div>
      ) : (
        <Panel>
          <PanelHeading
            title="Chronological holdout"
            detail="These are saved predictions for games in the test window, shown with their actual final scores."
            action={<span className="badge badge-neutral">BACKTEST · NOT LIVE</span>}
          />
          <div className="filters-panel compact-filters">
            <div className="filter-controls">
              <SelectField label="Team" value={historicTeam} onChange={setHistoricTeam} options={teams} />
              <SelectField
                label="Season"
                value={season}
                onChange={setSeason}
                options={[...seasons].reverse().map((value) => ({ value, label: value }))}
              />
            </div>
          </div>
          {history.isPending ? (
            <LoadingState label="Loading saved holdout predictions" />
          ) : history.isError ? (
            <ErrorState message={history.error.message} onRetry={() => void history.refetch()} />
          ) : history.data?.games.length ? (
            <HistoricalGameTable rows={history.data.games} />
          ) : (
            <EmptyState title="No historical holdout games match these filters" detail={history.data?.reason} />
          )}
        </Panel>
      )}
    </div>
  );
}

function ForecastResult({
  forecast,
  receivedAt,
}: {
  forecast: GameForecast;
  receivedAt: string | null;
}) {
  const featureRows = forecast.important_features.slice(0, 7).map((item) => ({
    feature: item.feature.replaceAll("_", " "),
    importance: item.importance,
  }));
  return (
    <div className="page-stack">
      <Panel className="forecast-result">
        <div className="forecast-result-top">
          <div>
            <span className="badge badge-warm">HYPOTHETICAL FORECAST</span>
            <h2>{forecast.away_team} <span>at</span> {forecast.home_team}</h2>
            <p>Model knowledge through {formatDate(forecast.knowledge_cutoff)} · requested game date {formatDate(forecast.game_date)}</p>
          </div>
          <div className="winner-tag"><small>MODEL PICK</small><strong>{forecast.predicted_winner}</strong></div>
        </div>
        <div className="probability-row">
          <div className="probability-side">
            <span>{forecast.home_team}</span>
            <strong>{formatPercent(forecast.home_win_probability, 100)}</strong>
          </div>
          <div className="probability-track">
            <span style={{ width: `${forecast.home_win_probability * 100}%` }} />
          </div>
          <div className="probability-side probability-away">
            <span>{forecast.away_team}</span>
            <strong>{formatPercent(forecast.away_win_probability, 100)}</strong>
          </div>
        </div>
        <div className="forecast-scoreline">
          <div><span>PROJECTED SCORE</span><strong>{formatNumber(forecast.predicted_away_points, 1)} <i>—</i> {formatNumber(forecast.predicted_home_points, 1)}</strong><small>{forecast.away_team} <span>·</span> {forecast.home_team}</small></div>
          <div><span>EXPECTED MARGIN</span><strong>{forecast.predicted_winning_margin > 0 ? "+" : ""}{formatNumber(forecast.predicted_winning_margin, 1)}</strong><small>Home perspective</small></div>
          <div><span>TOTAL POINTS</span><strong>{formatNumber(forecast.predicted_total_points, 1)}</strong><small>Derived from predicted scores</small></div>
          <div><span>MODEL CONFIDENCE</span><strong>{formatPercent(forecast.model_confidence, 100)}</strong><small>Maximum class probability</small></div>
        </div>
      </Panel>
      <div className="two-column-grid">
        <BarChartPanel
          title="Feature contribution ranking"
          detail={`Native feature importance · ${forecast.winner_model}`}
          data={featureRows}
          xKey="feature"
          yKey="importance"
          horizontal
        />
        <Panel>
          <PanelHeading title="Forecast provenance" detail="Use the model vintage and data availability to interpret this output." />
          <div className="provenance-list">
            <div><span>Winner model</span><strong>{forecast.winner_model}</strong></div>
            <div><span>Home score model</span><strong>{forecast.score_models.home}</strong></div>
            <div><span>Away score model</span><strong>{forecast.score_models.away}</strong></div>
            <div><span>Model vintage</span><strong>Through {formatDate(forecast.knowledge_cutoff)}</strong></div>
            <div><span>Received at</span><strong>{receivedAt ? new Date(receivedAt).toLocaleString() : "—"}</strong></div>
          </div>
          <div className="notice notice-neutral">
            <CircleHelp size={16} /><span>{forecast.lineup_or_availability_status}</span>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function HistoricalGameTable({ rows }: { rows: HistoricalGame[] }) {
  const [sortKey, setSortKey] = useState<"date" | "confidence" | "error">("date");
  const ordered = [...rows].sort((a, b) => {
    if (sortKey === "date") return b.date.localeCompare(a.date);
    if (sortKey === "confidence") return b.predicted_home_probability - a.predicted_home_probability;
    return Math.abs(a.actual_winning_margin - a.predicted_winning_margin)
      - Math.abs(b.actual_winning_margin - b.predicted_winning_margin);
  });
  return (
    <>
      <div className="table-toolbar">
        <span>{formatNumber(rows.length)} saved prediction rows</span>
        <label className="inline-select">Sort by
          <select value={sortKey} onChange={(event) => setSortKey(event.target.value as typeof sortKey)}>
            <option value="date">Most recent</option>
            <option value="confidence">Home probability</option>
            <option value="error">Margin error</option>
          </select>
        </label>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead><tr><th>Date / season</th><th>Matchup</th><th>Actual</th><th>Predicted</th><th>Home probability</th><th>Margin error</th></tr></thead>
          <tbody>{ordered.map((row, index) => (
            <tr key={`${row.game_id ?? index}-${row.date}`}>
              <td>{formatDate(row.date)}<small className="table-sub">{row.season}</small></td>
              <td><span className="muted">{row.away_team}</span> at <strong>{row.home_team}</strong></td>
              <td>{formatNumber(row.away_score)}–{formatNumber(row.home_score)}</td>
              <td>{formatNumber(row.predicted_away_score, 0)}–{formatNumber(row.predicted_home_score, 0)}</td>
              <td>{formatPercent(row.predicted_home_probability, 100)}</td>
              <td>{formatNumber(Math.abs(row.actual_winning_margin - row.predicted_winning_margin), 1)} pts</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </>
  );
}

export function PlayerPredictionsPage({
  teams,
}: {
  teams: Array<{ value: string; label: string }>;
}) {
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [selectedPlayer, setSelectedPlayer] = useState("");
  const [opponent, setOpponent] = useState("");
  const [home, setHome] = useState(true);
  const [playerRequestError, setPlayerRequestError] = useState("");

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedSearch(search.trim()), 250);
    return () => window.clearTimeout(timer);
  }, [search]);

  const players = useQuery({
    queryKey: ["predictions", "player-search", debouncedSearch],
    queryFn: ({ signal }) =>
      apiGet<{ available: boolean; players: PredictionPlayer[] }>(
        `/api/predictions/players/search?q=${encodeURIComponent(debouncedSearch)}&limit=20`,
        signal,
      ),
    enabled: debouncedSearch.length >= 2,
  });
  const forecast = useMutation({
    mutationFn: () => {
      const params = new URLSearchParams({
        player: selectedPlayer,
        home: String(home),
      });
      if (opponent) params.set("opponent", opponent);
      return apiGet<PlayerForecast>(`/api/predictions/players?${params.toString()}`);
    },
    onError: (error) => setPlayerRequestError(error.message),
    onSuccess: () => setPlayerRequestError(""),
  });
  const selected = players.data?.players.find((player) => player.player_id === selectedPlayer);
  const history = useQuery({
    queryKey: ["eda", "player-history", selectedPlayer],
    queryFn: ({ signal }) =>
      apiGet<PlayerAnalysis>(
        `/api/eda/players?player=${encodeURIComponent(selectedPlayer)}`,
        signal,
      ),
    enabled: Boolean(selectedPlayer),
  });
  const historyRow = history.data?.season_performance.filter(
    (row) => row.player === selected?.player || row.player === selectedPlayer,
  ) ?? [];
  const historyChart = historyRow.map((row) => ({
    season: String(row.season ?? ""),
    points: typeof row.points === "number" ? row.points : null,
  })).filter((row) => row.season);

  return (
    <div className="page-stack">
      <PageHeading
        eyebrow="STAGE 03 / PLAYER FORECASTS"
        title="Project the player line."
        description="Forecast supported box-score statistics from lagged history. Availability and lineup status are not supplied."
      />
      <Panel>
        <PanelHeading
          title="Player context"
          detail="Search the saved player model state, then choose optional opponent and court context."
        />
        <div className="player-form">
          <label className="field player-search-field">
            <span>Search player</span>
            <div className="input-with-icon"><Search size={16} /><input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Enter at least 2 characters" /></div>
          </label>
          <SelectField
            label="Player"
            value={selectedPlayer}
            onChange={setSelectedPlayer}
            options={(players.data?.players ?? []).map((player) => ({
              value: player.player_id,
              label: `${player.player} · ${player.prior_games} prior`,
            }))}
            placeholder={players.isPending && debouncedSearch.length >= 2 ? "Searching…" : "Choose result"}
          />
          <SelectField label="Opponent" value={opponent} onChange={setOpponent} options={teams} placeholder="Not specified" />
          <label className="field">
            <span>Venue context</span>
            <select value={home ? "home" : "away"} onChange={(event) => setHome(event.target.value === "home")}>
              <option value="home">Home</option>
              <option value="away">Away</option>
            </select>
          </label>
          <Button variant="primary" type="button" className="player-submit" disabled={!selectedPlayer || forecast.isPending} onClick={() => forecast.mutate()}>
            {forecast.isPending ? "Calculating…" : "Run player forecast"} <Sparkles size={15} />
          </Button>
        </div>
        {players.isError && <ErrorState message={players.error.message} />}
        {playerRequestError && <ErrorState message={playerRequestError} />}
        <div className="notice notice-warm">
          <CircleHelp size={16} />
          <span>
            This statistical expectation is not confirmation of player availability.
            No injury report, confirmed lineup, or rotation feed is connected.
          </span>
        </div>
      </Panel>

      {forecast.data && selected && (
        <div className="page-stack">
          <Panel>
            <div className="forecast-result-top">
              <div><span className="badge badge-neutral">MODEL EXPECTATION</span><h2>{forecast.data.player}</h2><p>{selected.prior_games} prior games in available history · data through {formatDate(forecast.data.knowledge_cutoff)}</p></div>
              <div className="forecast-meta">
                {forecast.data.opponent
                  ? `vs ${teams.find((team) => team.value === forecast.data.opponent)?.label ?? forecast.data.opponent}`
                  : "Opponent not specified"}
                <strong>{forecast.data.historical_matchup_games} prior matchup games</strong>
              </div>
            </div>
            <div className="player-stat-grid">
              {Object.entries(forecast.data.predicted_statistics).map(([metric, value]) => {
                const interval = forecast.data?.empirical_90_percent_intervals[metric];
                return (
                  <div className="player-stat" key={metric}>
                    <span>{metric.replaceAll("_", " ")}</span>
                    <strong>{formatNumber(value, 1)}</strong>
                    <small>{interval ? `${formatNumber(interval[0], 1)} – ${formatNumber(interval[1], 1)} empirical 90% interval` : "Interval unavailable"}</small>
                  </div>
                );
              })}
            </div>
            <p className="footnote">{forecast.data.interval_method}.</p>
          </Panel>
          {history.isPending ? (
            <LoadingState label="Loading player history" />
          ) : history.isError ? (
            <ErrorState message={history.error.message} onRetry={() => void history.refetch()} />
          ) : (
            <TrendChart
              title="Season scoring history"
              detail="Historical player averages from the box-score source; not a forecast."
              data={historyChart}
              xKey="season"
              yKey="points"
              yLabel="Points"
            />
          )}
        </div>
      )}
    </div>
  );
}
