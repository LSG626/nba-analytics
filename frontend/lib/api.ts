export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function apiGet<T>(
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      payload &&
      typeof payload === "object" &&
      "detail" in payload &&
      typeof payload.detail === "string"
        ? payload.detail
        : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return payload as T;
}

export interface DatasetSummary {
  name: string;
  row_count?: number | null;
  size_bytes?: number;
  available?: boolean;
}

export interface Overview {
  datasets: DatasetSummary[];
  counts: {
    games: number | null;
    seasons: number | null;
    season_labels: string[];
    teams: number | null;
    players: number | null;
    players_in_statistics: number | null;
  };
  date_coverage: { start: string | null; end: string | null };
  tables: Record<
    string,
    {
      available: boolean;
      rows?: number;
      completeness_percent?: number | null;
      duplicate_rows?: number;
      missing_cells?: number;
    }
  >;
  limitations: string[];
}

export interface GameAnalysis {
  available: boolean;
  reason?: string;
  game_count?: number;
  scored_game_count?: number;
  average_points_per_team?: number | null;
  average_total_points?: number | null;
  average_winning_margin?: number | null;
  home_win_percentage?: number | null;
  home_away_win_distribution?: {
    home_wins: number;
    away_wins: number;
    ties: number;
  };
  scoring_by_season?: Array<{
    season: string;
    average_points_per_team: number;
  }>;
  score_distribution?: { bin_centers: number[]; counts: number[] };
  highest_scoring_games?: Array<Record<string, string | number | null>>;
  lowest_scoring_games?: Array<Record<string, string | number | null>>;
  overtime?: {
    games_with_overtime: number;
    games_observed: number;
    frequency_percent: number | null;
  };
  unavailable_analyses?: string[];
}

export interface TeamRecord {
  team: string;
  team_id?: string;
  games: number;
  wins: number | null;
  win_percentage: number | null;
  average_points_scored: number | null;
  average_points_conceded: number | null;
  home_win_percentage?: number | null;
  away_win_percentage?: number | null;
  longest_winning_streak?: number | null;
}

export interface TeamAnalysis {
  available: boolean;
  reason?: string;
  game_count?: number;
  team_count?: number;
  teams: TeamRecord[];
  season_performance: Array<{
    season: string;
    team: string;
    games: number;
    win_percentage: number | null;
  }>;
  unavailable_analyses?: string[];
}

export interface PlayerRecord {
  player: string;
  player_id?: string;
  games: number;
  team?: string;
  metrics: Record<string, number | null>;
  consistency: Record<string, number | null>;
}

export interface PlayerAnalysis {
  available: boolean;
  reason?: string;
  game_count?: number;
  player_count?: number;
  players: PlayerRecord[];
  leaders: Record<string, PlayerRecord[]>;
  season_performance: Array<Record<string, string | number | null>>;
  supported_metrics?: string[];
  unavailable_analyses?: string[];
}

export interface CorrelationTable {
  source: string;
  columns: string[];
  matrix: Array<Array<number | null>>;
  observations: number;
}

export interface Correlations {
  available: boolean;
  correlations: Record<string, CorrelationTable>;
  relationships: Record<
    string,
    Record<
      string,
      {
        correlation_with_win: number | null;
        observations: number;
        scatter: { x: number[]; y: number[] };
      }
    >
  >;
  player_relationships?: Record<
    string,
    {
      x_label: string;
      y_label: string;
      x: number[];
      y: number[];
      observations: number;
    }
  >;
  unavailable_analyses: string[];
}

export interface ClassMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  log_loss: number;
  brier_score: number;
  confusion_matrix?: number[][];
  roc_curve?: {
    false_positive_rate: number[];
    true_positive_rate: number[];
    thresholds: number[];
  };
  calibration?: {
    mean_predicted_probability: number[];
    observed_home_win_rate: number[];
  };
}

export interface RegressionMetrics {
  mae: number;
  rmse: number;
  r2: number;
}

export interface ModelPerformance {
  available: boolean;
  reason?: string;
  generated_at?: string;
  data_sources?: Record<string, string | string[] | null>;
  game_data?: {
    games: number;
    date_start: string;
    date_end: string;
  };
  game_split?: {
    train: { rows: number; date_start: string; date_end: string };
    validation: { rows: number; date_start: string; date_end: string };
    test: { rows: number; date_start: string; date_end: string };
  };
  winner?: {
    selected_by_validation_log_loss: string;
    validation_candidates: Record<string, ClassMetrics>;
    test_metrics: ClassMetrics & { feature_importance?: unknown };
    feature_importance?: Array<{ feature: string; importance: number }>;
  };
  scores?: {
    selected_models: Record<string, string>;
    validation_candidates: Record<
      string,
      Record<string, { validation: RegressionMetrics; test?: RegressionMetrics }>
    >;
    test_metrics: Record<string, RegressionMetrics>;
  };
  players?: {
    available: boolean;
    source?: string;
    metrics?: Record<
      string,
      {
        test?: RegressionMetrics;
        test_interval_coverage_90?: number;
        test_rows?: number;
      }
    >;
    availability_note?: string;
  };
  seasonal_holdout_metrics?: {
    holdout_seasons: Record<
      string,
      {
        games: number;
        classification: ClassMetrics;
        scores: Record<string, RegressionMetrics>;
      }
    >;
  };
  limitations?: string[];
}

export interface Backtesting {
  available: boolean;
  reason?: string;
  method?: string;
  classifier?: string;
  folds: Array<{
    fold: number;
    train_end: string;
    evaluation_start: string;
    evaluation_end: string;
    train_rows: number;
    evaluation_rows: number;
    classification: {
      model: ClassMetrics;
      baseline_historical_home_rate: ClassMetrics;
    };
  }>;
}

export interface PredictionTeam {
  team_id: string;
  team: string;
  games: number | null;
}

export interface PredictionPlayer {
  player_id: string;
  player: string;
  team_id: string | null;
  prior_games: number;
}

export interface HistoricalGame {
  date: string;
  season: string;
  game_id: string | number | null;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  home_win: number;
  predicted_home_probability: number;
  predicted_away_probability: number;
  predicted_home_win: number;
  predicted_home_score: number;
  predicted_away_score: number;
  actual_total_points: number;
  predicted_total_points: number;
  actual_winning_margin: number;
  predicted_winning_margin: number;
}

export interface PlayerForecast {
  available: boolean;
  prediction_type: string;
  player: string;
  player_id: string;
  team_id: string | null;
  opponent: string | null;
  game_date: string;
  knowledge_cutoff: string;
  predicted_statistics: Record<string, number>;
  empirical_90_percent_intervals: Record<string, [number, number]>;
  interval_method: string;
  availability_status: string;
  historical_matchup_games: number;
}

export interface GameForecast {
  available: boolean;
  prediction_type: string;
  game_date: string;
  knowledge_cutoff: string;
  home_team: string;
  away_team: string;
  home_win_probability: number;
  away_win_probability: number;
  predicted_winner: string;
  model_confidence: number;
  winner_model: string;
  predicted_home_points: number;
  predicted_away_points: number;
  predicted_total_points: number;
  predicted_winning_margin: number;
  score_models: { home: string; away: string };
  important_features: Array<{ feature: string; importance: number }>;
  lineup_or_availability_status: string;
}
