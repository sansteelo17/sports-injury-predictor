export type CompetitionType = 'club' | 'international';

export interface CompetitionCapabilities {
  has_fpl: boolean;
  has_club_acwr_thresholds: boolean;
  has_team_badges: boolean;
  standings_kind: 'league_table' | 'group_stage' | 'none';
  risk_calibration_cohort: string;
  acwr_spike_threshold: number;
  fixture_label: string;
}

export interface Competition {
  id: string;
  name: string;
  type: CompetitionType;
  capabilities: CompetitionCapabilities;
}

export interface PlayerSummary {
  name: string;
  team: string;
  position: string;
  shirt_number: number | null;
  risk_level: 'High' | 'Medium' | 'Low' | 'Unknown';
  risk_probability: number;
  archetype: string;
  minutes_played: number;
  is_starter: boolean;
  player_image_url: string | null;
  days_since_last_injury: number;
  is_currently_injured: boolean;
  injury_news: string | null;
  chance_of_playing: number | null;
  competition_id: string;
  competition_type: CompetitionType;
}

export interface InjuryRecord {
  date: string | null;
  body_area: string;
  injury_type: string;
  injury_raw: string;
  severity_days: number;
  games_missed: number;
}

export interface RiskFactors {
  previous_injuries: number;
  total_days_lost: number;
  days_since_last_injury: number;
  avg_days_per_injury: number;
}

export interface ModelPredictions {
  ensemble: number;
  lgb: number;
  xgb: number;
  catboost: number;
}

export interface ImpliedOdds {
  american: string;
  decimal: number;
  fractional: string;
  implied_prob: number;
}

export interface ScoringOdds {
  score_probability: number;
  involvement_probability: number;
  goals_per_90: number;
  assists_per_90: number;
  american: string;
  decimal: number;
  fractional: string;
  availability_factor: number;
  analysis?: string | null;
}

export interface FPLValue {
  tier: string;
  tier_emoji: string;
  verdict: string;
  position_insight: string | null;
  adjusted_value: number;
  goals_per_90: number;
  assists_per_90: number;
  price: number;
  risk_factor: number;
}

export interface CleanSheetOdds {
  clean_sheet_probability: number;
  goals_conceded_per_game: number;
  american: string;
  decimal: number;
  availability_factor: number;
  analysis?: string | null;
}

export interface NextFixture {
  opponent: string;
  is_home: boolean;
  match_time: string | null;
  clean_sheet_odds: string | null;
  win_probability: number | null;
  fixture_insight: string | null;
  difficulty: number | null;
}

export interface UpcomingFixture {
  opponent: string;
  is_home: boolean;
  difficulty: number;
  match_time: string | null;
}

export interface YaraResponse {
  response_text: string;
  fpl_tip: string;
  market_probability: number | null;
  yara_probability: number;
  market_odds_decimal: number | null;
  bookmaker: string | null;
}

export interface BookmakerOddsLine {
  bookmaker: string;
  decimal_odds: number;
  implied_probability: number;
  source: string | null;
}

export interface BookmakerConsensus {
  market_type: 'score' | 'clean_sheet';
  market_label: string;
  average_decimal: number;
  average_probability: number;
  summary_text: string;
  market_line: string;
  lines: BookmakerOddsLine[];
}

export interface LabDriver {
  name: string;
  value: string | number;
  impact: 'risk_increasing' | 'protective' | 'neutral';
  explanation: string;
}

export interface TechnicalDetails {
  model_agreement: number;
  methodology: string;
  feature_highlights: { name: string; value: number }[];
}

export interface LabNotes {
  summary: string;
  key_drivers: LabDriver[];
  technical: TechnicalDetails;
}

export interface FPLPointsProjection {
  expected_points: number;
  base_points: number;
  injury_discount_pct: number;
  fixture_multiplier: number;
  confidence: 'high' | 'medium' | 'low';
  breakdown: string;
}

export interface RiskComparison {
  squad_avg_risk: number;
  position_avg_risk: number;
  squad_rank: number;
  squad_total: number;
  position_group: string;
  position_rank: number;
  position_total: number;
}

export interface PlayerImportance {
  score: number;
  tier: string;
  ownership_pct: number | null;
  price: number | null;
  price_tier: string | null;
  captaincy_proxy_pct: number | null;
  role_importance: string | null;
  form_signal: string | null;
  h2h_signal: string | null;
  summary: string;
}

export interface PlayerRisk {
  name: string;
  team: string;
  position: string;
  league: string;
  shirt_number: number | null;
  age: number;
  risk_level: 'High' | 'Medium' | 'Low' | 'Unknown';
  risk_probability: number;
  archetype: string;
  archetype_description: string;
  factors: RiskFactors;
  model_predictions: ModelPredictions;
  recommendations: string[];
  story: string;
  implied_odds: ImpliedOdds;
  last_injury_date: string | null;
  fpl_insight: string | null;
  scoring_odds: ScoringOdds | null;
  fpl_value: FPLValue | null;
  clean_sheet_odds: CleanSheetOdds | null;
  next_fixture: NextFixture | null;
  bookmaker_consensus: BookmakerConsensus | null;
  yara_response: YaraResponse | null;
  lab_notes: LabNotes | null;
  risk_percentile: number | null;
  player_image_url: string | null;
  team_badge_url: string | null;
  is_currently_injured: boolean;
  injury_news: string | null;
  chance_of_playing: number | null;
  upcoming_fixtures: UpcomingFixture[] | null;
  injury_records: InjuryRecord[];
  acwr: number | null;
  acute_load: number | null;
  chronic_load: number | null;
  spike_flag: boolean | null;
  fpl_points_projection: FPLPointsProjection | null;
  risk_comparison: RiskComparison | null;
  player_importance: PlayerImportance | null;
  competition_id: string;
  competition_type: CompetitionType;
  international_context: InternationalContext | null;
  news: NewsItem[];
}

export interface NewsItem {
  title: string;
  summary: string;
  source: string;
  url: string;
  published: string | null;
}

export interface InternationalContext {
  country: string;
  club_team: string | null;
  club_league: string | null;
  caps: number | null;
  intl_goals: number | null;
  tournament_role: 'Starter' | 'Squad' | 'Unknown';
  group: string | null;
  next_opponent: string | null;
  next_is_home: boolean | null;
  next_utc_date: string | null;
  next_stage: string | null;
  has_risk_features: boolean;
  summary: string;
  club_minutes: number | null;
  club_appearances: number | null;
  club_goals: number | null;
  club_assists: number | null;
  club_goals_per_90: number | null;
  club_assists_per_90: number | null;
  fifa_rating: number | null;
}

export interface TeamNextFixture {
  opponent: string;
  is_home: boolean;
  match_time: string | null;
  clean_sheet_odds: string | null;
  win_probability: number | null;
  fixture_insight: string | null;
  moneyline_1x2?: TeamMoneylineBook[];
}

export interface TeamMoneylineBook {
  bookmaker: string;
  home: string;
  draw: string;
  away: string;
  source?: string;
}

export interface TeamOverview {
  team: string;
  total_players: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  avg_risk: number;
  players: PlayerSummary[];
  team_badge_url: string | null;
  next_fixture: TeamNextFixture | null;
  competition_id: string;
  competition_type: CompetitionType;
}

// FPL Types
export interface LeagueStanding {
  id: number;
  name: string;
  short_name: string;
  position: number;
  played: number;
  wins: number;
  draws: number;
  losses: number;
  points: number;
  form: string | null;
  strength: number;
}

export interface GameweekSummary {
  gameweek: number;
  name: string;
  deadline: string | null;
  is_current: boolean;
  is_next: boolean;
  fixture_count: number;
  double_gameweek_teams: string[];
  featured_matches: string[];
}

export interface FPLInsights {
  current_gameweek: number | null;
  standings: LeagueStanding[];
  upcoming_gameweeks: GameweekSummary[];
  has_double_gameweek: boolean;
}

export interface WhatIfProjection {
  player_name: string;
  current_risk: number;
  projected_risk: number;
  scenario: string;
  delta: number;
  acwr_current: number;
  acwr_projected: number;
}

// Real League Standings
export interface TeamStanding {
  name: string;
  short_name: string;
  position?: number;
  points: number;
  played: number;
  form?: string;
  distance_from_top?: number;
  distance_from_safety?: number;
}

export interface StandingsSummary {
  leader: TeamStanding;
  second: TeamStanding;
  gap_to_second: number;
  safety_points: number;
  selected_team?: TeamStanding;
}

// La Liga Standings
export interface LaLigaStandingRow {
  position: number;
  name: string;
  full_name: string;
  badge_url: string | null;
  played: number;
  won: number;
  draw: number;
  lost: number;
  goals_for: number;
  goals_against: number;
  goal_difference: number;
  points: number;
  form: string | null;
}

// FPL Squad Sync
export interface FPLSquadPlayer extends PlayerSummary {
  is_captain: boolean;
  is_vice_captain: boolean;
  squad_position: number;
  multiplier: number;
}

export interface FPLSquadEntry {
  team_name: string;
  manager_name: string;
  total_points: number;
  gameweek: number;
  gameweek_points: number;
}

export interface FPLSquadSync {
  entry: FPLSquadEntry;
  players: FPLSquadPlayer[];
  unmatched: string[];
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  avg_risk: number;
  is_gw_finished: boolean;
}
