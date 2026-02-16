export interface PlayerSummary {
  name: string;
  team: string;
  position: string;
  risk_level: 'High' | 'Medium' | 'Low';
  risk_probability: number;
  archetype: string;
  minutes_played: number;
  is_starter: boolean;
  player_image_url: string | null;
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
}

export interface NextFixture {
  opponent: string;
  is_home: boolean;
  match_time: string | null;
  clean_sheet_odds: string | null;
  win_probability: number | null;
  fixture_insight: string | null;
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

export interface PlayerRisk {
  name: string;
  team: string;
  position: string;
  age: number;
  risk_level: 'High' | 'Medium' | 'Low';
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
