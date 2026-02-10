export interface PlayerSummary {
  name: string;
  team: string;
  position: string;
  risk_level: 'High' | 'Medium' | 'Low';
  risk_probability: number;
  archetype: string;
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
  american: string;     // e.g., "-150" or "+200"
  decimal: number;      // e.g., 1.67
  fractional: string;   // e.g., "2/3"
  implied_prob: number; // The probability used
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
  story: string;  // Personalized risk narrative
  implied_odds: ImpliedOdds;  // Betting odds representation
  last_injury_date: string | null;
}

export interface TeamOverview {
  team: string;
  total_players: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  avg_risk: number;
  players: PlayerSummary[];
}
