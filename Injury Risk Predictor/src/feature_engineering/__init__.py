"""Feature engineering modules for injury prediction."""

from .classification import build_classification_dataset
from .negative_sampling import generate_negative_samples, get_recommended_strategy
from .temporal_features import add_temporal_features, add_fixture_density_features
from .position_features import (
    add_position_risk_features,
    add_position_workload_interaction,
    normalize_position,
    POSITION_INJURY_RISK
)
from .player_stats_features import (
    load_player_stats,
    add_player_stats_features,
    get_high_risk_indicators
)
from .severity import (
    build_severity_dataset,
    clean_severity_dataset,
    build_injury_features,
    add_severity_prediction_features,
    add_team_recovery_features,
    add_player_injury_history_features,
    get_severity_feature_descriptions,
    INJURY_TYPE_SEVERITY,
    BODY_AREA_SEVERITY
)

__all__ = [
    # Classification
    "build_classification_dataset",
    # Negative sampling
    "generate_negative_samples",
    "get_recommended_strategy",
    # Temporal features
    "add_temporal_features",
    "add_fixture_density_features",
    # Position features
    "add_position_risk_features",
    "add_position_workload_interaction",
    "normalize_position",
    "POSITION_INJURY_RISK",
    # Player stats
    "load_player_stats",
    "add_player_stats_features",
    "get_high_risk_indicators",
    # Severity features
    "build_severity_dataset",
    "clean_severity_dataset",
    "build_injury_features",
    "add_severity_prediction_features",
    "add_team_recovery_features",
    "add_player_injury_history_features",
    "get_severity_feature_descriptions",
    "INJURY_TYPE_SEVERITY",
    "BODY_AREA_SEVERITY",
]
