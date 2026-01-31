"""Dashboard components for player injury risk visualization."""

from .player_dashboard import (
    get_latest_snapshot,
    build_player_dashboard,
    panel_player_overview,
    panel_injury_risk,
    panel_severity_projection,
    panel_archetype,
    panel_top_drivers,
    panel_training_flag,
    panel_minutes_guidance,
    panel_recommendation,
)

__all__ = [
    # Main entry points
    "get_latest_snapshot",
    "build_player_dashboard",
    # Individual panels
    "panel_player_overview",
    "panel_injury_risk",
    "panel_severity_projection",
    "panel_archetype",
    "panel_top_drivers",
    "panel_training_flag",
    "panel_minutes_guidance",
    "panel_recommendation",
]
