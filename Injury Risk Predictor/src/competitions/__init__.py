"""Competition registry. Competition is a first-class dimension across the
data model, API, and frontend; club leagues and international tournaments are
both Competitions, distinguished by ``competition_type``."""

from .registry import (
    Competition,
    CompetitionCapabilities,
    CompetitionType,
    PREMIER_LEAGUE,
    LA_LIGA,
    WORLD_CUP_2026,
    all_competitions,
    for_id,
    from_league_label,
    default_club_competition_id,
    resolve,
)

__all__ = [
    "Competition",
    "CompetitionCapabilities",
    "CompetitionType",
    "PREMIER_LEAGUE",
    "LA_LIGA",
    "WORLD_CUP_2026",
    "all_competitions",
    "for_id",
    "from_league_label",
    "default_club_competition_id",
    "resolve",
]
