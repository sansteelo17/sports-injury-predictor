"""Competition registry.

A ``Competition`` carries an id (slug), display name, type (club vs
international), and a ``CompetitionCapabilities`` block that tells callers
which features are meaningful for that competition (FPL, club ACWR thresholds,
standings source). The API and frontend route on capabilities rather than
hardcoding league names.

``from_league_label`` bridges the legacy free-text ``league`` column on
``inference_df`` to a competition id so the existing PL and La Liga data keeps
working without a migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


CompetitionType = str  # "club" | "international"


@dataclass(frozen=True)
class CompetitionCapabilities:
    """Boolean (and small enum) flags consumed by routing/UI/narrative layers.

    Why a capabilities block instead of branching on ``competition.id``:
    every new competition would otherwise add another ``if id == "..."``
    fork. Capabilities let the same code handle PL, La Liga, and the World
    Cup the moment the capability matrix is filled in.
    """

    has_fpl: bool = False
    has_club_acwr_thresholds: bool = True
    has_team_badges: bool = True
    standings_kind: str = "league_table"  # "league_table" | "group_stage" | "none"
    risk_calibration_cohort: str = "self"  # competition id to normalize within; "self" = own competition
    acwr_spike_threshold: float = 1.8  # club default; tournaments widen this
    fixture_label: str = "Gameweek"  # "Gameweek" | "Matchday" | "Round"


@dataclass(frozen=True)
class Competition:
    id: str
    name: str
    type: CompetitionType
    capabilities: CompetitionCapabilities = field(default_factory=CompetitionCapabilities)
    # Strings that may appear in inference_df.league for this competition.
    # Matched case-insensitively, trimmed.
    league_label_aliases: tuple = ()

    @property
    def is_club(self) -> bool:
        return self.type == "club"

    @property
    def is_international(self) -> bool:
        return self.type == "international"


PREMIER_LEAGUE = Competition(
    id="premier-league",
    name="Premier League",
    type="club",
    capabilities=CompetitionCapabilities(
        has_fpl=True,
        has_club_acwr_thresholds=True,
        has_team_badges=True,
        standings_kind="league_table",
        risk_calibration_cohort="self",
        acwr_spike_threshold=1.8,
        fixture_label="Gameweek",
    ),
    league_label_aliases=("premier league", "epl", "english premier league"),
)


LA_LIGA = Competition(
    id="la-liga",
    name="La Liga",
    type="club",
    capabilities=CompetitionCapabilities(
        has_fpl=False,
        has_club_acwr_thresholds=True,
        has_team_badges=True,
        standings_kind="league_table",
        risk_calibration_cohort="self",
        acwr_spike_threshold=1.8,
        fixture_label="Matchday",
    ),
    league_label_aliases=("la liga", "laliga", "primera división", "primera division"),
)


BUNDESLIGA = Competition(
    id="bundesliga",
    name="Bundesliga",
    type="club",
    capabilities=CompetitionCapabilities(
        has_fpl=False,
        has_club_acwr_thresholds=True,
        has_team_badges=True,
        standings_kind="league_table",
        risk_calibration_cohort="self",
        acwr_spike_threshold=1.8,
        # Bundesliga is 34 matchdays not 38, but the label is the same.
        fixture_label="Matchday",
    ),
    league_label_aliases=("bundesliga", "1. bundesliga", "german bundesliga"),
)


WORLD_CUP_2026 = Competition(
    id="world-cup-2026",
    name="FIFA World Cup 2026",
    type="international",
    capabilities=CompetitionCapabilities(
        has_fpl=False,
        # Tournament density differs from club season; widen the spike threshold
        # and surface that in narrative copy rather than silently reusing 1.8.
        has_club_acwr_thresholds=False,
        has_team_badges=True,  # flags
        standings_kind="group_stage",
        risk_calibration_cohort="self",
        acwr_spike_threshold=2.0,
        fixture_label="Matchday",
    ),
    league_label_aliases=("fifa world cup 2026", "world cup 2026", "world cup"),
)


_REGISTRY: Dict[str, Competition] = {
    c.id: c for c in (PREMIER_LEAGUE, LA_LIGA, BUNDESLIGA, WORLD_CUP_2026)
}


def all_competitions() -> List[Competition]:
    return list(_REGISTRY.values())


def for_id(competition_id: Optional[str]) -> Optional[Competition]:
    if not competition_id:
        return None
    return _REGISTRY.get(str(competition_id).strip().lower())


def default_club_competition_id() -> str:
    """Used to backfill rows without an explicit competition_id."""
    return PREMIER_LEAGUE.id


def from_league_label(label) -> Optional[Competition]:
    """Map the legacy free-text ``league`` column to a Competition.

    Returns ``None`` for unrecognised labels so callers can decide whether to
    default (e.g. to Premier League) or treat as unknown.
    """
    if label is None:
        return None
    try:
        # Handle pandas NaN without importing pandas in this module.
        if label != label:  # NaN check
            return None
    except Exception:
        pass
    text = str(label).strip().lower()
    if not text or text == "nan":
        return None
    for comp in _REGISTRY.values():
        if text in comp.league_label_aliases:
            return comp
    # Loose contains-match fallback for "la liga" variants etc.
    for comp in _REGISTRY.values():
        for alias in comp.league_label_aliases:
            if alias in text or text in alias:
                return comp
    return None


def resolve(competition_id: Optional[str], league_label=None) -> Competition:
    """Best-effort resolver: explicit id wins, else infer from league label,
    else fall back to the default club competition (PL).
    """
    by_id = for_id(competition_id)
    if by_id is not None:
        return by_id
    by_label = from_league_label(league_label)
    if by_label is not None:
        return by_label
    return _REGISTRY[default_club_competition_id()]
