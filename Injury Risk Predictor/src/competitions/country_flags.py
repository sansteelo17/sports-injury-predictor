"""Country name → flagcdn URL map for World Cup 2026 national teams.

football-data.org returns country names exactly like 'Cape Verde Islands',
'Congo DR', 'Curaçao'. The keys here are those exact strings (and a few
common spelling variants) so lookups stay simple. England and Scotland get
the GB subdivision codes because they compete as separate FAs.
"""

from __future__ import annotations


# ISO 3166-1 alpha-2 codes (lowercase) keyed by the country name as it
# appears in our WC inference frame. GB-* subdivisions for home nations.
WC_COUNTRY_ISO: dict[str, str] = {
    "Algeria": "dz",
    "Argentina": "ar",
    "Australia": "au",
    "Austria": "at",
    "Belgium": "be",
    "Bosnia-Herzegovina": "ba",
    "Brazil": "br",
    "Canada": "ca",
    "Cape Verde Islands": "cv",
    "Colombia": "co",
    "Congo DR": "cd",
    "Croatia": "hr",
    "Curaçao": "cw",
    "Curacao": "cw",
    "Czechia": "cz",
    "Ecuador": "ec",
    "Egypt": "eg",
    "England": "gb-eng",
    "France": "fr",
    "Germany": "de",
    "Ghana": "gh",
    "Haiti": "ht",
    "Iran": "ir",
    "Iraq": "iq",
    "Ivory Coast": "ci",
    "Japan": "jp",
    "Jordan": "jo",
    "Mexico": "mx",
    "Morocco": "ma",
    "Netherlands": "nl",
    "New Zealand": "nz",
    "Norway": "no",
    "Panama": "pa",
    "Paraguay": "py",
    "Portugal": "pt",
    "Qatar": "qa",
    "Saudi Arabia": "sa",
    "Scotland": "gb-sct",
    "Senegal": "sn",
    "South Africa": "za",
    "South Korea": "kr",
    "Spain": "es",
    "Sweden": "se",
    "Switzerland": "ch",
    "Tunisia": "tn",
    "Turkey": "tr",
    "United States": "us",
    "USA": "us",
    "Uruguay": "uy",
    "Uzbekistan": "uz",
}


def flag_url(country: str, size: str = "w160") -> str | None:
    """Return a flagcdn URL for ``country``, or None if unknown.

    Size accepts flagcdn width tokens like 'w40', 'w80', 'w160', 'w320'.
    """
    iso = WC_COUNTRY_ISO.get(country)
    if not iso:
        return None
    return f"https://flagcdn.com/{size}/{iso}.png"
