# Yara Studio — local social pipeline

Local Python replacement for the Gumloop "matchday data prep" flow plus the
downstream render/post loop. No Gumloop, no RPA, no headless browser.

```
fetch teams -> team overviews -> flatten candidates -> top N by risk
  -> per-player /risk enrich -> OpenAI journalist scoring
  -> rank by story_score -> render PIL cards -> write candidates.json
  -> optional post (text + card image) via the yara_autoposter clients
```

## Files

- `yara_studio.py` — the orchestrator (CLI).
- `yara_card.py` — `build_risk_card(data, out_path)`, a parameterized version of
  the old `node3_risk_card.py` prototype. Degrades gracefully (see caveats).
- Reuses `yara_autoposter.py` for the Twitter/Reddit clients, prominence
  ranking, and gameweek lookup.

## Run

```bash
# Dry run, no LLM spend, 3 candidates
python yara_studio.py --top-n 3 --no-journalist --dry-run --post

# Full run, post the top candidate with its card image
python yara_studio.py --league "Premier League" --top-n 30 --post

# Just produce candidates.json + cards, never post
python yara_studio.py --top-n 30 --no-journalist
```

Flags: `--league`, `--top-n`, `--gameweek`, `--no-journalist`, `--no-render`,
`--post`, `--dry-run`.

Output goes to `$YARA_OUTPUT_DIR` (default `studio_output/`):
`candidates.json` plus a PNG per top-5 candidate.

## Env

| Var | Default | Notes |
|-----|---------|-------|
| `YARA_API_BASE` | `https://www.yaraspeaks.com/api` | Point at a local `uvicorn` for dev |
| `OPENAI_API_KEY` | — | Required unless `--no-journalist` |
| `OPENAI_MODEL` | `gpt-5` | Set to an id your account actually exposes |
| `YARA_FONT_DIR` | `/home/user/fonts` | Inter `.ttf`s; falls back to PIL default font |
| `YARA_CUTOUT_DIR` | — | Optional `"<player name>.png"` transparent cutouts |
| `YARA_OUTPUT_DIR` | `studio_output` | Artifact + card output dir |

Twitter/Reddit credentials are read by `yara_autoposter.py` (`TWITTER_*`,
`REDDIT_*`, `DRY_RUN`, `TWITTER_ENABLED`, `REDDIT_ENABLED`).

## Caveats (read before relying on it)

- **Card cutouts.** The card's left panel wants a per-player transparent PNG in
  `YARA_CUTOUT_DIR` named `"<name>.png"`. Without one it renders a clean accent
  panel with the player's initials. Auto-generating cutouts is a separate,
  harder problem and is intentionally not attempted here.
- **No web search.** The journalist scores from the API payload only; `sources`
  comes back empty. Chat Completions has no native web search and the Responses
  API was deliberately avoided. Add later if needed.
- **Image tweets** use the Twitter v1.1 media-upload endpoint via tweepy, which
  requires elevated/v1.1 access on the app. Text-only posting falls back
  automatically if media upload fails.
