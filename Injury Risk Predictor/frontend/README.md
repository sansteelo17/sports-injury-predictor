# EPL Injury Predictor - Frontend

React + Next.js + TypeScript frontend for the EPL Injury Risk Predictor.

## Quick Start

### 1. Install dependencies

```bash
cd frontend
npm install
```

### 2. Start the API (in another terminal)

```bash
# From project root
pip install fastapi uvicorn
python -m uvicorn api.main:app --reload
```

### 3. Start the frontend

```bash
npm run dev
```

Open http://localhost:3000

## Optional analytics

To send lightweight pageview and product-usage events to Sygna, set:

```bash
NEXT_PUBLIC_SYGNA_API_URL=http://localhost:4000
NEXT_PUBLIC_SYGNA_INGEST_KEY=
```

If that variable is not set, the frontend keeps working normally and analytics stays disabled.

`NEXT_PUBLIC_SYGNA_INGEST_KEY` is only needed when Sygna is configured with `SYGNA_INGEST_SECRET`. OpenAI keys are not needed in the Yara frontend. They only belong on the Sygna server when optional AI summary polishing is enabled.

## Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── globals.css      # Tailwind + custom styles
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Main page
│   ├── components/
│   │   ├── PlayerCard.tsx   # Detailed player risk view
│   │   ├── PlayerList.tsx   # Squad player list
│   │   ├── RiskBadge.tsx    # Risk level indicator
│   │   ├── RiskMeter.tsx    # Visual risk gauge
│   │   ├── TeamOverview.tsx # Team risk summary
│   │   └── TeamSelector.tsx # Team dropdown
│   ├── lib/
│   │   └── api.ts           # API client functions
│   └── types/
│       └── api.ts           # TypeScript interfaces
├── package.json
├── tailwind.config.ts
└── tsconfig.json
```

## API Endpoints

The frontend expects these endpoints from the FastAPI backend:

- `GET /api/teams` - List all teams
- `GET /api/teams/{team}/overview` - Team risk summary
- `GET /api/players/{name}/risk` - Player risk details
- `GET /api/players` - List all players (with filters)
