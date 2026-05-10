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
