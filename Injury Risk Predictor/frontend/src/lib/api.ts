import { PlayerSummary, PlayerRisk, TeamOverview, FPLInsights, LeagueStanding, StandingsSummary, WhatIfProjection, FPLSquadSync, LaLigaStandingRow } from '@/types/api';

const rawApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/+$/, '');
const API_BASE = rawApiUrl
  ? (rawApiUrl.endsWith('/api') ? rawApiUrl : `${rawApiUrl}/api`)
  : '/api';

async function fetchAPI<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return res.json();
}

export async function getPlayers(team?: string, riskLevel?: string): Promise<PlayerSummary[]> {
  const params = new URLSearchParams();
  if (team) params.append('team', team);
  if (riskLevel) params.append('risk_level', riskLevel);

  const query = params.toString() ? `?${params.toString()}` : '';
  return fetchAPI<PlayerSummary[]>(`/players${query}`);
}

export async function getPlayerRisk(playerName: string): Promise<PlayerRisk> {
  return fetchAPI<PlayerRisk>(`/players/${encodeURIComponent(playerName)}/risk`);
}

export async function getTeams(league?: string): Promise<string[]> {
  const q = league ? `?league=${encodeURIComponent(league)}` : '';
  return fetchAPI<string[]>(`/teams${q}`);
}

export async function getTeamOverview(teamName: string): Promise<TeamOverview> {
  return fetchAPI<TeamOverview>(`/teams/${encodeURIComponent(teamName)}/overview`);
}

export async function getArchetypes(): Promise<Record<string, string>> {
  return fetchAPI<Record<string, string>>('/archetypes');
}

// FPL API functions
export async function getFPLInsights(): Promise<FPLInsights> {
  return fetchAPI<FPLInsights>('/fpl/insights');
}

export async function getLeagueStandings(): Promise<LeagueStanding[]> {
  return fetchAPI<LeagueStanding[]>('/fpl/standings');
}

export async function getDoubleGameweeks(): Promise<Record<string, string[]>> {
  return fetchAPI<Record<string, string[]>>('/fpl/double-gameweeks');
}

export async function getStandingsSummary(team?: string): Promise<StandingsSummary> {
  const query = team ? `?team=${encodeURIComponent(team)}` : '';
  return fetchAPI<StandingsSummary>(`/standings/summary${query}`);
}

export async function getTeamBadges(): Promise<Record<string, string>> {
  return fetchAPI<Record<string, string>>('/teams/badges');
}

export async function getWhatIf(playerName: string, scenario: 'rest_next' | 'play_all'): Promise<WhatIfProjection> {
  const param = scenario === 'rest_next' ? 'rest_next=true' : 'play_all=true';
  return fetchAPI<WhatIfProjection>(`/players/${encodeURIComponent(playerName)}/what-if?${param}`);
}

export async function getFPLSquad(teamId: string): Promise<FPLSquadSync> {
  return fetchAPI<FPLSquadSync>(`/fpl/squad/${encodeURIComponent(teamId)}`);
}

export async function getLaLigaStandings(): Promise<LaLigaStandingRow[]> {
  return fetchAPI<LaLigaStandingRow[]>('/la-liga/standings');
}
