import { PlayerSummary, PlayerRisk, TeamOverview, FPLInsights, LeagueStanding } from '@/types/api';

const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api`
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

export async function getTeams(): Promise<string[]> {
  return fetchAPI<string[]>('/teams');
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
