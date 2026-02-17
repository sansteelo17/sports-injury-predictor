'use client';

import { TeamOverview as TeamOverviewType } from '@/types/api';
import { Users, AlertTriangle, AlertCircle, CheckCircle, TrendingUp, TrendingDown, Calendar } from 'lucide-react';

interface TeamOverviewProps {
  team: TeamOverviewType;
  darkMode?: boolean;
}

export function TeamOverview({ team, darkMode = true }: TeamOverviewProps) {
  const highPct = Math.round((team.high_risk_count / team.total_players) * 100);
  const medPct = Math.round((team.medium_risk_count / team.total_players) * 100);
  const lowPct = Math.round((team.low_risk_count / team.total_players) * 100);

  // Key positions that typically affect match odds (attackers & midfielders)
  const isKeyPosition = (position: string) => {
    const pos = position.toLowerCase();
    return pos.includes('forward') ||
           pos.includes('winger') ||
           pos.includes('striker') ||
           pos.includes('attack') ||
           pos.includes('midfield');
  };

  // Get regular starters - use minutes if available, otherwise use key positions as proxy
  const hasMinutesData = team.players.some(p => p.minutes_played > 0);
  const keyPlayers = hasMinutesData
    ? team.players.filter(p => p.is_starter)
    : team.players.filter(p => isKeyPosition(p.position));

  // Get key players at high risk
  const keyPlayersAtRisk = keyPlayers
    .filter(p => p.risk_level === 'High')
    .sort((a, b) => b.risk_probability - a.risk_probability)
    .slice(0, 3)
    .map(p => p.name.split(' ').pop()); // Get last name

  // Get healthy key players (low risk)
  const keyPlayersHealthy = keyPlayers
    .filter(p => p.risk_level === 'Low')
    .sort((a, b) => a.risk_probability - b.risk_probability)
    .slice(0, 3)
    .map(p => p.name.split(' ').pop());

  // Show negative insight if 2+ key players are at high risk
  const showNegativeInsight = keyPlayersAtRisk.length >= 2;

  // Show positive insight if most key players are healthy
  const healthyKeyCount = keyPlayers.filter(p => p.risk_level === 'Low').length;
  const showPositiveInsight = !showNegativeInsight &&
    healthyKeyCount >= Math.ceil(keyPlayers.length * 0.5) &&
    keyPlayersHealthy.length >= 2;

  const bookmakerLogos: Record<string, string> = {
    SkyBet: '/bookies/skybet.svg',
    'Paddy Power': '/bookies/paddypower.svg',
    Betway: '/bookies/betway.svg',
  };

  const moneylineRows = team.next_fixture?.moneyline_1x2 ?? [];
  const dynamicMarketInsight = team.next_fixture?.fixture_insight?.trim() ?? '';
  const showDynamicInsight = dynamicMarketInsight.length > 0;

  return (
    <div className={`holo-card rounded-2xl overflow-hidden ${
      darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-lg'
    }`}>
      {/* Header */}
      <div className={`px-4 sm:px-6 py-4 sm:py-5 ${
        darkMode
          ? 'bg-gradient-to-r from-[#1f1f1f] to-[#141414] border-b border-[#1f1f1f]'
          : 'bg-gradient-to-r from-emerald-600 to-emerald-800 text-white'
      }`}>
        <div className="flex items-center gap-3">
          {team.team_badge_url && (
            <img src={team.team_badge_url} alt="" className="w-8 h-8 object-contain" />
          )}
          <h2 className={`text-lg sm:text-xl font-bold ${darkMode ? 'text-white' : ''}`}>{team.team}</h2>
        </div>
        <p className={`text-xs sm:text-sm ${darkMode ? 'text-gray-500' : 'text-emerald-100'}`}>
          {team.total_players} players analyzed
        </p>
      </div>

      {/* Risk Distribution Bar */}
      <div className={`px-4 sm:px-6 py-3 sm:py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
        <div className={`text-xs sm:text-sm mb-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          Squad Risk Distribution
        </div>
        <div className="flex h-6 rounded-full overflow-hidden">
          {team.high_risk_count > 0 && (
            <div
              className="bg-red-500 flex items-center justify-center text-white text-xs font-medium"
              style={{ width: `${highPct}%` }}
            >
              {highPct > 10 ? `${highPct}%` : ''}
            </div>
          )}
          {team.medium_risk_count > 0 && (
            <div
              className="bg-amber-400 flex items-center justify-center text-amber-900 text-xs font-medium"
              style={{ width: `${medPct}%` }}
            >
              {medPct > 10 ? `${medPct}%` : ''}
            </div>
          )}
          {team.low_risk_count > 0 && (
            <div
              className="bg-[#86efac] flex items-center justify-center text-[#0a0a0a] text-xs font-medium"
              style={{ width: `${lowPct}%` }}
            >
              {lowPct > 10 ? `${lowPct}%` : ''}
            </div>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className={`grid grid-cols-2 sm:grid-cols-4 ${darkMode ? 'divide-x divide-[#1f1f1f]' : 'divide-x divide-gray-100'}`}>
        <StatBox
          icon={<Users className={darkMode ? 'text-gray-500' : 'text-gray-400'} size={18} />}
          value={team.total_players}
          label="Players"
          darkMode={darkMode}
        />
        <StatBox
          icon={<AlertTriangle className="text-red-500" size={18} />}
          value={team.high_risk_count}
          label="High Risk"
          highlight="red"
          darkMode={darkMode}
        />
        <StatBox
          icon={<AlertCircle className="text-amber-500" size={18} />}
          value={team.medium_risk_count}
          label="Medium"
          highlight="amber"
          darkMode={darkMode}
        />
        <StatBox
          icon={<CheckCircle className={darkMode ? 'text-[#86efac]' : 'text-emerald-600'} size={18} />}
          value={team.low_risk_count}
          label="Low Risk"
          highlight="green"
          darkMode={darkMode}
        />
      </div>

      {/* Average Risk */}
      <div className={`px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between ${
        darkMode ? 'bg-[#0a0a0a]' : 'bg-gray-50'
      }`}>
        <div className={`flex items-center gap-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          <TrendingUp size={18} />
          <span className="text-sm">Squad Average Risk</span>
        </div>
        <span className={`text-lg font-bold ${
          team.avg_risk >= 0.45 ? 'text-red-500' :
          team.avg_risk >= 0.30 ? 'text-amber-500' : (darkMode ? 'text-[#86efac]' : 'text-emerald-600')
        }`}>
          {Math.round(team.avg_risk * 100)}%
        </span>
      </div>

      {/* Next Fixture */}
      {team.next_fixture && (
        <div className={`px-4 py-3 ${
          darkMode ? 'bg-cyan-500/10 border-t border-cyan-500/20' : 'bg-cyan-50 border-t border-cyan-200'
        }`}>
          <div className="flex items-center gap-3">
            <Calendar className="text-cyan-400 flex-shrink-0" size={16} />
            <div className="flex-1 min-w-0">
              <div className={`text-xs font-medium ${darkMode ? 'text-cyan-300' : 'text-cyan-700'}`}>
                Next Match
              </div>
              <div className={`text-sm font-bold truncate ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                {team.next_fixture.is_home
                  ? `vs ${team.next_fixture.opponent} (H)`
                  : `@ ${team.next_fixture.opponent} (A)`}
              </div>
              {team.next_fixture.match_time && (
                <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  {new Date(team.next_fixture.match_time).toLocaleDateString('en-GB', {
                    weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
                  })}
                </div>
              )}
            </div>
            {team.next_fixture.win_probability != null && (
              <div className="text-right flex-shrink-0">
                <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>Win</div>
                <div className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {Math.round(team.next_fixture.win_probability * 100)}%
                </div>
              </div>
            )}
          </div>

          {moneylineRows.length > 0 && (
            <div className={`mt-3 rounded-lg p-3 ${
              darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f]' : 'bg-white border border-cyan-100'
            }`}>
              <div className="flex items-center justify-between mb-2">
                <div className={`text-[11px] uppercase tracking-wider ${darkMode ? 'text-cyan-300' : 'text-cyan-700'}`}>
                  Moneyline (1X2)
                </div>
                <div className={`grid grid-cols-3 gap-3 text-[10px] uppercase tracking-wider ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                  <span className="text-center">1</span>
                  <span className="text-center">X</span>
                  <span className="text-center">2</span>
                </div>
              </div>

              <div className="space-y-2">
                {moneylineRows.map((line) => (
                  <div key={line.bookmaker} className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2 min-w-0">
                      <img
                        src={bookmakerLogos[line.bookmaker] ?? '/bookies/default.svg'}
                        alt={line.bookmaker}
                        className="w-6 h-6 rounded-md object-contain flex-shrink-0"
                        onError={(e) => {
                          (e.target as HTMLImageElement).src = '/bookies/default.svg';
                        }}
                      />
                      <div className="min-w-0">
                        <div className={`text-xs font-medium truncate ${darkMode ? 'text-white' : 'text-gray-900'}`}>{line.bookmaker}</div>
                        {line.source && (
                          <div className={`text-[10px] ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{line.source}</div>
                        )}
                      </div>
                    </div>
                    <div className={`grid grid-cols-3 gap-3 text-xs font-semibold ${darkMode ? 'text-cyan-200' : 'text-cyan-700'}`}>
                      <span className="text-center w-10">{line.home}</span>
                      <span className="text-center w-10">{line.draw}</span>
                      <span className="text-center w-10">{line.away}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {moneylineRows.length === 0 && (
            <div className={`mt-3 text-xs ${
              darkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
              No live 1X2 lines returned yet. Preferred books are SkyBet, Paddy Power, and Betway; fallback books are shown automatically when available.
            </div>
          )}
        </div>
      )}

      {/* Market Insight Banner - Dynamic */}
      {showDynamicInsight && (
        <div className={`px-4 py-3 ${
          darkMode ? 'bg-amber-500/10 border-t border-amber-500/20' : 'bg-amber-50 border-t border-amber-200'
        }`}>
          <div className="flex items-start gap-2">
            <TrendingDown className="text-amber-500 flex-shrink-0 mt-0.5" size={16} />
            <p className={`text-xs ${darkMode ? 'text-amber-200' : 'text-amber-800'}`}>
              {dynamicMarketInsight}
            </p>
          </div>
        </div>
      )}

      {/* Market Insight Banner - Negative (fallback) */}
      {!showDynamicInsight && showNegativeInsight && (
        <div className={`px-4 py-3 ${
          darkMode ? 'bg-amber-500/10 border-t border-amber-500/20' : 'bg-amber-50 border-t border-amber-200'
        }`}>
          <div className="flex items-start gap-2">
            <TrendingDown className="text-amber-500 flex-shrink-0 mt-0.5" size={16} />
            <p className={`text-xs ${darkMode ? 'text-amber-200' : 'text-amber-800'}`}>
              <span className="font-semibold">Market Insight:</span>{' '}
              Key players {keyPlayersAtRisk.join(', ')} at elevated injury risk — could affect{' '}
              {team.team}&apos;s match odds.
            </p>
          </div>
        </div>
      )}

      {/* Market Insight Banner - Positive (fallback) */}
      {!showDynamicInsight && showPositiveInsight && (
        <div className={`px-4 py-3 ${
          darkMode ? 'bg-[#86efac]/10 border-t border-[#86efac]/20' : 'bg-emerald-50 border-t border-emerald-200'
        }`}>
          <div className="flex items-start gap-2">
            <TrendingUp className={darkMode ? 'text-[#86efac]' : 'text-emerald-600'} size={16} />
            <p className={`text-xs ${darkMode ? 'text-[#86efac]' : 'text-emerald-700'}`}>
              <span className="font-semibold">Market Insight:</span>{' '}
              Key players {keyPlayersHealthy.join(', ')} all low risk.{' '}
              Strong availability may favor {team.team} in upcoming fixtures.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function StatBox({
  icon,
  value,
  label,
  highlight,
  darkMode = true,
}: {
  icon: React.ReactNode;
  value: number;
  label: string;
  highlight?: 'red' | 'amber' | 'green';
  darkMode?: boolean;
}) {
  const highlightClasses = {
    red: 'text-red-500',
    amber: 'text-amber-500',
    green: darkMode ? 'text-[#86efac]' : 'text-emerald-600',
  };

  return (
    <div className="px-3 sm:px-4 py-3 sm:py-4 text-center">
      <div className="flex justify-center mb-1">{icon}</div>
      <div className={`text-lg sm:text-xl font-bold ${
        highlight ? highlightClasses[highlight] : (darkMode ? 'text-white' : 'text-gray-900')
      }`}>
        {value}
      </div>
      <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{label}</div>
    </div>
  );
}
