'use client';

import { FPLInsights as FPLInsightsType, LeagueStanding } from '@/types/api';
import { Trophy, Calendar, TrendingUp, AlertTriangle, Zap } from 'lucide-react';

interface FPLInsightsProps {
  insights: FPLInsightsType;
  selectedTeam?: string;
  darkMode?: boolean;
}

export function FPLInsights({ insights, selectedTeam, darkMode = true }: FPLInsightsProps) {
  // Find if selected team has a double gameweek
  const upcomingDGW = insights.upcoming_gameweeks.find(gw =>
    gw.double_gameweek_teams.some(t => t.toLowerCase() === selectedTeam?.toLowerCase())
  );

  // Get selected team's standing
  const teamStanding = insights.standings.find(
    s => s.name.toLowerCase() === selectedTeam?.toLowerCase()
  );

  // Get opponent info from upcoming fixtures
  const currentGW = insights.upcoming_gameweeks.find(gw => gw.is_current || gw.is_next);

  return (
    <div className={`rounded-2xl overflow-hidden ${
      darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-lg'
    }`}>
      {/* Header */}
      <div className={`px-6 py-4 flex items-center gap-3 ${
        darkMode
          ? 'bg-gradient-to-r from-purple-500/20 to-[#141414] border-b border-[#1f1f1f]'
          : 'bg-gradient-to-r from-purple-100 to-white border-b border-gray-100'
      }`}>
        <Trophy className={darkMode ? 'text-purple-400' : 'text-purple-600'} size={22} />
        <div>
          <h3 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            FPL Context
          </h3>
          <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
            Gameweek {insights.current_gameweek || '?'}
          </p>
        </div>
      </div>

      {/* Double Gameweek Alert */}
      {upcomingDGW && (
        <div className={`px-4 py-3 flex items-start gap-2 ${
          darkMode ? 'bg-amber-500/10 border-b border-[#1f1f1f]' : 'bg-amber-50 border-b border-gray-100'
        }`}>
          <Zap className="text-amber-500 flex-shrink-0 mt-0.5" size={16} />
          <div>
            <p className={`text-xs font-semibold ${darkMode ? 'text-amber-300' : 'text-amber-700'}`}>
              Double Gameweek {upcomingDGW.gameweek}
            </p>
            <p className={`text-xs ${darkMode ? 'text-amber-200/80' : 'text-amber-600'}`}>
              {selectedTeam} players could score double points. Consider healthy starters for your FPL team.
            </p>
          </div>
        </div>
      )}

      {/* Team Standing */}
      {teamStanding && (
        <div className={`px-4 py-3 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                teamStanding.position <= 4
                  ? 'bg-blue-500/20 text-blue-400'
                  : teamStanding.position <= 6
                  ? 'bg-green-500/20 text-green-400'
                  : darkMode ? 'bg-[#1f1f1f] text-gray-400' : 'bg-gray-100 text-gray-600'
              }`}>
                {teamStanding.position}
              </div>
              <div>
                <p className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {teamStanding.name}
                </p>
                <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                  {teamStanding.played} played, {teamStanding.points} pts
                </p>
              </div>
            </div>
            {teamStanding.form && (
              <div className="flex gap-0.5">
                {teamStanding.form.split('').slice(0, 5).map((result, i) => (
                  <span
                    key={i}
                    className={`w-5 h-5 rounded text-xs font-bold flex items-center justify-center ${
                      result === 'W'
                        ? 'bg-green-500/20 text-green-400'
                        : result === 'D'
                        ? 'bg-gray-500/20 text-gray-400'
                        : 'bg-red-500/20 text-red-400'
                    }`}
                  >
                    {result}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Upcoming Featured Matches */}
      {currentGW && currentGW.featured_matches.length > 0 && (
        <div className={`px-4 py-3 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <p className={`text-xs font-medium mb-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            <Calendar size={12} className="inline mr-1" />
            Top 6 Clashes This Week
          </p>
          <div className="flex flex-wrap gap-2">
            {currentGW.featured_matches.map((match, i) => (
              <span
                key={i}
                className={`text-xs px-2 py-1 rounded ${
                  darkMode ? 'bg-[#1f1f1f] text-gray-300' : 'bg-gray-100 text-gray-700'
                }`}
              >
                {match}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Quick Stats */}
      <div className="px-4 py-3">
        <div className="grid grid-cols-2 gap-3">
          <QuickStat
            label="League Leaders"
            value={insights.standings[0]?.short_name || '-'}
            subvalue={`${insights.standings[0]?.points || 0} pts`}
            darkMode={darkMode}
          />
          <QuickStat
            label="Title Race"
            value={`${insights.standings[0]?.points - (insights.standings[1]?.points || 0)} pts`}
            subvalue="gap to 2nd"
            darkMode={darkMode}
          />
        </div>
      </div>

      {/* FPL Tip */}
      {insights.has_double_gameweek && (
        <div className={`px-4 py-3 ${
          darkMode ? 'bg-[#0a0a0a]' : 'bg-gray-50'
        }`}>
          <div className="flex items-start gap-2">
            <TrendingUp className={darkMode ? 'text-[#86efac]' : 'text-emerald-600'} size={14} />
            <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              <span className={`font-semibold ${darkMode ? 'text-[#86efac]' : 'text-emerald-600'}`}>FPL Tip:</span>{' '}
              Double gameweeks coming up. Low-risk starters from {insights.upcoming_gameweeks
                .find(gw => gw.double_gameweek_teams.length > 0)
                ?.double_gameweek_teams.slice(0, 2).join(', ') || 'affected teams'
              } could be valuable picks.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function QuickStat({
  label,
  value,
  subvalue,
  darkMode = true,
}: {
  label: string;
  value: string;
  subvalue: string;
  darkMode?: boolean;
}) {
  return (
    <div className={`rounded-lg p-2 ${darkMode ? 'bg-[#0a0a0a]' : 'bg-gray-50'}`}>
      <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{label}</p>
      <p className={`text-sm font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{value}</p>
      <p className={`text-xs ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>{subvalue}</p>
    </div>
  );
}
