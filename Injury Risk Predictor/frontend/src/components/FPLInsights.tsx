'use client';

import { FPLInsights as FPLInsightsType } from '@/types/api';
import { Calendar, Zap, Swords } from 'lucide-react';

interface FPLInsightsProps {
  insights: FPLInsightsType;
  selectedTeam?: string;
  darkMode?: boolean;
}

export function FPLInsights({ insights, selectedTeam, darkMode = true }: FPLInsightsProps) {
  // Find if selected team has a double gameweek in upcoming fixtures
  const teamDGW = insights.upcoming_gameweeks.find(gw =>
    gw.double_gameweek_teams.some(t => t.toLowerCase() === selectedTeam?.toLowerCase())
  );

  // Get current/next gameweek info
  const currentGW = insights.upcoming_gameweeks.find(gw => gw.is_current || gw.is_next);

  // Check if selected team is playing in a featured match (top 6 clash)
  const teamFeaturedMatch = currentGW?.featured_matches.find(match =>
    match.toLowerCase().includes(selectedTeam?.toLowerCase().replace('man ', 'man').slice(0, 3) || '')
  );

  // If no relevant insights for this team, don't show the panel
  if (!teamDGW && !teamFeaturedMatch && !currentGW) {
    return null;
  }

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
        <Calendar className={darkMode ? 'text-purple-400' : 'text-purple-600'} size={20} />
        <div>
          <h3 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            FPL Context
          </h3>
          <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
            {currentGW?.name || `Gameweek ${insights.current_gameweek || '?'}`}
          </p>
        </div>
      </div>

      {/* Double Gameweek Alert - Only show if THIS team has one */}
      {teamDGW && (
        <div className={`px-4 py-3 flex items-start gap-2 ${
          darkMode ? 'bg-amber-500/10 border-b border-[#1f1f1f]' : 'bg-amber-50 border-b border-gray-100'
        }`}>
          <Zap className="text-amber-500 flex-shrink-0 mt-0.5" size={16} />
          <div>
            <p className={`text-xs font-semibold ${darkMode ? 'text-amber-300' : 'text-amber-700'}`}>
              Double Gameweek {teamDGW.gameweek}
            </p>
            <p className={`text-xs ${darkMode ? 'text-amber-200/80' : 'text-amber-600'}`}>
              {selectedTeam} play twice this week. FPL managers may target their healthy players, but watch for fatigue risk.
            </p>
          </div>
        </div>
      )}

      {/* Featured Match - Only show if this team is in a top 6 clash */}
      {teamFeaturedMatch && (
        <div className={`px-4 py-3 flex items-start gap-2 ${
          darkMode ? 'bg-blue-500/10 border-b border-[#1f1f1f]' : 'bg-blue-50 border-b border-gray-100'
        }`}>
          <Swords className={darkMode ? 'text-blue-400' : 'text-blue-600'} size={16} />
          <div>
            <p className={`text-xs font-semibold ${darkMode ? 'text-blue-300' : 'text-blue-700'}`}>
              Top 6 Clash
            </p>
            <p className={`text-xs ${darkMode ? 'text-blue-200/80' : 'text-blue-600'}`}>
              {teamFeaturedMatch} — high-stakes fixture this week.
            </p>
          </div>
        </div>
      )}

      {/* Upcoming Fixtures Summary */}
      {currentGW && (
        <div className={`px-4 py-3 ${darkMode ? '' : ''}`}>
          <p className={`text-xs mb-2 ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
            {currentGW.fixture_count} fixtures this gameweek
            {currentGW.deadline && (
              <span> · Deadline: {new Date(currentGW.deadline).toLocaleDateString('en-GB', {
                weekday: 'short',
                day: 'numeric',
                month: 'short',
                hour: '2-digit',
                minute: '2-digit'
              })}</span>
            )}
          </p>

          {/* Other teams with double gameweeks (if any, and not the selected team) */}
          {insights.upcoming_gameweeks.some(gw =>
            gw.double_gameweek_teams.length > 0 &&
            !gw.double_gameweek_teams.some(t => t.toLowerCase() === selectedTeam?.toLowerCase())
          ) && (
            <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              <span className="text-amber-500">DGW:</span>{' '}
              {insights.upcoming_gameweeks
                .flatMap(gw => gw.double_gameweek_teams)
                .filter(t => t.toLowerCase() !== selectedTeam?.toLowerCase())
                .slice(0, 3)
                .join(', ')
              }
            </p>
          )}
        </div>
      )}
    </div>
  );
}
