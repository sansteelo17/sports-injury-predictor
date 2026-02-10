'use client';

import { StandingsSummary } from '@/types/api';
import { Trophy, Medal, MapPin } from 'lucide-react';

interface StandingsCardsProps {
  standings: StandingsSummary;
  darkMode?: boolean;
}

export function StandingsCards({ standings, darkMode = true }: StandingsCardsProps) {
  return (
    <div className="grid grid-cols-3 gap-3">
      {/* League Leaders */}
      <div className={`rounded-xl p-3 ${
        darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-sm border border-gray-100'
      }`}>
        <div className="flex items-center gap-2 mb-2">
          <Trophy className="text-amber-500" size={16} />
          <span className={`text-xs font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Leaders
          </span>
        </div>
        <div className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
          {standings.leader.short_name}
        </div>
        <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          {standings.leader.points} pts ({standings.leader.played} played)
        </div>
      </div>

      {/* Selected Team Position */}
      {standings.selected_team ? (
        <div className={`rounded-xl p-3 ${
          darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-sm border border-gray-100'
        }`}>
          <div className="flex items-center gap-2 mb-2">
            <MapPin className={darkMode ? 'text-[#86efac]' : 'text-emerald-600'} size={16} />
            <span className={`text-xs font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Position
            </span>
          </div>
          <div className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            {standings.selected_team.position}{getOrdinalSuffix(standings.selected_team.position || 0)}
          </div>
          <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            {standings.selected_team.points} pts
            {standings.selected_team.distance_from_top !== undefined && standings.selected_team.distance_from_top > 0 && (
              <span className="text-amber-500"> (-{standings.selected_team.distance_from_top})</span>
            )}
          </div>
        </div>
      ) : (
        <div className={`rounded-xl p-3 ${
          darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-sm border border-gray-100'
        }`}>
          <div className="flex items-center gap-2 mb-2">
            <Medal className="text-gray-400" size={16} />
            <span className={`text-xs font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              2nd Place
            </span>
          </div>
          <div className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            {standings.second.short_name}
          </div>
          <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            {standings.second.points} pts
          </div>
        </div>
      )}

      {/* Title Race Gap */}
      <div className={`rounded-xl p-3 ${
        darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-sm border border-gray-100'
      }`}>
        <div className="flex items-center gap-2 mb-2">
          <Medal className="text-gray-400" size={16} />
          <span className={`text-xs font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Title Race
          </span>
        </div>
        <div className={`text-sm font-bold ${
          standings.gap_to_second <= 3
            ? 'text-red-500'
            : standings.gap_to_second <= 6
            ? 'text-amber-500'
            : darkMode ? 'text-white' : 'text-gray-900'
        }`}>
          {standings.gap_to_second === 0 ? 'Level!' : `${standings.gap_to_second} pts`}
        </div>
        <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          gap to 2nd
        </div>
      </div>
    </div>
  );
}

function getOrdinalSuffix(n: number): string {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return s[(v - 20) % 10] || s[v] || s[0];
}
