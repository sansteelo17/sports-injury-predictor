'use client';

import { useState } from 'react';
import { StandingsSummary } from '@/types/api';
import { Trophy, Medal, MapPin, AlertTriangle } from 'lucide-react';

interface StandingsCardsProps {
  standings: StandingsSummary;
  darkMode?: boolean;
  teamBadges?: Record<string, string>;
}

function SmallBadge({ name, badges }: { name: string; badges?: Record<string, string> }) {
  const [errored, setErrored] = useState(false);
  const url = badges?.[name];

  if (!url || errored) return null;

  return (
    <img
      src={url}
      alt=""
      width={16}
      height={16}
      className="flex-shrink-0 object-contain"
      onError={() => setErrored(true)}
    />
  );
}

export function StandingsCards({ standings, darkMode = true, teamBadges }: StandingsCardsProps) {
  return (
    <div className="grid grid-cols-3 gap-2 sm:gap-3">
      {/* League Leaders */}
      <div className={`rounded-xl p-2.5 sm:p-3 ${
        darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-sm border border-gray-100'
      }`}>
        <div className="flex items-center gap-1.5 mb-2">
          <Trophy className="text-amber-500 flex-shrink-0" size={14} />
          <span className={`text-xs font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Leaders
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <SmallBadge name={standings.leader.name} badges={teamBadges} />
          <span className={`text-sm font-bold truncate ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            {standings.leader.short_name}
          </span>
        </div>
        <div className={`text-xs mt-0.5 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          {standings.leader.points} pts ({standings.leader.played} played)
        </div>
      </div>

      {/* Selected Team Position */}
      {standings.selected_team ? (
        <div className={`rounded-xl p-2.5 sm:p-3 ${
          darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-sm border border-gray-100'
        }`}>
          <div className="flex items-center gap-1.5 mb-2">
            <MapPin className={`flex-shrink-0 ${darkMode ? 'text-[#86efac]' : 'text-emerald-600'}`} size={14} />
            <span className={`text-xs font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Position
            </span>
          </div>
          <div className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            {standings.selected_team.position}{getOrdinalSuffix(standings.selected_team.position || 0)}
          </div>
          <div className={`text-xs mt-0.5 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            {standings.selected_team.points} pts
            {standings.selected_team.distance_from_top !== undefined && standings.selected_team.distance_from_top > 0 && (
              <span className="text-amber-500"> (-{standings.selected_team.distance_from_top})</span>
            )}
          </div>
        </div>
      ) : (
        <div className={`rounded-xl p-2.5 sm:p-3 ${
          darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-sm border border-gray-100'
        }`}>
          <div className="flex items-center gap-1.5 mb-2">
            <Medal className="text-gray-400 flex-shrink-0" size={14} />
            <span className={`text-xs font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              2nd Place
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <SmallBadge name={standings.second.name} badges={teamBadges} />
            <span className={`text-sm font-bold truncate ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              {standings.second.short_name}
            </span>
          </div>
          <div className={`text-xs mt-0.5 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            {standings.second.points} pts
          </div>
        </div>
      )}

      {/* Context card */}
      <div className={`rounded-xl p-2.5 sm:p-3 ${
        darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-sm border border-gray-100'
      }`}>
        {(() => {
          const position = standings.selected_team?.position || 0;

          if (position >= 18) {
            const distanceFromSafety = standings.selected_team?.distance_from_safety || 0;
            return (
              <>
                <div className="flex items-center gap-1.5 mb-2">
                  <AlertTriangle className="text-red-500 flex-shrink-0" size={14} />
                  <span className={`text-xs font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Relegation
                  </span>
                </div>
                <div className={`text-sm font-bold ${distanceFromSafety > 3 ? 'text-red-500' : 'text-amber-500'}`}>
                  {distanceFromSafety === 0 ? 'On the line!' : `${distanceFromSafety} pts`}
                </div>
                <div className={`text-xs mt-0.5 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  from safety
                </div>
              </>
            );
          }

          if (position > 4) {
            return (
              <>
                <div className="flex items-center gap-1.5 mb-2">
                  <Trophy className="text-amber-500 flex-shrink-0" size={14} />
                  <span className={`text-xs font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    From Leaders
                  </span>
                </div>
                <div className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {standings.selected_team?.distance_from_top || 0} pts
                </div>
                <div className={`text-xs mt-0.5 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  behind {standings.leader.short_name}
                </div>
              </>
            );
          }

          return (
            <>
              <div className="flex items-center gap-1.5 mb-2">
                <Medal className="text-gray-400 flex-shrink-0" size={14} />
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
              <div className={`text-xs mt-0.5 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                gap to 2nd
              </div>
            </>
          );
        })()}
      </div>
    </div>
  );
}

function getOrdinalSuffix(n: number): string {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return s[(v - 20) % 10] || s[v] || s[0];
}
