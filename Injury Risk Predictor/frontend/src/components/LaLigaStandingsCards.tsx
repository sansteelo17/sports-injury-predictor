'use client';

import { useState } from 'react';
import { LaLigaStandingRow } from '@/types/api';
import { Trophy, Medal, MapPin, AlertTriangle } from 'lucide-react';

interface LaLigaStandingsCardsProps {
  standings: LaLigaStandingRow[];
  selectedTeam?: string;
  darkMode?: boolean;
}

function TeamBadge({ url, name }: { url: string | null; name: string }) {
  const [errored, setErrored] = useState(false);
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

function getOrdinalSuffix(n: number): string {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return s[(v - 20) % 10] || s[v] || s[0];
}

// La Liga is 38 matchdays. Once the leader has played them all, surface the
// champion instead of the live race.
const LA_LIGA_MATCHDAYS = 38;

export function LaLigaStandingsCards({ standings, selectedTeam, darkMode = true }: LaLigaStandingsCardsProps) {
  if (!standings.length) return null;

  const leader = standings[0];
  const second = standings[1];
  const selected = selectedTeam
    ? standings.find((r) => r.name.toLowerCase() === selectedTeam.toLowerCase())
    : null;

  const seasonOver = (leader.played || 0) >= LA_LIGA_MATCHDAYS;
  if (seasonOver) {
    const gap = second ? leader.points - second.points : 0;
    return (
      <div className={`rounded-xl p-4 sm:p-5 border ${
        darkMode
          ? 'bg-gradient-to-br from-amber-500/10 to-amber-500/0 border-amber-500/30'
          : 'bg-gradient-to-br from-amber-50 to-white border-amber-200'
      }`}>
        <div className="flex items-center gap-2 mb-2">
          <Trophy className="text-amber-500 flex-shrink-0" size={18} />
          <span className={`text-xs font-medium uppercase tracking-wider ${darkMode ? 'text-amber-300' : 'text-amber-700'}`}>
            La Liga — Season Concluded
          </span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <TeamBadge url={leader.badge_url} name={leader.name} />
          <span className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            {leader.name}
          </span>
          <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
            crowned champions
          </span>
        </div>
        <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          {leader.points} pts from {leader.played} matches
          {gap > 0 && second && ` · ${gap} clear of ${second.name}`}
        </div>
        {selected && selected.name.toLowerCase() !== leader.name.toLowerCase() && (
          <div className={`mt-3 pt-3 border-t flex items-center gap-2 text-xs ${
            darkMode ? 'border-amber-500/20 text-gray-400' : 'border-amber-200 text-gray-600'
          }`}>
            <MapPin size={12} className={darkMode ? 'text-[#86efac]' : 'text-emerald-600'} />
            {selected.name} finished{' '}
            <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              {selected.position}{getOrdinalSuffix(selected.position)}
            </span>{' '}
            on {selected.points} pts
          </div>
        )}
      </div>
    );
  }

  const cardBase = darkMode
    ? 'bg-[#141414] border border-[#1f1f1f]'
    : 'bg-white shadow-sm border border-gray-100';
  const labelClass = darkMode ? 'text-gray-400' : 'text-gray-500';
  const valueClass = darkMode ? 'text-white' : 'text-gray-900';
  const subClass = darkMode ? 'text-gray-500' : 'text-gray-400';

  const distanceFromTop = selected ? leader.points - selected.points : null;
  const distanceFromSafety = selected
    ? selected.points - (standings[17]?.points ?? 0)
    : null;

  return (
    <div className="grid grid-cols-3 gap-2">
      {/* Leader */}
      <div className={`rounded-xl p-2.5 sm:p-3 ${cardBase}`}>
        <div className="flex items-center gap-1.5 mb-2">
          <Trophy className="text-amber-500 flex-shrink-0" size={14} />
          <span className={`text-xs font-medium ${labelClass}`}>Leaders</span>
        </div>
        <div className="flex items-center gap-1.5">
          <TeamBadge url={leader.badge_url} name={leader.name} />
          <span className={`text-sm font-bold truncate ${valueClass}`}>{leader.name}</span>
        </div>
        <div className={`text-xs mt-0.5 ${subClass}`}>
          {leader.points} pts ({leader.played} played)
        </div>
      </div>

      {/* Selected team position or 2nd place */}
      {selected ? (
        <div className={`rounded-xl p-2.5 sm:p-3 ${cardBase}`}>
          <div className="flex items-center gap-1.5 mb-2">
            <MapPin
              className={`flex-shrink-0 ${darkMode ? 'text-[#86efac]' : 'text-emerald-600'}`}
              size={14}
            />
            <span className={`text-xs font-medium ${labelClass}`}>Position</span>
          </div>
          <div className={`text-sm font-bold ${valueClass}`}>
            {selected.position}{getOrdinalSuffix(selected.position)}
          </div>
          <div className={`text-xs mt-0.5 ${subClass}`}>
            {selected.points} pts
            {distanceFromTop !== null && distanceFromTop > 0 && (
              <span className="text-amber-500"> (-{distanceFromTop})</span>
            )}
          </div>
        </div>
      ) : (
        <div className={`rounded-xl p-2.5 sm:p-3 ${cardBase}`}>
          <div className="flex items-center gap-1.5 mb-2">
            <Medal className="text-gray-400 flex-shrink-0" size={14} />
            <span className={`text-xs font-medium ${labelClass}`}>2nd Place</span>
          </div>
          <div className="flex items-center gap-1.5">
            <TeamBadge url={second?.badge_url ?? null} name={second?.name ?? ''} />
            <span className={`text-sm font-bold truncate ${valueClass}`}>{second?.name ?? ''}</span>
          </div>
          <div className={`text-xs mt-0.5 ${subClass}`}>{second?.points ?? 0} pts</div>
        </div>
      )}

      {/* Context: relegation / title race / gap */}
      <div className={`rounded-xl p-2.5 sm:p-3 ${cardBase}`}>
        {selected && selected.position >= 18 ? (
          <>
            <div className="flex items-center gap-1.5 mb-2">
              <AlertTriangle className="text-red-500 flex-shrink-0" size={14} />
              <span className={`text-xs font-medium ${labelClass}`}>Relegation</span>
            </div>
            <div className={`text-sm font-bold ${(distanceFromSafety ?? 0) <= 3 ? 'text-red-500' : 'text-amber-500'}`}>
              {distanceFromSafety === 0 ? 'On the line!' : `${distanceFromSafety} pts`}
            </div>
            <div className={`text-xs mt-0.5 ${subClass}`}>above relegation</div>
          </>
        ) : selected && selected.position === 1 ? (
          <>
            <div className="flex items-center gap-1.5 mb-2">
              <Trophy className="text-amber-500 flex-shrink-0" size={14} />
              <span className={`text-xs font-medium ${labelClass}`}>Title Race</span>
            </div>
            <div className={`text-sm font-bold ${(second && leader.points - second.points) <= 3 ? 'text-amber-500' : valueClass}`}>
              {second ? `${leader.points - second.points} pts` : '--'}
            </div>
            <div className={`text-xs mt-0.5 ${subClass}`}>ahead of {second?.name ?? '2nd'}</div>
          </>
        ) : selected && selected.position <= 4 ? (
          <>
            <div className="flex items-center gap-1.5 mb-2">
              <Medal className="text-gray-400 flex-shrink-0" size={14} />
              <span className={`text-xs font-medium ${labelClass}`}>Title Race</span>
            </div>
            <div
              className={`text-sm font-bold ${
                (distanceFromTop ?? 0) <= 3
                  ? 'text-amber-500'
                  : valueClass
              }`}
            >
              {distanceFromTop === 0 ? 'Level!' : `${distanceFromTop} pts`}
            </div>
            <div className={`text-xs mt-0.5 ${subClass}`}>behind {leader.name}</div>
          </>
        ) : (
          <>
            <div className="flex items-center gap-1.5 mb-2">
              <Trophy className="text-amber-500 flex-shrink-0" size={14} />
              <span className={`text-xs font-medium ${labelClass}`}>From Leaders</span>
            </div>
            <div className={`text-sm font-bold ${valueClass}`}>{distanceFromTop ?? leader.points - (second?.points ?? 0)} pts</div>
            <div className={`text-xs mt-0.5 ${subClass}`}>behind {leader.name}</div>
          </>
        )}
      </div>
    </div>
  );
}
