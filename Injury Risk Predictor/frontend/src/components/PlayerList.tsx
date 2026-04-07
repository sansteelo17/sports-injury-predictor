'use client';

import { useState } from 'react';
import { PlayerSummary } from '@/types/api';
import { RiskBadge } from './RiskBadge';
import { User, Cross } from 'lucide-react';
import { toAbsoluteApiUrl } from '@/lib/api';

interface PlayerListProps {
  players: PlayerSummary[];
  onSelectPlayer: (name: string) => void;
  selectedPlayer?: string;
  darkMode?: boolean;
}

function PlayerImage({ url, name, darkMode }: { url: string | null; name: string; darkMode: boolean }) {
  const [errored, setErrored] = useState(false);

  if (!url || errored) {
    return (
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
        darkMode ? 'bg-[#1f1f1f]' : 'bg-gray-100'
      }`}>
        <User size={14} className={darkMode ? 'text-gray-600' : 'text-gray-400'} />
      </div>
    );
  }

  return (
    <img
      src={toAbsoluteApiUrl(url)}
      alt={name}
      className="w-8 h-8 rounded-full object-cover flex-shrink-0"
      onError={() => setErrored(true)}
    />
  );
}

export function PlayerList({ players, onSelectPlayer, selectedPlayer, darkMode = true }: PlayerListProps) {
  return (
    <div className="space-y-2">
      {players.map((player) => (
        <button
          key={player.name}
          onClick={() => onSelectPlayer(player.name)}
          className={`w-full text-left px-3 py-2.5 rounded-lg transition-all ${
            selectedPlayer === player.name
              ? darkMode
                ? 'bg-[#86efac]/10 border-2 border-[#86efac]/50'
                : 'bg-emerald-50 border-2 border-emerald-500'
              : darkMode
                ? 'bg-[#141414] hover:bg-[#1f1f1f] border-2 border-transparent hover:border-[#86efac]/30'
                : 'bg-white hover:bg-gray-50 border-2 border-transparent hover:border-emerald-300'
          }`}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-2.5 min-w-0 flex-1">
              <PlayerImage url={player.player_image_url} name={player.name} darkMode={darkMode} />
              <div className="min-w-0 flex-1">
                <div className={`font-medium text-sm leading-tight flex items-start gap-1 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  <span className="min-w-0 flex-1 break-words sm:truncate">{player.name}</span>
                  {player.is_currently_injured && (
                    <Cross size={10} className="text-red-400 flex-shrink-0 mt-0.5" />
                  )}
                </div>
                <div className={`text-xs mt-0.5 leading-tight break-words ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                  {player.position}
                  {player.shirt_number != null ? ` \u00B7 #${player.shirt_number}` : ''}
                  {player.minutes_played > 0 ? ` \u00B7 ${player.minutes_played}'` : ''}
                </div>
              </div>
            </div>
            <div className="text-right flex-shrink-0 pt-0.5">
              {player.is_currently_injured ? (
                <>
                  <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${
                    darkMode ? 'bg-red-900/40 text-red-400' : 'bg-red-100 text-red-600'
                  }`}>OUT</span>
                  <div className={`text-xs mt-0.5 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
                    Injured
                  </div>
                </>
              ) : (
                <>
                  <RiskBadge level={player.risk_level} size="sm" darkMode={darkMode} />
                  <div className={`text-xs mt-0.5 font-medium ${
                    player.risk_level === 'High' ? 'text-red-400' :
                    player.risk_level === 'Medium' ? 'text-amber-400' :
                    darkMode ? 'text-[#86efac]' : 'text-emerald-600'
                  }`}>
                    {Math.round(player.risk_probability * 100)}%
                  </div>
                </>
              )}
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}
