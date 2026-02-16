'use client';

import { useState } from 'react';
import { PlayerSummary } from '@/types/api';
import { RiskBadge } from './RiskBadge';
import { User } from 'lucide-react';

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
      src={url}
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
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2.5 min-w-0">
              <PlayerImage url={player.player_image_url} name={player.name} darkMode={darkMode} />
              <div className="min-w-0">
                <div className={`font-medium text-sm truncate ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {player.name}
                </div>
                <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                  {player.position} {player.minutes_played > 0 ? `\u00B7 ${player.minutes_played}'` : ''}
                </div>
              </div>
            </div>
            <div className="text-right flex-shrink-0">
              <RiskBadge level={player.risk_level} size="sm" darkMode={darkMode} />
              <div className={`text-xs mt-0.5 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                {Math.round(player.risk_probability * 100)}%
              </div>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}
