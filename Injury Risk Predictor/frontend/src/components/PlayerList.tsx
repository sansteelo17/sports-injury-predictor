'use client';

import { PlayerSummary } from '@/types/api';
import { RiskBadge } from './RiskBadge';

interface PlayerListProps {
  players: PlayerSummary[];
  onSelectPlayer: (name: string) => void;
  selectedPlayer?: string;
  darkMode?: boolean;
}

export function PlayerList({ players, onSelectPlayer, selectedPlayer, darkMode = true }: PlayerListProps) {
  return (
    <div className="space-y-2">
      {players.map((player) => (
        <button
          key={player.name}
          onClick={() => onSelectPlayer(player.name)}
          className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
            selectedPlayer === player.name
              ? darkMode
                ? 'bg-[#86efac]/10 border-2 border-[#86efac]/50'
                : 'bg-emerald-50 border-2 border-emerald-500'
              : darkMode
                ? 'bg-[#141414] hover:bg-[#1f1f1f] border-2 border-transparent hover:border-[#86efac]/30'
                : 'bg-white hover:bg-gray-50 border-2 border-transparent hover:border-emerald-300'
          }`}
        >
          <div className="flex items-center justify-between">
            <div>
              <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                {player.name}
              </div>
              <div className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                {player.position}
              </div>
            </div>
            <div className="text-right">
              <RiskBadge level={player.risk_level} size="sm" darkMode={darkMode} />
              <div className={`text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                {Math.round(player.risk_probability * 100)}%
              </div>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}
