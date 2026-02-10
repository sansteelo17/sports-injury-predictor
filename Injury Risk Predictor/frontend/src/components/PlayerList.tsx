'use client';

import { PlayerSummary } from '@/types/api';
import { RiskBadge } from './RiskBadge';

interface PlayerListProps {
  players: PlayerSummary[];
  onSelectPlayer: (name: string) => void;
  selectedPlayer?: string;
}

export function PlayerList({ players, onSelectPlayer, selectedPlayer }: PlayerListProps) {
  return (
    <div className="space-y-2">
      {players.map((player) => (
        <button
          key={player.name}
          onClick={() => onSelectPlayer(player.name)}
          className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
            selectedPlayer === player.name
              ? 'bg-purple-100 border-2 border-purple-500'
              : 'bg-white hover:bg-gray-50 border-2 border-transparent'
          } shadow-sm`}
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium text-gray-900">{player.name}</div>
              <div className="text-sm text-gray-500">{player.position}</div>
            </div>
            <div className="text-right">
              <RiskBadge level={player.risk_level} size="sm" />
              <div className="text-xs text-gray-400 mt-1">
                {Math.round(player.risk_probability * 100)}%
              </div>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}
