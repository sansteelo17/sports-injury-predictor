'use client';

import { ChevronDown } from 'lucide-react';

interface TeamSelectorProps {
  teams: string[];
  selectedTeam: string;
  onSelectTeam: (team: string) => void;
}

export function TeamSelector({ teams, selectedTeam, onSelectTeam }: TeamSelectorProps) {
  return (
    <div className="relative">
      <select
        value={selectedTeam}
        onChange={(e) => onSelectTeam(e.target.value)}
        className="w-full appearance-none bg-white border-2 border-gray-200 rounded-xl px-4 py-3 pr-10 text-gray-900 font-medium focus:outline-none focus:border-purple-500 transition-colors cursor-pointer"
      >
        <option value="">Select a team...</option>
        {teams.map((team) => (
          <option key={team} value={team}>
            {team}
          </option>
        ))}
      </select>
      <ChevronDown
        className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
        size={20}
      />
    </div>
  );
}
