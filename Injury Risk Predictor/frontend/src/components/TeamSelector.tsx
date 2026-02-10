'use client';

import { ChevronDown } from 'lucide-react';

interface TeamSelectorProps {
  teams: string[];
  selectedTeam: string;
  onSelectTeam: (team: string) => void;
  darkMode?: boolean;
}

export function TeamSelector({ teams, selectedTeam, onSelectTeam, darkMode = true }: TeamSelectorProps) {
  return (
    <div className="relative">
      <select
        value={selectedTeam}
        onChange={(e) => onSelectTeam(e.target.value)}
        className={`w-full appearance-none border-2 rounded-xl px-4 py-3 pr-10 font-medium focus:outline-none transition-colors cursor-pointer ${
          darkMode
            ? 'bg-[#141414] border-[#1f1f1f] text-white focus:border-[#86efac]'
            : 'bg-white border-gray-200 text-gray-900 focus:border-green-500'
        }`}
      >
        <option value="">Select a team...</option>
        {teams.map((team) => (
          <option key={team} value={team}>
            {team}
          </option>
        ))}
      </select>
      <ChevronDown
        className={`absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}
        size={20}
      />
    </div>
  );
}
