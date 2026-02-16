'use client';

import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Search } from 'lucide-react';

interface TeamSelectorProps {
  teams: string[];
  selectedTeam: string;
  onSelectTeam: (team: string) => void;
  darkMode?: boolean;
  teamBadges?: Record<string, string>;
}

function TeamBadge({ team, badges, size = 20 }: { team: string; badges?: Record<string, string>; size?: number }) {
  const [errored, setErrored] = useState(false);
  const url = badges?.[team];

  if (!url || errored) return null;

  return (
    <img
      src={url}
      alt=""
      width={size}
      height={size}
      className="flex-shrink-0 object-contain"
      onError={() => setErrored(true)}
    />
  );
}

export function TeamSelector({ teams, selectedTeam, onSelectTeam, darkMode = true, teamBadges }: TeamSelectorProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setSearch('');
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus();
    }
  }, [open]);

  const filtered = teams.filter(t => t.toLowerCase().includes(search.toLowerCase()));

  return (
    <div ref={ref} className="relative">
      {/* Selected value button */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className={`w-full flex items-center gap-3 border-2 rounded-xl px-4 py-3 font-medium focus:outline-none transition-colors cursor-pointer text-left ${
          darkMode
            ? 'bg-[#141414] border-[#1f1f1f] text-white focus:border-[#86efac]'
            : 'bg-white border-gray-200 text-gray-900 focus:border-emerald-500'
        }`}
      >
        {selectedTeam ? (
          <>
            <TeamBadge team={selectedTeam} badges={teamBadges} size={22} />
            <span className="flex-1">{selectedTeam}</span>
          </>
        ) : (
          <span className={`flex-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            Select a team...
          </span>
        )}
        <ChevronDown
          className={`flex-shrink-0 transition-transform ${open ? 'rotate-180' : ''} ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}
          size={20}
        />
      </button>

      {/* Dropdown */}
      {open && (
        <div className={`absolute z-50 w-full mt-2 rounded-xl border-2 shadow-xl overflow-hidden ${
          darkMode ? 'bg-[#141414] border-[#1f1f1f]' : 'bg-white border-gray-200'
        }`}>
          {/* Search input */}
          <div className={`flex items-center gap-2 px-3 py-2 border-b ${
            darkMode ? 'border-[#1f1f1f]' : 'border-gray-100'
          }`}>
            <Search size={14} className={darkMode ? 'text-gray-500' : 'text-gray-400'} />
            <input
              ref={inputRef}
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search teams..."
              className={`w-full text-sm bg-transparent outline-none ${
                darkMode ? 'text-white placeholder:text-gray-600' : 'text-gray-900 placeholder:text-gray-400'
              }`}
            />
          </div>

          {/* Options */}
          <div className="max-h-64 overflow-y-auto">
            {filtered.length === 0 && (
              <div className={`px-4 py-3 text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                No teams found
              </div>
            )}
            {filtered.map((team) => (
              <button
                key={team}
                type="button"
                onClick={() => {
                  onSelectTeam(team);
                  setOpen(false);
                  setSearch('');
                }}
                className={`w-full flex items-center gap-3 px-4 py-2.5 text-sm text-left transition-colors ${
                  team === selectedTeam
                    ? darkMode
                      ? 'bg-[#86efac]/10 text-[#86efac]'
                      : 'bg-emerald-50 text-emerald-700'
                    : darkMode
                      ? 'text-white hover:bg-[#1f1f1f]'
                      : 'text-gray-900 hover:bg-gray-50'
                }`}
              >
                <TeamBadge team={team} badges={teamBadges} size={20} />
                <span>{team}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
