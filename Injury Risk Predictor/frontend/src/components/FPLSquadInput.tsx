'use client';

import { useState, useEffect } from 'react';
import { Zap } from 'lucide-react';

const STORAGE_KEY = 'yara_fpl_team_id';

interface FPLSquadInputProps {
  onSync: (teamId: string) => void;
  loading: boolean;
  error: string | null;
  darkMode: boolean;
}

export function FPLSquadInput({ onSync, loading, error, darkMode }: FPLSquadInputProps) {
  const [teamId, setTeamId] = useState('');

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) setTeamId(saved);
  }, []);

  const handleSync = () => {
    const id = teamId.trim();
    if (!id) return;
    localStorage.setItem(STORAGE_KEY, id);
    onSync(id);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSync();
  };

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <input
          type="text"
          inputMode="numeric"
          pattern="[0-9]*"
          value={teamId}
          onChange={(e) => setTeamId(e.target.value.replace(/\D/g, ''))}
          onKeyDown={handleKeyDown}
          placeholder="Enter your FPL Team ID"
          disabled={loading}
          className={`flex-1 px-3 py-2.5 rounded-lg text-sm outline-none transition-colors ${
            darkMode
              ? 'bg-[#1f1f1f] text-white border border-[#333] focus:border-[#86efac] placeholder-gray-600'
              : 'bg-white text-gray-900 border border-gray-300 focus:border-emerald-500 placeholder-gray-400'
          } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
        />
        <button
          onClick={handleSync}
          disabled={loading || !teamId.trim()}
          className={`px-4 py-2.5 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 shrink-0 ${
            loading || !teamId.trim()
              ? darkMode
                ? 'bg-[#1f1f1f] text-gray-600 cursor-not-allowed'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-[#86efac] text-black hover:bg-[#6dd89a] active:scale-[0.98]'
          }`}
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-black/20 border-t-black" />
              Syncing...
            </>
          ) : (
            <>
              <Zap size={14} />
              Sync Squad
            </>
          )}
        </button>
      </div>

      <p className={`text-xs ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
        Find your ID at fantasy.premierleague.com &rarr; My Team &rarr; check the URL
      </p>

      {error && (
        <p className={`text-xs ${darkMode ? 'text-red-400' : 'text-red-500'}`}>
          {error}
        </p>
      )}
    </div>
  );
}
