'use client';

import { useState } from 'react';
import { LaLigaStandingRow } from '@/types/api';

interface UclStandingsCardsProps {
  standings: LaLigaStandingRow[];
  selectedTeam?: string;
  darkMode?: boolean;
}

function TeamBadge({ url }: { url: string | null }) {
  const [errored, setErrored] = useState(false);
  if (!url || errored) return null;
  return (
    <img
      src={url}
      alt=""
      width={18}
      height={18}
      className="flex-shrink-0 object-contain"
      onError={() => setErrored(true)}
    />
  );
}

// New 36-team league phase: 1-8 advance straight to the round of 16,
// 9-24 into a knockout play-off, 25-36 are eliminated.
function zone(pos: number): "qualify" | "playoff" | "out" {
  if (pos <= 8) return "qualify";
  if (pos <= 24) return "playoff";
  return "out";
}

export function UclStandingsCards({ standings, selectedTeam, darkMode = true }: UclStandingsCardsProps) {
  if (!standings.length) return null;

  const accent = (z: ReturnType<typeof zone>) =>
    z === "qualify"
      ? darkMode ? "border-l-2 border-l-emerald-400" : "border-l-2 border-l-emerald-500"
      : z === "playoff"
        ? darkMode ? "border-l-2 border-l-amber-400" : "border-l-2 border-l-amber-500"
        : darkMode ? "border-l-2 border-l-gray-600" : "border-l-2 border-l-gray-300";

  return (
    <div className={`rounded-xl border overflow-hidden ${darkMode ? "bg-[#0b1220] border-gray-800" : "bg-white border-gray-200"}`}>
      <div className={`px-3 py-2 flex items-center justify-between ${darkMode ? "border-b border-gray-800" : "border-b border-gray-100"}`}>
        <span className={`text-xs font-medium uppercase tracking-wider ${darkMode ? "text-[#86efac]" : "text-emerald-700"}`}>
          🇪🇺 Champions League — League Phase
        </span>
        <span className={`text-[10px] ${darkMode ? "text-gray-500" : "text-gray-400"}`}>P · W-D-L · GD · Pts</span>
      </div>
      <div className="max-h-[28rem] overflow-y-auto">
        {standings.map((r) => {
          const pos = r.position ?? 0;
          const z = zone(pos);
          const selected = selectedTeam && r.name.toLowerCase() === selectedTeam.toLowerCase();
          return (
            <div
              key={`${pos}-${r.name}`}
              className={`flex items-center gap-2 px-3 py-1.5 text-sm ${accent(z)} ${
                selected
                  ? darkMode ? "bg-[#86efac]/10" : "bg-emerald-50"
                  : darkMode ? "hover:bg-white/5" : "hover:bg-gray-50"
              }`}
            >
              <span className={`w-5 text-right tabular-nums text-xs ${darkMode ? "text-gray-500" : "text-gray-400"}`}>{pos}</span>
              <TeamBadge url={r.badge_url} />
              <span className={`flex-1 truncate ${darkMode ? "text-gray-200" : "text-gray-800"}`}>{r.name}</span>
              <span className={`tabular-nums text-xs ${darkMode ? "text-gray-500" : "text-gray-400"}`}>{r.played}</span>
              <span className={`hidden sm:inline tabular-nums text-xs w-14 text-center ${darkMode ? "text-gray-500" : "text-gray-400"}`}>
                {r.won}-{r.draw}-{r.lost}
              </span>
              <span className={`tabular-nums text-xs w-8 text-right ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
                {(r.goal_difference ?? 0) > 0 ? `+${r.goal_difference}` : r.goal_difference}
              </span>
              <span className={`tabular-nums text-sm font-semibold w-6 text-right ${darkMode ? "text-white" : "text-gray-900"}`}>{r.points}</span>
            </div>
          );
        })}
      </div>
      <div className={`px-3 py-2 flex gap-3 text-[10px] ${darkMode ? "text-gray-500 border-t border-gray-800" : "text-gray-400 border-t border-gray-100"}`}>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm bg-emerald-500 inline-block" /> 1–8 Round of 16</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm bg-amber-500 inline-block" /> 9–24 Play-off</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm bg-gray-400 inline-block" /> 25–36 Out</span>
      </div>
    </div>
  );
}
