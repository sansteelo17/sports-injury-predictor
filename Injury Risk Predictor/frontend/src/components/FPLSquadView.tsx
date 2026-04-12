'use client';

import { FPLSquadSync, FPLSquadPlayer } from '@/types/api';
import { RiskBadge } from './RiskBadge';
import { User, RefreshCw, AlertTriangle, LayoutGrid, List } from 'lucide-react';
import { useState } from 'react';
import { toAbsoluteApiUrl } from '@/lib/api';

interface FPLSquadViewProps {
  squad: FPLSquadSync;
  onSelectPlayer: (name: string) => void;
  selectedPlayer?: string;
  onRefresh: () => void;
  darkMode: boolean;
}

function PlayerImage({ url, name, darkMode, size = 'sm' }: { url: string | null; name: string; darkMode: boolean; size?: 'sm' | 'lg' }) {
  const [errored, setErrored] = useState(false);
  const dim = size === 'lg' ? 'w-11 h-11' : 'w-8 h-8';
  const iconSize = size === 'lg' ? 18 : 14;
  if (!url || errored) {
    return (
      <div className={`${dim} rounded-full flex items-center justify-center flex-shrink-0 ${
        darkMode ? 'bg-[#1f1f1f]' : 'bg-gray-100'
      }`}>
        <User size={iconSize} className={darkMode ? 'text-gray-600' : 'text-gray-400'} />
      </div>
    );
  }
  return (
    <img
      src={toAbsoluteApiUrl(url)}
      alt={name}
      className={`${dim} rounded-full object-cover flex-shrink-0`}
      onError={() => setErrored(true)}
    />
  );
}

function CaptainBadge({ type, darkMode, size = 'sm' }: { type: 'C' | 'V'; darkMode: boolean; size?: 'sm' | 'lg' }) {
  const isCaptain = type === 'C';
  const dim = size === 'lg' ? 'w-5 h-5 text-[10px]' : 'w-4.5 h-4.5 text-[9px]';
  return (
    <span className={`inline-flex items-center justify-center rounded-full font-black flex-shrink-0 ${dim} ${
      isCaptain
        ? 'bg-amber-400 text-black'
        : darkMode
          ? 'bg-gray-500 text-black'
          : 'bg-gray-400 text-white'
    }`}>
      {type}
    </span>
  );
}

function riskColor(player: FPLSquadPlayer, darkMode: boolean): string {
  if (player.is_currently_injured) return 'border-red-500';
  if (player.risk_level === 'High') return 'border-red-400';
  if (player.risk_level === 'Medium') return 'border-amber-400';
  return darkMode ? 'border-[#86efac]' : 'border-emerald-500';
}

function riskDot(player: FPLSquadPlayer, darkMode: boolean): string {
  if (player.is_currently_injured) return 'bg-red-500';
  if (player.risk_level === 'High') return 'bg-red-400';
  if (player.risk_level === 'Medium') return 'bg-amber-400';
  return darkMode ? 'bg-[#86efac]' : 'bg-emerald-500';
}

function compactPitchName(name: string): string {
  const clean = name.trim();
  const parts = clean.split(/\s+/).filter(Boolean);

  if (clean.length <= 12 || parts.length <= 1) return clean;

  const last = parts[parts.length - 1];
  if (last.length <= 12) return last;

  const first = parts[0];
  if (first.length <= 12) return first;

  return `${first.slice(0, 10)}.`;
}

/* ─── List view player row (numbered) ─── */

function SquadPlayerRow({ player, index, onSelect, isSelected, darkMode }: {
  player: FPLSquadPlayer;
  index: number;
  onSelect: () => void;
  isSelected: boolean;
  darkMode: boolean;
}) {
  const isBench = player.squad_position > 11;
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left px-3 py-2.5 rounded-lg transition-all ${
        isSelected
          ? darkMode
            ? 'bg-[#86efac]/10 border-2 border-[#86efac]/50'
            : 'bg-emerald-50 border-2 border-emerald-500'
          : darkMode
            ? 'bg-[#141414] hover:bg-[#1f1f1f] border-2 border-transparent hover:border-[#86efac]/30'
            : 'bg-white hover:bg-gray-50 border-2 border-transparent hover:border-emerald-300'
      } ${isBench ? 'opacity-60' : ''}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2.5 min-w-0 flex-1">
          {/* Number */}
          <span className={`text-xs font-mono w-5 text-center flex-shrink-0 ${
            darkMode ? 'text-gray-600' : 'text-gray-400'
          }`}>
            {index}
          </span>
          <PlayerImage url={player.player_image_url} name={player.name} darkMode={darkMode} />
          <div className="min-w-0 flex-1">
            <div className={`font-medium text-sm leading-tight flex items-start gap-1 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              <span className="min-w-0 flex-1 break-words sm:truncate">{player.name}</span>
              {player.is_captain && <CaptainBadge type="C" darkMode={darkMode} />}
              {player.is_vice_captain && <CaptainBadge type="V" darkMode={darkMode} />}
            </div>
            <div className={`text-xs mt-0.5 leading-tight break-words ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
              {player.position} · {player.team}
              {player.shirt_number != null ? ` · #${player.shirt_number}` : ''}
            </div>
          </div>
        </div>
        <div className="text-right flex-shrink-0 pt-0.5">
          {player.is_currently_injured ? (
            <>
              <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${
                darkMode ? 'bg-red-900/40 text-red-400' : 'bg-red-100 text-red-600'
              }`}>OUT</span>
              <div className={`text-xs mt-0.5 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>Injured</div>
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
  );
}

/* ─── Formation pitch player chip ─── */

function PitchPlayer({ player, onSelect, isSelected, darkMode }: {
  player: FPLSquadPlayer;
  onSelect: () => void;
  isSelected: boolean;
  darkMode: boolean;
}) {
  const borderColor = riskColor(player, darkMode);
  const dotColor = riskDot(player, darkMode);

  return (
    <button
      onClick={onSelect}
      className={`flex flex-col items-center gap-1 transition-all group ${
        isSelected ? 'scale-105' : 'hover:scale-105'
      }`}
      title={`${player.name} - ${player.is_currently_injured ? 'Injured' : `${Math.round(player.risk_probability * 100)}% risk`}`}
    >
      {/* Photo with risk ring */}
      <div className="relative">
        <div className={`rounded-full border-2 ${borderColor} ${isSelected ? 'ring-2 ring-offset-1 ring-[#86efac]/60' : ''}`}>
          <PlayerImage url={player.player_image_url} name={player.name} darkMode={darkMode} size="lg" />
        </div>
        {/* Captain badge overlay */}
        {(player.is_captain || player.is_vice_captain) && (
          <div className="absolute -top-1 -right-1">
            <CaptainBadge type={player.is_captain ? 'C' : 'V'} darkMode={darkMode} size="lg" />
          </div>
        )}
        {/* Injury cross */}
        {player.is_currently_injured && (
          <div className="absolute -bottom-0.5 -right-0.5 w-4 h-4 rounded-full bg-red-500 flex items-center justify-center">
            <span className="text-white text-[8px] font-bold">X</span>
          </div>
        )}
      </div>
      {/* Name + risk */}
      <div className="text-center max-w-[82px] sm:max-w-[72px]">
        <div
          className={`text-[10px] font-semibold leading-[1.05] whitespace-normal break-words ${
          darkMode ? 'text-white' : 'text-gray-900'
        }`}
          title={player.name}
        >
          {compactPitchName(player.name)}
        </div>
        <div className="flex items-center justify-center gap-0.5 mt-0.5">
          <div className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
          <span className={`text-[9px] font-medium ${
            player.is_currently_injured ? 'text-red-400' :
            player.risk_level === 'High' ? 'text-red-400' :
            player.risk_level === 'Medium' ? 'text-amber-400' :
            darkMode ? 'text-[#86efac]' : 'text-emerald-600'
          }`}>
            {player.is_currently_injured ? 'OUT' : `${Math.round(player.risk_probability * 100)}%`}
          </span>
        </div>
      </div>
    </button>
  );
}

/* ─── Formation pitch ─── */

function FormationView({ starters, bench, onSelectPlayer, selectedPlayer, darkMode }: {
  starters: FPLSquadPlayer[];
  bench: FPLSquadPlayer[];
  onSelectPlayer: (name: string) => void;
  selectedPlayer?: string;
  darkMode: boolean;
}) {
  // Position classifiers
  const isGK = (p: FPLSquadPlayer) => /^(GK|Goalkeeper)/i.test(p.position);
  const isDEF = (p: FPLSquadPlayer) => /^(DEF|Defender|Centre-Back|Right-Back|Left-Back|Defence)/i.test(p.position);
  const isMID = (p: FPLSquadPlayer) => /^(MID|Midfielder|Central Mid|Attacking Mid|Defensive Mid|Right Mid|Left Mid|Midfield)/i.test(p.position);
  const isFWD = (p: FPLSquadPlayer) => /^(FWD|Forward|Centre-Forward|Right Wing|Left Wing|Offence|Striker)/i.test(p.position);

  const gk = starters.filter(isGK);
  const def = starters.filter(isDEF);
  const mid = starters.filter(isMID);
  const fwd = starters.filter(p => isFWD(p) || (!isGK(p) && !isDEF(p) && !isMID(p)));

  // Derive formation string (excluding GKs)
  const formation = `${def.length}-${mid.length}-${fwd.length}`;

  const renderRow = (players: FPLSquadPlayer[]) => (
    <div className="flex justify-center gap-3 sm:gap-5">
      {players.map(p => (
        <PitchPlayer
          key={p.name}
          player={p}
          onSelect={() => onSelectPlayer(p.name)}
          isSelected={selectedPlayer === p.name}
          darkMode={darkMode}
        />
      ))}
    </div>
  );

  return (
    <div className="space-y-3">
      {/* Pitch */}
      <div className={`relative rounded-xl overflow-hidden ${
        darkMode ? 'bg-gradient-to-b from-[#1a472a] to-[#0d2818]' : 'bg-gradient-to-b from-emerald-600 to-emerald-800'
      }`}>
        {/* Pitch markings */}
        <div className="absolute inset-0 opacity-10 pointer-events-none">
          {/* Halfway line */}
          <div className="absolute top-1/2 left-4 right-4 border-t border-white" />
          {/* Center circle */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-20 h-20 rounded-full border border-white" />
          {/* Penalty area top */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-32 h-12 border-b border-l border-r border-white" />
          {/* Penalty area bottom */}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-32 h-12 border-t border-l border-r border-white" />
        </div>

        {/* Formation label */}
        <div className="absolute top-2 right-3 z-10">
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
            darkMode ? 'bg-black/40 text-white/70' : 'bg-white/20 text-white/80'
          }`}>
            {formation}
          </span>
        </div>

        {/* Player rows */}
        <div className="relative z-10 py-5 px-2 space-y-5">
          {fwd.length > 0 && renderRow(fwd)}
          {mid.length > 0 && renderRow(mid)}
          {def.length > 0 && renderRow(def)}
          {gk.length > 0 && renderRow(gk)}
        </div>
      </div>

      {/* Bench bar */}
      {bench.length > 0 && (
        <div className={`rounded-xl p-3 ${
          darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-gray-50 border border-gray-200'
        }`}>
          <div className={`text-[10px] font-semibold uppercase tracking-wider mb-2 ${
            darkMode ? 'text-gray-500' : 'text-gray-400'
          }`}>
            Bench
          </div>
          <div className="flex justify-center gap-3 sm:gap-5 opacity-70">
            {bench.map(p => (
              <PitchPlayer
                key={p.name}
                player={p}
                onSelect={() => onSelectPlayer(p.name)}
                isSelected={selectedPlayer === p.name}
                darkMode={darkMode}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Main component ─── */

export function FPLSquadView({ squad, onSelectPlayer, selectedPlayer, onRefresh, darkMode }: FPLSquadViewProps) {
  const [view, setView] = useState<'formation' | 'list'>('formation');

  const textClass = darkMode ? 'text-white' : 'text-gray-900';
  const mutedClass = darkMode ? 'text-gray-500' : 'text-gray-500';
  const cardClass = darkMode ? 'bg-[#141414] border-[#1f1f1f]' : 'bg-white border-gray-200';

  const starters = squad.players.filter(p => p.squad_position <= 11);
  const bench = squad.players.filter(p => p.squad_position > 11);

  // All players sorted by squad_position for numbered list view
  const allSorted = [...squad.players].sort((a, b) => a.squad_position - b.squad_position);

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className={`holo-panel ${cardClass} border rounded-xl p-3 sm:p-4`}>
        <div className="flex items-center justify-between gap-2 mb-3">
          <div>
            <h3 className={`font-semibold text-sm sm:text-base ${textClass}`}>
              {squad.entry.team_name}
            </h3>
            <p className={`text-xs ${mutedClass}`}>
              {squad.entry.manager_name} · GW{squad.entry.gameweek}
              {squad.entry.gameweek_points > 0 ? ` · ${squad.entry.gameweek_points} pts` : ''}
              {squad.is_gw_finished && ' · Final'}
            </p>
          </div>
          <div className="flex items-center gap-1">
            {/* View toggle */}
            <div className={`flex rounded-lg border ${darkMode ? 'border-[#1f1f1f]' : 'border-gray-200'}`}>
              <button
                onClick={() => setView('formation')}
                className={`p-1.5 rounded-l-lg transition-colors ${
                  view === 'formation'
                    ? darkMode ? 'bg-[#86efac]/20 text-[#86efac]' : 'bg-emerald-100 text-emerald-700'
                    : darkMode ? 'text-gray-500 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600'
                }`}
                title="Formation view"
              >
                <LayoutGrid size={14} />
              </button>
              <button
                onClick={() => setView('list')}
                className={`p-1.5 rounded-r-lg transition-colors ${
                  view === 'list'
                    ? darkMode ? 'bg-[#86efac]/20 text-[#86efac]' : 'bg-emerald-100 text-emerald-700'
                    : darkMode ? 'text-gray-500 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600'
                }`}
                title="List view"
              >
                <List size={14} />
              </button>
            </div>
            <button
              onClick={onRefresh}
              className={`p-2 rounded-lg transition-colors ${
                darkMode ? 'hover:bg-white/10 text-gray-400 hover:text-white' : 'hover:bg-gray-100 text-gray-500'
              }`}
              title="Refresh squad"
            >
              <RefreshCw size={14} />
            </button>
          </div>
        </div>

        {/* Risk summary */}
        <div className="flex gap-3 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-red-400" />
            <span className={mutedClass}>{squad.high_risk_count} high</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-amber-400" />
            <span className={mutedClass}>{squad.medium_risk_count} med</span>
          </div>
          <div className="flex items-center gap-1">
            <div className={`w-2 h-2 rounded-full ${darkMode ? 'bg-[#86efac]' : 'bg-emerald-500'}`} />
            <span className={mutedClass}>{squad.low_risk_count} low</span>
          </div>
        </div>

        {squad.is_gw_finished && (
          <p className={`text-[10px] mt-2 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
            GW{squad.entry.gameweek} is complete. Pending transfers applied. Starter/bench split updates when GW{squad.entry.gameweek + 1} begins.
          </p>
        )}
      </div>

      {/* Formation or List view */}
      {view === 'formation' ? (
        <FormationView
          starters={starters}
          bench={bench}
          onSelectPlayer={onSelectPlayer}
          selectedPlayer={selectedPlayer}
          darkMode={darkMode}
        />
      ) : (
        <div className={`holo-panel ${cardClass} border rounded-xl p-3 sm:p-4`}>
          <div className="space-y-2">
            {allSorted.map((player, i) => (
              <SquadPlayerRow
                key={player.name}
                player={player}
                index={i + 1}
                onSelect={() => onSelectPlayer(player.name)}
                isSelected={selectedPlayer === player.name}
                darkMode={darkMode}
              />
            ))}
          </div>
        </div>
      )}

      {/* Unmatched warning */}
      {squad.unmatched.length > 0 && (
        <div className={`flex items-start gap-2 text-xs p-3 rounded-lg ${
          darkMode ? 'bg-amber-900/10 text-amber-400/70' : 'bg-amber-50 text-amber-600'
        }`}>
          <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
          <span>Risk data unavailable for: {squad.unmatched.join(', ')}</span>
        </div>
      )}
    </div>
  );
}
