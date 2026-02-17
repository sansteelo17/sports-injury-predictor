'use client';

import { PlayerRisk } from '@/types/api';
import { Microscope, BrainCircuit, BarChart3, ScrollText } from 'lucide-react';

interface LabNotesProps {
  player: PlayerRisk;
  darkMode?: boolean;
}

export function LabNotes({ player, darkMode = true }: LabNotesProps) {
  if (!player.lab_notes) return null;

  const drivers = player.lab_notes.key_drivers.slice(0, 4);
  const marketOdds = player.bookmaker_consensus?.lines ?? [];
  const hasMarketMovement = marketOdds.length >= 2;
  const marketSpread = hasMarketMovement
    ? Math.max(...marketOdds.map((line) => line.decimal_odds)) -
      Math.min(...marketOdds.map((line) => line.decimal_odds))
    : 0;

  const featureRows = player.lab_notes.technical.feature_highlights.slice(0, 5);
  const featureTotal = featureRows.reduce((sum, feat) => sum + Math.abs(Number(feat.value) || 0), 0);

  return (
    <div className={`holo-card rounded-2xl overflow-hidden ${
      darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-xl'
    }`}>
      {/* Header */}
      <div className={`px-4 sm:px-6 py-4 flex items-center gap-3 ${
        darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'
      }`}>
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
          darkMode ? 'bg-indigo-500/15' : 'bg-indigo-50'
        }`}>
          <Microscope size={20} className={darkMode ? 'text-indigo-400' : 'text-indigo-600'} />
        </div>
        <div>
          <h2 className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Yara&apos;s Lab Notes
          </h2>
          <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            Structured explainability for {player.name}
          </p>
        </div>
      </div>

      <div className={`px-4 sm:px-6 py-4 space-y-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
        <div className={`rounded-lg p-4 ${darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f]' : 'bg-gray-50 border border-gray-200'}`}>
          <div className="flex items-center gap-2 mb-3">
            <BrainCircuit size={16} className={darkMode ? 'text-cyan-300' : 'text-cyan-700'} />
            <h3 className={`text-sm font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Model Drivers This Week
            </h3>
          </div>
          <ul className="space-y-2">
            {drivers.map((driver, i) => (
              <li key={i} className={`text-sm leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                • {driver.name}: {driver.explanation}
              </li>
            ))}
            {hasMarketMovement && (
              <li className={`text-sm leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                • Market odds movement (±): {marketSpread.toFixed(2)} decimal spread across tracked books.
              </li>
            )}
          </ul>
        </div>

        <div className={`rounded-lg p-4 ${darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f]' : 'bg-gray-50 border border-gray-200'}`}>
          <details className="group">
            <summary
              className={`list-none cursor-pointer select-none flex items-center justify-between text-sm font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}
            >
              <span className="flex items-center gap-2">
                <BarChart3 size={16} className={darkMode ? 'text-emerald-300' : 'text-emerald-700'} />
                <span>Feature Importance Snapshot</span>
              </span>
              <span className={`text-xs font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Show
              </span>
            </summary>
            <div className="mt-3 space-y-3">
              <ul className="space-y-2">
                {featureRows.map((feat, i) => {
                  const raw = Number(feat.value) || 0;
                  const weight = featureTotal > 0 ? Math.abs(raw) / featureTotal : 0;
                  return (
                    <li key={i} className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      • {feat.name}: {weight.toFixed(2)} weight
                    </li>
                  );
                })}
                {featureRows.length === 0 && (
                  <li className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    • Feature-level weights are unavailable for this player snapshot.
                  </li>
                )}
              </ul>
              <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                Model agreement: {Math.round(player.lab_notes.technical.model_agreement * 100)}%
              </p>
            </div>
          </details>
        </div>

        <div className={`rounded-lg p-4 ${darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f]' : 'bg-gray-50 border border-gray-200'}`}>
          <div className="flex items-center gap-2 mb-2">
            <ScrollText size={16} className={darkMode ? 'text-amber-300' : 'text-amber-700'} />
            <h3 className={`text-sm font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Transparency Note
            </h3>
          </div>
          <p className={`text-sm leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            This model does not use medical records or private data. It uses publicly available match and performance data.
          </p>
          <p className={`text-xs mt-3 ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
            {player.lab_notes.summary}
          </p>
        </div>
      </div>

      {player.lab_notes.technical?.methodology && (
        <div className="px-4 sm:px-6 py-4">
          <p className={`text-xs leading-relaxed ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
            {player.lab_notes.technical.methodology}
          </p>
        </div>
      )}
    </div>
  );
}
