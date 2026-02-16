'use client';

import { PlayerRisk } from '@/types/api';
import { Microscope, AlertTriangle, ShieldCheck, Minus } from 'lucide-react';

interface LabNotesProps {
  player: PlayerRisk;
  darkMode?: boolean;
}

export function LabNotes({ player, darkMode = true }: LabNotesProps) {
  if (!player.lab_notes) return null;

  const impactIcon = (impact: string) => {
    if (impact === 'risk_increasing') return <AlertTriangle size={14} className="text-red-400" />;
    if (impact === 'protective') return <ShieldCheck size={14} className="text-green-400" />;
    return <Minus size={14} className={darkMode ? 'text-gray-500' : 'text-gray-400'} />;
  };

  return (
    <div className={`rounded-2xl overflow-hidden ${
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
            Risk analysis for {player.name}
          </p>
        </div>
      </div>

      {/* Summary */}
      <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
        <p className={`text-sm leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          {player.lab_notes.summary}
        </p>
      </div>

      {/* Key Drivers */}
      {player.lab_notes.key_drivers.length > 0 && (
        <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <h3 className={`text-sm font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Key Drivers
          </h3>
          <div className="space-y-3">
            {player.lab_notes.key_drivers.map((driver, i) => (
              <div key={i} className={`rounded-lg p-3 ${
                driver.impact === 'risk_increasing'
                  ? darkMode ? 'bg-red-500/10 border border-red-500/20' : 'bg-red-50 border border-red-100'
                  : driver.impact === 'protective'
                  ? darkMode ? 'bg-green-500/10 border border-green-500/20' : 'bg-green-50 border border-green-100'
                  : darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f]' : 'bg-gray-50 border border-gray-100'
              }`}>
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    {impactIcon(driver.impact)}
                    <span className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      {driver.name}
                    </span>
                  </div>
                  <span className={`text-sm font-mono font-bold ${
                    driver.impact === 'risk_increasing'
                      ? darkMode ? 'text-red-400' : 'text-red-600'
                      : driver.impact === 'protective'
                      ? darkMode ? 'text-green-400' : 'text-green-600'
                      : darkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    {driver.value}
                  </span>
                </div>
                <p className={`text-xs leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {driver.explanation}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Technical Details */}
      {player.lab_notes.technical && (
        <div className="px-4 sm:px-6 py-4">
          <h3 className={`text-sm font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Technical Details
          </h3>

          {/* Model Agreement */}
          <div className={`rounded-lg p-3 mb-3 ${
            darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f]' : 'bg-gray-50 border border-gray-100'
          }`}>
            <div className="flex items-center justify-between mb-2">
              <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Model Agreement</span>
              <span className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                {Math.round(player.lab_notes.technical.model_agreement * 100)}%
              </span>
            </div>
            <div className={`w-full h-1.5 rounded-full ${darkMode ? 'bg-[#1f1f1f]' : 'bg-gray-200'}`}>
              <div
                className={`h-full rounded-full ${
                  player.lab_notes.technical.model_agreement >= 0.8
                    ? 'bg-green-500'
                    : player.lab_notes.technical.model_agreement >= 0.6
                    ? 'bg-yellow-500'
                    : 'bg-red-500'
                }`}
                style={{ width: `${Math.round(player.lab_notes.technical.model_agreement * 100)}%` }}
              />
            </div>
          </div>

          {/* Methodology */}
          <p className={`text-xs leading-relaxed mb-3 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            {player.lab_notes.technical.methodology}
          </p>

          {/* Feature Highlights */}
          {player.lab_notes.technical.feature_highlights.length > 0 && (
            <div className={`rounded-lg overflow-hidden ${
              darkMode ? 'border border-[#1f1f1f]' : 'border border-gray-200'
            }`}>
              <div className={`grid grid-cols-2 gap-px ${darkMode ? 'bg-[#1f1f1f]' : 'bg-gray-200'}`}>
                {player.lab_notes.technical.feature_highlights.map((feat, i) => (
                  <div key={i} className={`flex justify-between items-center px-3 py-2 ${
                    darkMode ? 'bg-[#0a0a0a]' : 'bg-white'
                  }`}>
                    <span className={`text-xs truncate mr-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {feat.name}
                    </span>
                    <span className={`text-xs font-mono font-medium flex-shrink-0 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      {typeof feat.value === 'number' ? feat.value.toFixed(2) : feat.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
