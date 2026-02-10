'use client';

import { TeamOverview as TeamOverviewType } from '@/types/api';
import { Users, AlertTriangle, AlertCircle, CheckCircle, TrendingUp } from 'lucide-react';

interface TeamOverviewProps {
  team: TeamOverviewType;
  darkMode?: boolean;
}

export function TeamOverview({ team, darkMode = true }: TeamOverviewProps) {
  const highPct = Math.round((team.high_risk_count / team.total_players) * 100);
  const medPct = Math.round((team.medium_risk_count / team.total_players) * 100);
  const lowPct = Math.round((team.low_risk_count / team.total_players) * 100);

  return (
    <div className={`rounded-2xl overflow-hidden ${
      darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-lg'
    }`}>
      {/* Header */}
      <div className={`px-6 py-5 ${
        darkMode
          ? 'bg-gradient-to-r from-[#1f1f1f] to-[#141414] border-b border-[#1f1f1f]'
          : 'bg-gradient-to-r from-purple-600 to-purple-900 text-white'
      }`}>
        <h2 className={`text-xl font-bold ${darkMode ? 'text-white' : ''}`}>{team.team}</h2>
        <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-purple-200'}`}>
          {team.total_players} players analyzed
        </p>
      </div>

      {/* Risk Distribution Bar */}
      <div className={`px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
        <div className={`text-sm mb-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          Squad Risk Distribution
        </div>
        <div className="flex h-6 rounded-full overflow-hidden">
          {team.high_risk_count > 0 && (
            <div
              className="bg-red-500 flex items-center justify-center text-white text-xs font-medium"
              style={{ width: `${highPct}%` }}
            >
              {highPct > 10 ? `${highPct}%` : ''}
            </div>
          )}
          {team.medium_risk_count > 0 && (
            <div
              className="bg-amber-400 flex items-center justify-center text-amber-900 text-xs font-medium"
              style={{ width: `${medPct}%` }}
            >
              {medPct > 10 ? `${medPct}%` : ''}
            </div>
          )}
          {team.low_risk_count > 0 && (
            <div
              className="bg-[#86efac] flex items-center justify-center text-[#0a0a0a] text-xs font-medium"
              style={{ width: `${lowPct}%` }}
            >
              {lowPct > 10 ? `${lowPct}%` : ''}
            </div>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className={`grid grid-cols-4 ${darkMode ? 'divide-x divide-[#1f1f1f]' : 'divide-x divide-gray-100'}`}>
        <StatBox
          icon={<Users className={darkMode ? 'text-gray-500' : 'text-gray-400'} size={18} />}
          value={team.total_players}
          label="Players"
          darkMode={darkMode}
        />
        <StatBox
          icon={<AlertTriangle className="text-red-500" size={18} />}
          value={team.high_risk_count}
          label="High Risk"
          highlight="red"
          darkMode={darkMode}
        />
        <StatBox
          icon={<AlertCircle className="text-amber-500" size={18} />}
          value={team.medium_risk_count}
          label="Medium"
          highlight="amber"
          darkMode={darkMode}
        />
        <StatBox
          icon={<CheckCircle className="text-[#86efac]" size={18} />}
          value={team.low_risk_count}
          label="Low Risk"
          highlight="green"
          darkMode={darkMode}
        />
      </div>

      {/* Average Risk */}
      <div className={`px-6 py-4 flex items-center justify-between ${
        darkMode ? 'bg-[#0a0a0a]' : 'bg-gray-50'
      }`}>
        <div className={`flex items-center gap-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          <TrendingUp size={18} />
          <span className="text-sm">Squad Average Risk</span>
        </div>
        <span className={`text-lg font-bold ${
          team.avg_risk >= 0.6 ? 'text-red-500' :
          team.avg_risk >= 0.35 ? 'text-amber-500' : 'text-[#86efac]'
        }`}>
          {Math.round(team.avg_risk * 100)}%
        </span>
      </div>
    </div>
  );
}

function StatBox({
  icon,
  value,
  label,
  highlight,
  darkMode = true,
}: {
  icon: React.ReactNode;
  value: number;
  label: string;
  highlight?: 'red' | 'amber' | 'green';
  darkMode?: boolean;
}) {
  const highlightClasses = {
    red: 'text-red-500',
    amber: 'text-amber-500',
    green: 'text-[#86efac]',
  };

  return (
    <div className="px-4 py-4 text-center">
      <div className="flex justify-center mb-1">{icon}</div>
      <div className={`text-xl font-bold ${
        highlight ? highlightClasses[highlight] : (darkMode ? 'text-white' : 'text-gray-900')
      }`}>
        {value}
      </div>
      <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{label}</div>
    </div>
  );
}
