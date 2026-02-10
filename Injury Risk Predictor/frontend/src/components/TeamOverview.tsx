'use client';

import { TeamOverview as TeamOverviewType } from '@/types/api';
import { Users, AlertTriangle, AlertCircle, CheckCircle, TrendingUp } from 'lucide-react';

interface TeamOverviewProps {
  team: TeamOverviewType;
}

export function TeamOverview({ team }: TeamOverviewProps) {
  const highPct = Math.round((team.high_risk_count / team.total_players) * 100);
  const medPct = Math.round((team.medium_risk_count / team.total_players) * 100);
  const lowPct = Math.round((team.low_risk_count / team.total_players) * 100);

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-pl-purple to-purple-900 px-6 py-5 text-white">
        <h2 className="text-xl font-bold">{team.team}</h2>
        <p className="text-purple-200 text-sm">{team.total_players} players analyzed</p>
      </div>

      {/* Risk Distribution Bar */}
      <div className="px-6 py-4 border-b border-gray-100">
        <div className="text-sm text-gray-600 mb-2">Squad Risk Distribution</div>
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
              className="bg-green-500 flex items-center justify-center text-white text-xs font-medium"
              style={{ width: `${lowPct}%` }}
            >
              {lowPct > 10 ? `${lowPct}%` : ''}
            </div>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 divide-x divide-gray-100">
        <StatBox
          icon={<Users className="text-gray-400" size={18} />}
          value={team.total_players}
          label="Players"
        />
        <StatBox
          icon={<AlertTriangle className="text-red-500" size={18} />}
          value={team.high_risk_count}
          label="High Risk"
          highlight="red"
        />
        <StatBox
          icon={<AlertCircle className="text-amber-500" size={18} />}
          value={team.medium_risk_count}
          label="Medium"
          highlight="amber"
        />
        <StatBox
          icon={<CheckCircle className="text-green-500" size={18} />}
          value={team.low_risk_count}
          label="Low Risk"
          highlight="green"
        />
      </div>

      {/* Average Risk */}
      <div className="px-6 py-4 bg-gray-50 flex items-center justify-between">
        <div className="flex items-center gap-2 text-gray-600">
          <TrendingUp size={18} />
          <span className="text-sm">Squad Average Risk</span>
        </div>
        <span className={`text-lg font-bold ${
          team.avg_risk >= 0.6 ? 'text-red-600' :
          team.avg_risk >= 0.35 ? 'text-amber-600' : 'text-green-600'
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
}: {
  icon: React.ReactNode;
  value: number;
  label: string;
  highlight?: 'red' | 'amber' | 'green';
}) {
  const highlightClasses = {
    red: 'text-red-600',
    amber: 'text-amber-600',
    green: 'text-green-600',
  };

  return (
    <div className="px-4 py-4 text-center">
      <div className="flex justify-center mb-1">{icon}</div>
      <div className={`text-xl font-bold ${highlight ? highlightClasses[highlight] : 'text-gray-900'}`}>
        {value}
      </div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}
