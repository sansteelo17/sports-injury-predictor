'use client';

import { PlayerRisk } from '@/types/api';
import { RiskBadge } from './RiskBadge';
import { RiskMeter } from './RiskMeter';
import {
  User,
  Calendar,
  Activity,
  Clock,
  TrendingUp,
  Shield,
  AlertCircle,
  ChevronRight,
  FileText
} from 'lucide-react';

interface PlayerCardProps {
  player: PlayerRisk;
  darkMode?: boolean;
}

export function PlayerCard({ player, darkMode = true }: PlayerCardProps) {
  const archetypeColors: Record<string, { dark: string; light: string }> = {
    'Durable': {
      dark: 'bg-[#86efac]/20 border-[#86efac]/30 text-[#86efac]',
      light: 'bg-green-50 border-green-200 text-green-800',
    },
    'Fragile': {
      dark: 'bg-red-500/20 border-red-500/30 text-red-400',
      light: 'bg-red-50 border-red-200 text-red-800',
    },
    'Currently Vulnerable': {
      dark: 'bg-orange-500/20 border-orange-500/30 text-orange-400',
      light: 'bg-orange-50 border-orange-200 text-orange-800',
    },
    'Injury Prone': {
      dark: 'bg-amber-500/20 border-amber-500/30 text-amber-400',
      light: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    },
    'Recurring': {
      dark: 'bg-purple-500/20 border-purple-500/30 text-purple-400',
      light: 'bg-purple-50 border-purple-200 text-purple-800',
    },
    'Moderate Risk': {
      dark: 'bg-blue-500/20 border-blue-500/30 text-blue-400',
      light: 'bg-blue-50 border-blue-200 text-blue-800',
    },
    'Clean Record': {
      dark: 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400',
      light: 'bg-emerald-50 border-emerald-200 text-emerald-800',
    },
  };

  const getArchetypeClasses = (archetype: string) => {
    const colors = archetypeColors[archetype] || {
      dark: 'bg-gray-500/20 border-gray-500/30 text-gray-400',
      light: 'bg-gray-50 border-gray-200 text-gray-800',
    };
    return darkMode ? colors.dark : colors.light;
  };

  return (
    <div className={`rounded-2xl overflow-hidden max-w-2xl mx-auto ${
      darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-xl'
    }`}>
      {/* Header */}
      <div className={`px-6 py-6 ${
        darkMode
          ? 'bg-gradient-to-r from-[#1f1f1f] to-[#141414] border-b border-[#1f1f1f]'
          : 'bg-gradient-to-r from-purple-600 to-purple-900 text-white'
      }`}>
        <div className="flex justify-between items-start">
          <div>
            <h2 className={`text-2xl font-bold ${darkMode ? 'text-white' : ''}`}>{player.name}</h2>
            <div className={`flex items-center gap-2 mt-1 ${darkMode ? 'text-gray-400' : 'text-purple-200'}`}>
              <span>{player.team}</span>
              <span>•</span>
              <span>{player.position}</span>
              <span>•</span>
              <span>Age {player.age}</span>
            </div>
          </div>
          <RiskBadge level={player.risk_level} probability={player.risk_probability} size="lg" darkMode={darkMode} />
        </div>
      </div>

      {/* Risk Meter */}
      <div className={`px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
        <RiskMeter probability={player.risk_probability} darkMode={darkMode} />
      </div>

      {/* Stats Grid */}
      <div className={`grid grid-cols-2 md:grid-cols-4 gap-4 p-6 ${darkMode ? 'bg-[#0a0a0a]' : 'bg-gray-50'}`}>
        <StatCard
          icon={<Activity size={20} />}
          label="Previous Injuries"
          value={player.factors.previous_injuries.toString()}
          darkMode={darkMode}
        />
        <StatCard
          icon={<Clock size={20} />}
          label="Days Lost"
          value={player.factors.total_days_lost.toString()}
          darkMode={darkMode}
        />
        <StatCard
          icon={<Calendar size={20} />}
          label="Days Since Last"
          value={player.factors.days_since_last_injury.toString()}
          darkMode={darkMode}
        />
        <StatCard
          icon={<TrendingUp size={20} />}
          label="Avg Days/Injury"
          value={player.factors.avg_days_per_injury.toFixed(1)}
          darkMode={darkMode}
        />
      </div>

      {/* Archetype */}
      <div className={`px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
        <div className="flex items-start gap-3">
          <Shield className={darkMode ? 'text-gray-500 mt-0.5' : 'text-gray-400 mt-0.5'} size={20} />
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Player Profile</span>
              <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${getArchetypeClasses(player.archetype)}`}>
                {player.archetype}
              </span>
            </div>
            <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              {player.archetype_description}
            </p>
          </div>
        </div>
      </div>

      {/* Risk Story */}
      {player.story && (
        <div className={`px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className="flex items-start gap-3">
            <FileText className={darkMode ? 'text-[#86efac] mt-0.5' : 'text-green-500 mt-0.5'} size={20} />
            <div className="flex-1">
              <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Risk Analysis</span>
              <p className={`text-sm mt-2 leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                {player.story}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="px-6 py-4">
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle className="text-amber-500" size={20} />
          <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Key Factors & Recommendations</span>
        </div>
        <ul className="space-y-2">
          {player.recommendations.map((rec, i) => (
            <li key={i} className={`flex items-start gap-2 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              <ChevronRight className={`mt-0.5 flex-shrink-0 ${darkMode ? 'text-[#86efac]' : 'text-gray-400'}`} size={16} />
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Model Predictions (Collapsible) */}
      <details className={darkMode ? 'border-t border-[#1f1f1f]' : 'border-t border-gray-100'}>
        <summary className={`px-6 py-3 cursor-pointer text-sm ${
          darkMode ? 'text-gray-500 hover:bg-[#1f1f1f]' : 'text-gray-500 hover:bg-gray-50'
        }`}>
          Model Details
        </summary>
        <div className="px-6 pb-4">
          <div className="grid grid-cols-3 gap-4 text-center">
            <ModelScore label="LightGBM" value={player.model_predictions.lgb} darkMode={darkMode} />
            <ModelScore label="XGBoost" value={player.model_predictions.xgb} darkMode={darkMode} />
            <ModelScore label="CatBoost" value={player.model_predictions.catboost} darkMode={darkMode} />
          </div>
        </div>
      </details>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  darkMode = true,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  darkMode?: boolean;
}) {
  return (
    <div className={`rounded-lg p-3 text-center ${
      darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-sm'
    }`}>
      <div className={`flex justify-center mb-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{icon}</div>
      <div className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{value}</div>
      <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{label}</div>
    </div>
  );
}

function ModelScore({
  label,
  value,
  darkMode = true,
}: {
  label: string;
  value: number;
  darkMode?: boolean;
}) {
  return (
    <div>
      <div className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
        {Math.round(value * 100)}%
      </div>
      <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{label}</div>
    </div>
  );
}
