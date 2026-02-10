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
  ChevronRight
} from 'lucide-react';

interface PlayerCardProps {
  player: PlayerRisk;
}

export function PlayerCard({ player }: PlayerCardProps) {
  const archetypeColors: Record<string, string> = {
    'Durable': 'bg-green-50 border-green-200 text-green-800',
    'Fragile': 'bg-red-50 border-red-200 text-red-800',
    'Currently Vulnerable': 'bg-orange-50 border-orange-200 text-orange-800',
    'Injury Prone': 'bg-yellow-50 border-yellow-200 text-yellow-800',
    'Recurring': 'bg-purple-50 border-purple-200 text-purple-800',
    'Moderate Risk': 'bg-blue-50 border-blue-200 text-blue-800',
    'Clean Record': 'bg-emerald-50 border-emerald-200 text-emerald-800',
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl overflow-hidden max-w-2xl mx-auto">
      {/* Header */}
      <div className="bg-gradient-to-r from-pl-purple to-purple-900 px-6 py-6 text-white">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="text-2xl font-bold">{player.name}</h2>
            <div className="flex items-center gap-2 mt-1 text-purple-200">
              <span>{player.team}</span>
              <span>•</span>
              <span>{player.position}</span>
              <span>•</span>
              <span>Age {player.age}</span>
            </div>
          </div>
          <RiskBadge level={player.risk_level} probability={player.risk_probability} size="lg" />
        </div>
      </div>

      {/* Risk Meter */}
      <div className="px-6 py-4 border-b border-gray-100">
        <RiskMeter probability={player.risk_probability} />
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-6 bg-gray-50">
        <StatCard
          icon={<Activity size={20} />}
          label="Previous Injuries"
          value={player.factors.previous_injuries.toString()}
        />
        <StatCard
          icon={<Clock size={20} />}
          label="Days Lost"
          value={player.factors.total_days_lost.toString()}
        />
        <StatCard
          icon={<Calendar size={20} />}
          label="Days Since Last"
          value={player.factors.days_since_last_injury.toString()}
        />
        <StatCard
          icon={<TrendingUp size={20} />}
          label="Avg Days/Injury"
          value={player.factors.avg_days_per_injury.toFixed(1)}
        />
      </div>

      {/* Archetype */}
      <div className="px-6 py-4 border-b border-gray-100">
        <div className="flex items-start gap-3">
          <Shield className="text-gray-400 mt-0.5" size={20} />
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-gray-900">Player Profile</span>
              <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${archetypeColors[player.archetype] || 'bg-gray-50 border-gray-200 text-gray-800'}`}>
                {player.archetype}
              </span>
            </div>
            <p className="text-sm text-gray-600 mt-1">{player.archetype_description}</p>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="px-6 py-4">
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle className="text-amber-500" size={20} />
          <span className="font-semibold text-gray-900">Recommendations</span>
        </div>
        <ul className="space-y-2">
          {player.recommendations.map((rec, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
              <ChevronRight className="text-gray-400 mt-0.5 flex-shrink-0" size={16} />
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Model Predictions (Collapsible) */}
      <details className="border-t border-gray-100">
        <summary className="px-6 py-3 cursor-pointer text-sm text-gray-500 hover:bg-gray-50">
          Model Details
        </summary>
        <div className="px-6 pb-4">
          <div className="grid grid-cols-3 gap-4 text-center">
            <ModelScore label="LightGBM" value={player.model_predictions.lgb} />
            <ModelScore label="XGBoost" value={player.model_predictions.xgb} />
            <ModelScore label="CatBoost" value={player.model_predictions.catboost} />
          </div>
        </div>
      </details>
    </div>
  );
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="bg-white rounded-lg p-3 text-center shadow-sm">
      <div className="flex justify-center text-gray-400 mb-1">{icon}</div>
      <div className="text-xl font-bold text-gray-900">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

function ModelScore({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="text-lg font-semibold text-gray-800">{Math.round(value * 100)}%</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}
