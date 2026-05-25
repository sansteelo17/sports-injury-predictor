'use client';

import { AlertTriangle, CheckCircle, AlertCircle, HelpCircle } from 'lucide-react';

interface RiskBadgeProps {
  level: 'High' | 'Medium' | 'Low' | 'Unknown';
  probability?: number;
  size?: 'sm' | 'md' | 'lg';
  darkMode?: boolean;
}

export function RiskBadge({ level, probability, size = 'md', darkMode = true }: RiskBadgeProps) {
  const config = {
    High: {
      dark: 'bg-red-500/20 text-red-400 border-red-500/30',
      light: 'bg-red-100 text-red-800 border-red-200',
      icon: AlertTriangle,
    },
    Medium: {
      dark: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
      light: 'bg-amber-100 text-amber-800 border-amber-200',
      icon: AlertCircle,
    },
    Low: {
      dark: 'bg-[#86efac]/20 text-[#86efac] border-[#86efac]/30',
      light: 'bg-emerald-100 text-emerald-700 border-emerald-200',
      icon: CheckCircle,
    },
    Unknown: {
      // No model signal (e.g. WC squad player whose club league we don't track).
      // Neutral styling so it doesn't look like a Low-risk verdict.
      dark: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
      light: 'bg-slate-100 text-slate-600 border-slate-200',
      icon: HelpCircle,
    },
  };

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-2 text-base',
  };

  const iconSizes = {
    sm: 12,
    md: 14,
    lg: 18,
  };

  const { dark, light, icon: Icon } = config[level];
  const colorClasses = darkMode ? dark : light;

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-semibold rounded-full border ${colorClasses} ${sizeClasses[size]}`}
    >
      <Icon size={iconSizes[size]} />
      {level} Risk
      {probability !== undefined && (
        <span className="opacity-75">({Math.round(probability * 100)}%)</span>
      )}
    </span>
  );
}
