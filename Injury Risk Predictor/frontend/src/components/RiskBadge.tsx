'use client';

import { AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react';

interface RiskBadgeProps {
  level: 'High' | 'Medium' | 'Low';
  probability?: number;
  size?: 'sm' | 'md' | 'lg';
}

export function RiskBadge({ level, probability, size = 'md' }: RiskBadgeProps) {
  const config = {
    High: {
      bg: 'bg-red-100',
      text: 'text-red-800',
      border: 'border-red-200',
      icon: AlertTriangle,
    },
    Medium: {
      bg: 'bg-amber-100',
      text: 'text-amber-800',
      border: 'border-amber-200',
      icon: AlertCircle,
    },
    Low: {
      bg: 'bg-green-100',
      text: 'text-green-800',
      border: 'border-green-200',
      icon: CheckCircle,
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

  const { bg, text, border, icon: Icon } = config[level];

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-semibold rounded-full border ${bg} ${text} ${border} ${sizeClasses[size]}`}
    >
      <Icon size={iconSizes[size]} />
      {level} Risk
      {probability !== undefined && (
        <span className="opacity-75">({Math.round(probability * 100)}%)</span>
      )}
    </span>
  );
}
