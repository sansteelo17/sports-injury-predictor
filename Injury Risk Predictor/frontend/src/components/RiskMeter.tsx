'use client';

interface RiskMeterProps {
  probability: number;
  showPercentage?: boolean;
}

export function RiskMeter({ probability, showPercentage = true }: RiskMeterProps) {
  const percentage = Math.round(probability * 100);

  // Color gradient from green -> yellow -> red
  const getColor = (pct: number) => {
    if (pct >= 60) return 'bg-gradient-to-r from-red-400 to-red-600';
    if (pct >= 35) return 'bg-gradient-to-r from-amber-400 to-amber-500';
    return 'bg-gradient-to-r from-green-400 to-green-500';
  };

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm text-gray-600">2-Week Injury Risk</span>
        {showPercentage && (
          <span className="text-lg font-bold text-gray-900">{percentage}%</span>
        )}
      </div>
      <div className="risk-meter h-3 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`risk-meter-fill h-full rounded-full ${getColor(percentage)}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>Low</span>
        <span>High</span>
      </div>
    </div>
  );
}
