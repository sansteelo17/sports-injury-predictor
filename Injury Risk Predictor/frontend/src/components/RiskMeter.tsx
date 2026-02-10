'use client';

interface RiskMeterProps {
  probability: number;
  showPercentage?: boolean;
  darkMode?: boolean;
}

export function RiskMeter({ probability, showPercentage = true, darkMode = true }: RiskMeterProps) {
  const percentage = Math.round(probability * 100);

  const getColor = (pct: number) => {
    if (pct >= 60) return 'bg-gradient-to-r from-red-400 to-red-600';
    if (pct >= 35) return 'bg-gradient-to-r from-amber-400 to-amber-500';
    return 'bg-gradient-to-r from-[#4ade80] to-[#86efac]';
  };

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-1">
        <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          2-Week Injury Risk
        </span>
        {showPercentage && (
          <span className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            {percentage}%
          </span>
        )}
      </div>
      <div className={`h-3 rounded-full overflow-hidden ${darkMode ? 'bg-[#1f1f1f]' : 'bg-gray-200'}`}>
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${getColor(percentage)}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className={`flex justify-between text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
        <span>Low</span>
        <span>High</span>
      </div>
    </div>
  );
}
