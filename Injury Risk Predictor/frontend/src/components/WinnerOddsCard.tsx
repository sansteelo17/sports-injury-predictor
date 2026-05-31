import { Trophy } from "lucide-react";
import { WinnerOdds } from "@/types/api";

interface WinnerOddsCardProps {
  data: WinnerOdds;
  darkMode?: boolean;
}

/**
 * Tournament-winner odds, aggregated across bookmakers and vig-adjusted.
 * Shown with a clear disclaimer: these are market prices, not a Yara
 * prediction, and not betting advice.
 */
export function WinnerOddsCard({ data, darkMode = true }: WinnerOddsCardProps) {
  if (!data?.available || !data.markets?.length) return null;

  const cardClass = darkMode
    ? "bg-[#141414] border-[#1f1f1f]"
    : "bg-white border-gray-200";
  const muted = darkMode ? "text-gray-500" : "text-gray-500";

  return (
    <div className={`holo-panel ${cardClass} border rounded-xl p-4 sm:p-5 mb-4 sm:mb-6`}>
      <div className="flex items-center gap-2 mb-3">
        <Trophy size={16} className={darkMode ? "text-[#86efac]" : "text-emerald-600"} />
        <h3 className="text-sm font-semibold">Who wins it</h3>
        <span className={`text-[10px] ${muted}`}>bookmaker market</span>
      </div>

      <div className="space-y-2">
        {data.markets.map((row) => (
          <div key={row.team} className="flex items-center gap-3">
            <span className="text-sm w-28 truncate">{row.team}</span>
            <div className={`flex-1 h-2 rounded-full overflow-hidden ${darkMode ? "bg-[#1f1f1f]" : "bg-gray-100"}`}>
              <div
                className={darkMode ? "h-full bg-[#86efac]" : "h-full bg-emerald-500"}
                style={{ width: `${Math.min(100, Math.round(row.win_probability * 100 * 3))}%` }}
              />
            </div>
            <span className="text-sm font-semibold tabular-nums w-12 text-right">
              {Math.round(row.win_probability * 100)}%
            </span>
            <span className={`text-xs tabular-nums w-12 text-right ${muted}`}>
              {row.decimal_odds.toFixed(1)}
            </span>
          </div>
        ))}
      </div>

      <p className={`text-[11px] mt-3 ${muted}`}>{data.disclaimer}</p>
    </div>
  );
}
