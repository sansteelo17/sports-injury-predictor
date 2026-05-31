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

      {data.bookmakers?.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <span className={`text-[10px] uppercase tracking-wider ${muted}`}>
            {data.bookmakers.length} books
          </span>
          {data.bookmakers.map((b) => (
            <span
              key={b.key}
              className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] ${
                darkMode ? "bg-[#1f1f1f] text-gray-400" : "bg-gray-100 text-gray-600"
              }`}
            >
              <img
                src={`https://logo.clearbit.com/${bookieDomain(b.key, b.title)}`}
                alt=""
                width={12}
                height={12}
                className="rounded-sm"
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).style.display = "none";
                }}
              />
              {b.title}
            </span>
          ))}
        </div>
      )}

      <p className={`text-[11px] mt-3 ${muted}`}>{data.disclaimer}</p>
    </div>
  );
}

// Best-effort logo domains for common sportsbooks (Clearbit serves by domain).
// Unknown books fall back to text only via the img onError handler.
const BOOKIE_DOMAINS: Record<string, string> = {
  bet365: "bet365.com",
  williamhill: "williamhill.com",
  williamhill_us: "williamhill.com",
  betfair: "betfair.com",
  betfair_ex_uk: "betfair.com",
  unibet: "unibet.com",
  unibet_uk: "unibet.co.uk",
  pinnacle: "pinnacle.com",
  betway: "betway.com",
  ladbrokes_uk: "ladbrokes.com",
  coral: "coral.co.uk",
  skybet: "skybet.com",
  paddypower: "paddypower.com",
  draftkings: "draftkings.com",
  fanduel: "fanduel.com",
  betmgm: "betmgm.com",
  betrivers: "betrivers.com",
  pointsbetus: "pointsbet.com",
  betonlineag: "betonline.ag",
  bovada: "bovada.lv",
  mybookieag: "mybookie.ag",
  marathonbet: "marathonbet.com",
  nordicbet: "nordicbet.com",
  betsson: "betsson.com",
  onexbet: "1xbet.com",
  matchbook: "matchbook.com",
  casumo: "casumo.com",
  grosvenor: "grosvenorcasinos.com",
  virginbet: "virginbet.com",
  betvictor: "betvictor.com",
  boylesports: "boylesports.com",
  mrgreen: "mrgreen.com",
  leovegas: "leovegas.com",
};

function bookieDomain(key: string, title: string): string {
  if (BOOKIE_DOMAINS[key]) return BOOKIE_DOMAINS[key];
  // Heuristic: collapse the title to a .com guess (works for many books).
  const slug = title.toLowerCase().replace(/[^a-z0-9]/g, "");
  return `${slug}.com`;
}
