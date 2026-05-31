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
    <div className={`holo-panel ${cardClass} border rounded-xl p-3 mb-4`}>
      <div className="flex items-center gap-1.5 mb-2">
        <Trophy size={13} className={darkMode ? "text-[#86efac]" : "text-emerald-600"} />
        <h3 className="text-xs font-semibold">Who wins it</h3>
        <span className={`text-[9px] ${muted}`}>top 10 · bookmaker market</span>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        {data.markets.slice(0, 10).map((row) => (
          <div key={row.team} className="flex items-center gap-1.5 min-w-0">
            {row.flag_url ? (
              <img
                src={row.flag_url}
                alt=""
                width={16}
                height={11}
                className="flex-shrink-0 rounded-[2px] object-cover"
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).style.visibility = "hidden";
                }}
              />
            ) : (
              <span className="w-4 flex-shrink-0" />
            )}
            <span className="text-xs truncate flex-1 min-w-0">{row.team}</span>
            <span className="text-xs font-semibold tabular-nums">
              {Math.round(row.win_probability * 100)}%
            </span>
          </div>
        ))}
      </div>

      {data.bookmakers?.length > 0 && (
        <div className={`flex flex-wrap items-center gap-x-2 gap-y-1 mt-3 pt-2 border-t ${darkMode ? "border-[#1f1f1f]" : "border-gray-200"} ${muted}`}>
          <span className="text-[9px]">Odds from</span>
          {data.bookmakers.map((b) => (
            <span key={b.key} className="inline-flex items-center gap-1">
              <img
                src={`https://logo.clearbit.com/${bookieDomain(b.key, b.title)}`}
                alt=""
                title={b.title}
                width={12}
                height={12}
                className="rounded-sm opacity-80"
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).style.display = "none";
                }}
              />
              <span className="text-[9px]">{b.title}</span>
            </span>
          ))}
        </div>
      )}

      <p className={`text-[10px] mt-2 ${muted}`}>Prediction shown is the bookmaker market, not betting advice.</p>
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
