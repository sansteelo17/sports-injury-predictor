"use client";

import { useState } from "react";
import { PlayerRisk } from "@/types/api";
import { RiskBadge } from "./RiskBadge";
import { RiskMeter } from "./RiskMeter";
import {
  Activity,
  Ambulance,
  Calendar,
  Clock,
  TrendingUp,
  Shield,
  AlertCircle,
  ChevronRight,
  FileText,
  Coins,
  Target,
  Star,
  BarChart3,
  User,
  Users,
  Zap,
} from "lucide-react";

interface PlayerCardProps {
  player: PlayerRisk;
  darkMode?: boolean;
}

const FPL_LOGO_SOURCES = [
  "/fpl-logo-official.png",
  "/fpl-logo-official.svg",
  "https://upload.wikimedia.org/wikipedia/en/f/f2/Fantasy_Premier_League_logo.svg",
  "https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Fantasy_Premier_League_logo.svg/512px-Fantasy_Premier_League_logo.svg.png",
  "https://fantasy.premierleague.com/dist/img/fpl-logo.svg",
  "/fpl-badge.svg",
];

function formatStoryLines(story: string): string[] {
  const normalized = (story || "")
    .replace(/\s+/g, " ")
    .replace(/([.!?])\s+(?=[A-Z])/g, "$1\n")
    .replace(/\)\s+(?=[A-Z])/g, ")\n")
    .trim();

  return normalized
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

export function PlayerCard({ player, darkMode = true }: PlayerCardProps) {
  const archetypeColors: Record<string, { dark: string; light: string }> = {
    Durable: {
      dark: "bg-[#86efac]/20 border-[#86efac]/30 text-[#86efac]",
      light: "bg-green-50 border-green-200 text-green-800",
    },
    Fragile: {
      dark: "bg-red-500/20 border-red-500/30 text-red-400",
      light: "bg-red-50 border-red-200 text-red-800",
    },
    "Currently Vulnerable": {
      dark: "bg-orange-500/20 border-orange-500/30 text-orange-400",
      light: "bg-orange-50 border-orange-200 text-orange-800",
    },
    "Injury Prone": {
      dark: "bg-amber-500/20 border-amber-500/30 text-amber-400",
      light: "bg-yellow-50 border-yellow-200 text-yellow-800",
    },
    Recurring: {
      dark: "bg-purple-500/20 border-purple-500/30 text-purple-400",
      light: "bg-purple-50 border-purple-200 text-purple-800",
    },
    "Recurring Issues": {
      dark: "bg-purple-500/20 border-purple-500/30 text-purple-400",
      light: "bg-purple-50 border-purple-200 text-purple-800",
    },
    Unpredictable: {
      dark: "bg-orange-500/20 border-orange-500/30 text-orange-400",
      light: "bg-orange-50 border-orange-200 text-orange-800",
    },
    "Moderate Risk": {
      dark: "bg-blue-500/20 border-blue-500/30 text-blue-400",
      light: "bg-blue-50 border-blue-200 text-blue-800",
    },
    "Clean Record": {
      dark: "bg-emerald-500/20 border-emerald-500/30 text-emerald-400",
      light: "bg-emerald-50 border-emerald-200 text-emerald-800",
    },
  };

  const getArchetypeClasses = (archetype: string) => {
    const colors = archetypeColors[archetype] || {
      dark: "bg-gray-500/20 border-gray-500/30 text-gray-400",
      light: "bg-gray-50 border-gray-200 text-gray-800",
    };
    return darkMode ? colors.dark : colors.light;
  };

  const isCleanSheetMarket =
    player.bookmaker_consensus?.market_type === "clean_sheet" ||
    (!!player.clean_sheet_odds && !player.scoring_odds);

  const yaraModelProbability = isCleanSheetMarket
    ? (player.clean_sheet_odds?.clean_sheet_probability ??
      player.yara_response?.yara_probability ??
      null)
    : (player.scoring_odds?.score_probability ??
      player.yara_response?.yara_probability ??
      null);

  const storyLines = player.story ? formatStoryLines(player.story) : [];

  const tabs = [
    { id: "overview" as const, label: "Risk", icon: <BarChart3 size={14} /> },
    { id: "fpl" as const, label: "FPL", icon: <Star size={14} /> },
    { id: "market" as const, label: "Odds", icon: <Coins size={14} /> },
  ];
  type TabId = (typeof tabs)[number]["id"];
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  return (
    <div
      className={`holo-card rounded-2xl overflow-hidden mx-auto ${
        darkMode ? "bg-[#141414] border border-[#1f1f1f]" : "bg-white shadow-xl"
      }`}
    >
      {/* Header with player image and club badge */}
      <div
        className={`px-4 sm:px-6 py-5 ${
          darkMode
            ? "bg-gradient-to-r from-[#1f1f1f] to-[#141414] border-b border-[#1f1f1f]"
            : "bg-gradient-to-r from-emerald-600 to-emerald-800 text-white"
        }`}
      >
        <div className="flex items-start gap-3 sm:gap-4">
          {/* Player Image */}
          {player.player_image_url && (
            <img
              src={player.player_image_url}
              alt={player.name}
              className="w-14 h-14 sm:w-16 sm:h-16 rounded-full object-cover object-top border-2 border-white/20 flex-shrink-0"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = "none";
              }}
            />
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <h2
                  className={`text-xl sm:text-2xl font-bold truncate flex items-center gap-2 ${darkMode ? "text-white" : ""}`}
                >
                  {player.name}
                  {player.is_currently_injured && (
                    <span
                      className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-red-500/20 text-red-400 border border-red-500/30 flex-shrink-0"
                      title="Injured"
                      aria-label="Injured"
                    >
                      <Ambulance size={12} />
                    </span>
                  )}
                </h2>
                <div
                  className={`flex items-center gap-2 mt-1 flex-wrap ${darkMode ? "text-gray-400" : "text-emerald-100"}`}
                >
                  {player.team_badge_url && (
                    <img
                      src={player.team_badge_url}
                      alt=""
                      className="w-4 h-4 flex-shrink-0"
                    />
                  )}
                  <span className="text-sm">{player.team}</span>
                  <span className="hidden sm:inline">·</span>
                  <span className="text-sm">{player.position}</span>
                  {player.shirt_number != null && (
                    <>
                      <span className="hidden sm:inline">·</span>
                      <span className="text-sm">#{player.shirt_number}</span>
                    </>
                  )}
                  <span className="hidden sm:inline">·</span>
                  <span className="text-sm">Age {player.age}</span>
                </div>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                {/* Share button commented out for now */}
                <RiskBadge
                  level={player.risk_level}
                  probability={player.risk_probability}
                  size="lg"
                  darkMode={darkMode}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Meter */}
      <div
        className={`px-4 sm:px-6 py-4 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}
      >
        <RiskMeter probability={player.risk_probability} darkMode={darkMode} />
        {player.risk_percentile != null && player.risk_percentile >= 0.7 && (
          <p
            className={`text-xs mt-2 ${darkMode ? "text-gray-500" : "text-gray-400"}`}
          >
            Higher risk than {Math.round(player.risk_percentile * 100)}% of
            Premier League players tracked
          </p>
        )}
      </div>

      {/* Stats Grid */}
      <div
        className={`grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 sm:p-6 ${darkMode ? "bg-[#0a0a0a]" : "bg-gray-50"}`}
      >
        <StatCard
          icon={<Activity size={18} />}
          label="Injuries"
          value={player.factors.previous_injuries.toString()}
          description="On record"
          darkMode={darkMode}
        />
        <StatCard
          icon={<Clock size={18} />}
          label="Days Lost"
          value={player.factors.total_days_lost.toString()}
          description="Career total"
          darkMode={darkMode}
        />
        <StatCard
          icon={<Calendar size={18} />}
          label="Days Since Last"
          value={player.factors.days_since_last_injury.toString()}
          description="Latest setback"
          darkMode={darkMode}
        />
        <StatCard
          icon={<TrendingUp size={18} />}
          label="Avg Layoff"
          value={player.factors.avg_days_per_injury.toFixed(1)}
          description="Days per injury"
          darkMode={darkMode}
        />
      </div>

      {/* Tab Bar */}
      <div
        className={`flex border-b ${darkMode ? "border-[#1f1f1f]" : "border-gray-200"}`}
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-4 sm:px-5 py-3 text-sm font-medium transition-colors relative ${
              activeTab === tab.id
                ? darkMode
                  ? "text-[#86efac]"
                  : "text-emerald-700"
                : darkMode
                  ? "text-gray-500 hover:text-gray-300"
                  : "text-gray-500 hover:text-gray-700"
            }`}
          >
            {tab.icon}
            {tab.label}
            {activeTab === tab.id && (
              <span
                className={`absolute bottom-0 left-0 right-0 h-0.5 ${
                  darkMode ? "bg-[#86efac]" : "bg-emerald-600"
                }`}
              />
            )}
          </button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {activeTab === "overview" && (
        <>
          {/* Archetype */}
          <div
            className={`px-4 sm:px-6 py-4 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}
          >
            <div className="flex items-start gap-3">
              <User
                className={
                  darkMode ? "text-gray-500 mt-0.5" : "text-gray-400 mt-0.5"
                }
                size={20}
              />
              <div className="flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span
                    className={`font-semibold ${darkMode ? "text-white" : "text-gray-900"}`}
                  >
                    Player Profile
                  </span>
                  <span
                    className={`px-2 py-0.5 text-xs font-medium rounded-full border ${getArchetypeClasses(player.archetype)}`}
                  >
                    {player.archetype}
                  </span>
                </div>
                <p
                  className={`text-sm mt-1 ${darkMode ? "text-gray-400" : "text-gray-600"}`}
                >
                  {player.archetype_description}
                </p>
              </div>
            </div>
          </div>

          {/* Risk Snapshot — 3 key stats matching StatCard style */}
          {player.risk_comparison && (() => {
            const rc = player.risk_comparison;
            const squadPctile = Math.round((rc.squad_rank / rc.squad_total) * 100);
            const squadLabel = rc.squad_rank <= 3
              ? `Top ${rc.squad_rank}`
              : rc.squad_rank >= rc.squad_total - 2
                ? `Bottom ${rc.squad_total - rc.squad_rank + 1}`
                : `Top ${squadPctile}%`;
            const posPctile = Math.round((rc.position_rank / rc.position_total) * 100);
            const posLabel = rc.position_rank <= 5
              ? `Top ${rc.position_rank}`
              : `Top ${posPctile}%`;
            const riskScore5 = (player.risk_probability * 5).toFixed(1);
            const positionLabel = (player.position || rc.position_group || "").toLowerCase();
            const isForwardProfile = [
              "forward",
              "striker",
              "winger",
              "attacker",
              "centre-forward",
            ].some((k) => positionLabel.includes(k));
            const isGoalkeeperProfile = [
              "goalkeeper",
              "keeper",
              "gk",
            ].some((k) => positionLabel.includes(k));
            const isDefenderProfile = [
              "def",
              "back",
            ].some((k) => positionLabel.includes(k));
            const isMidfieldProfile = [
              "midfield",
              "midfielder",
              "playmaker",
            ].some((k) => positionLabel.includes(k));

            return (
              <div className={`grid grid-cols-1 sm:grid-cols-3 gap-3 p-4 sm:p-6 ${darkMode ? "bg-[#0a0a0a]" : "bg-gray-50"}`}>
                <StatCard
                  icon={<Users size={18} />}
                  label="Team Rank"
                  value={squadLabel}
                  description={`${rc.squad_rank}/${rc.squad_total} in ${player.team}`}
                  darkMode={darkMode}
                />
                <StatCard
                  icon={
                    isGoalkeeperProfile
                      ? <KeeperGlovesIcon size={18} />
                      : isDefenderProfile
                        ? <DefenderTackleIcon size={18} />
                      : isMidfieldProfile
                        ? <MidfieldPassIcon size={18} />
                      : isForwardProfile
                        ? <StrikerShotIcon size={18} />
                        : <StrikerShotIcon size={18} />
                  }
                  label="Position Rank"
                  value={posLabel}
                  description={`${rc.position_rank}/${rc.position_total} PL ${rc.position_group}s`}
                  darkMode={darkMode}
                />
                <StatCard
                  icon={<Zap size={18} />}
                  label="Risk Score"
                  value={`${riskScore5}/5`}
                  description="2-week injury model"
                  darkMode={darkMode}
                />
              </div>
            );
          })()}

          {/* Risk Story */}
          {player.story && (
            <div
              className={`px-4 sm:px-6 py-4 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}
            >
              <div className="flex items-start gap-3">
                <FileText
                  className={
                    darkMode
                      ? "text-[#86efac] mt-0.5"
                      : "text-emerald-600 mt-0.5"
                  }
                  size={20}
                />
                <div className="flex-1">
                  <span
                    className={`font-semibold ${darkMode ? "text-white" : "text-gray-900"}`}
                  >
                    Risk Analysis
                  </span>
                  <div className="mt-2 space-y-2">
                    {storyLines.map((line, index) => (
                      <p
                        key={`${index}-${line.slice(0, 24)}`}
                        className={`text-sm leading-relaxed pl-3 border-l ${
                          index === 0
                            ? darkMode
                              ? "text-gray-200 border-[#86efac]/45"
                              : "text-gray-700 border-emerald-300"
                            : darkMode
                              ? "text-gray-300 border-white/10"
                              : "text-gray-600 border-gray-200"
                        }`}
                      >
                        {line}
                      </p>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Recommendations */}
          <div className="px-4 sm:px-6 py-4">
            <div className="flex items-center gap-2 mb-3">
              <AlertCircle className="text-amber-500" size={20} />
              <span
                className={`font-semibold ${darkMode ? "text-white" : "text-gray-900"}`}
              >
                Key Factors & Recommendations
              </span>
            </div>
            <ul className="space-y-2">
              {player.recommendations.map((rec, i) => (
                <li
                  key={i}
                  className={`flex items-start gap-2 text-sm ${darkMode ? "text-gray-400" : "text-gray-600"}`}
                >
                  <ChevronRight
                    className={`mt-0.5 flex-shrink-0 ${darkMode ? "text-[#86efac]" : "text-emerald-600"}`}
                    size={16}
                  />
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Fixture Difficulty */}
          {player.upcoming_fixtures && player.upcoming_fixtures.length > 0 && (
            <div className={`px-4 sm:px-6 py-4 ${darkMode ? "border-t border-[#1f1f1f]" : "border-t border-gray-100"}`}>
              <div className="flex items-center gap-2 mb-3">
                <Calendar className={darkMode ? "text-cyan-400" : "text-cyan-600"} size={18} />
                <span className={`font-semibold text-sm ${darkMode ? "text-white" : "text-gray-900"}`}>
                  Upcoming Fixture Difficulty
                </span>
              </div>
              <div className="flex gap-2">
                {player.upcoming_fixtures.map((fix, i) => {
                  const fdrColors: Record<number, string> = {
                    1: "bg-[#01FC7A] text-[#0a0a0a]",
                    2: "bg-[#02D76A] text-[#0a0a0a]",
                    3: "bg-[#E7E7E7] text-[#333]",
                    4: "bg-[#FF1744] text-white",
                    5: "bg-[#B71C1C] text-white",
                  };
                  return (
                    <div
                      key={i}
                      className={`flex-1 rounded-lg p-2 text-center ${fdrColors[fix.difficulty] || fdrColors[3]}`}
                    >
                      <div className="text-xs font-bold">{fix.opponent}</div>
                      <div className="text-[10px] opacity-80">{fix.is_home ? "H" : "A"}</div>
                    </div>
                  );
                })}
              </div>
              <div className={`flex justify-between mt-2 text-[10px] ${darkMode ? "text-gray-600" : "text-gray-400"}`}>
                <span>FDR: 1 (easy) → 5 (hard)</span>
              </div>
            </div>
          )}

          {/* Injury Heatmap */}
          {player.factors.previous_injuries > 0 && (
            <InjuryHeatmap player={player} darkMode={darkMode} />
          )}

        </>
      )}

      {/* ── FPL TAB ── */}
      {activeTab === "fpl" && (
        <>
          {/* Expected Points Hero Card */}
          {player.fpl_points_projection && (
            <div className={`px-4 sm:px-6 py-4 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}>
              <div className={`rounded-xl p-4 ${
                darkMode
                  ? "bg-gradient-to-br from-[#86efac]/10 to-[#86efac]/5 border border-[#86efac]/20"
                  : "bg-gradient-to-br from-emerald-50 to-emerald-100 border border-emerald-200"
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div
                      className={`h-6 px-1 rounded-full flex items-center justify-center ${
                        darkMode
                          ? "bg-white/95 ring-1 ring-white/20 shadow-[0_0_10px_rgba(255,255,255,0.08)]"
                          : "bg-white/90 ring-1 ring-emerald-200"
                      }`}
                    >
                      <img
                        src={FPL_LOGO_SOURCES[0]}
                        alt="Fantasy Premier League"
                        className="h-4 w-auto object-contain"
                        referrerPolicy="no-referrer"
                        onError={(e) => {
                          const img = e.currentTarget;
                          const currentIndex = Number(img.dataset.logoSourceIndex || "0");
                          const nextIndex = currentIndex + 1;
                          if (nextIndex < FPL_LOGO_SOURCES.length) {
                            img.dataset.logoSourceIndex = String(nextIndex);
                            img.src = FPL_LOGO_SOURCES[nextIndex];
                          }
                        }}
                      />
                    </div>
                    <span className={`text-xs uppercase tracking-wider font-semibold ${darkMode ? "text-[#86efac]/70" : "text-emerald-600"}`}>
                      Projected Points
                    </span>
                  </div>
                </div>

                <div className="flex items-baseline gap-2">
                  <span className={`text-4xl font-black ${darkMode ? "text-[#86efac]" : "text-emerald-700"}`}>
                    {player.fpl_points_projection.expected_points.toFixed(1)}
                  </span>
                  <span className={`text-sm ${darkMode ? "text-gray-500" : "text-gray-400"}`}>pts</span>
                </div>

                {player.fpl_points_projection.injury_discount_pct > 0.05 && (
                  <div className="flex items-center gap-1 mt-1">
                    <AlertCircle size={12} className="text-amber-400" />
                    <span className={`text-xs ${darkMode ? "text-amber-400" : "text-amber-600"}`}>
                      {Math.round(player.fpl_points_projection.injury_discount_pct * 100)}% injury discount applied
                    </span>
                  </div>
                )}

                <div className="grid grid-cols-3 gap-2 mt-3">
                  <div className="text-center">
                    <div className={`text-xs ${darkMode ? "text-gray-500" : "text-gray-400"}`}>Base</div>
                    <div className={`text-sm font-bold ${darkMode ? "text-white" : "text-gray-900"}`}>
                      {player.fpl_points_projection.base_points.toFixed(1)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className={`text-xs ${darkMode ? "text-gray-500" : "text-gray-400"}`}>Fixture Adj</div>
                    <div className={`text-sm font-bold ${darkMode ? "text-white" : "text-gray-900"}`}>
                      {player.fpl_points_projection.fixture_multiplier.toFixed(2)}x
                    </div>
                  </div>
                  <div className="text-center">
                    <div className={`text-xs ${darkMode ? "text-gray-500" : "text-gray-400"}`}>Risk Adj</div>
                    <div className={`text-sm font-bold ${darkMode ? "text-red-400" : "text-red-600"}`}>
                      -{Math.round(player.fpl_points_projection.injury_discount_pct * 100)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* FPL Insight */}
          {player.fpl_insight && (
            <div
              className={`px-4 sm:px-6 py-4 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}
            >
              <div
                className={`rounded-lg p-4 ${
                  darkMode
                    ? "bg-purple-500/10 border border-purple-500/30"
                    : "bg-purple-50 border border-purple-200"
                }`}
              >
                <div className="flex items-start gap-3">
                  <TrendingUp
                    size={18}
                    className={`mt-0.5 flex-shrink-0 ${darkMode ? "text-purple-300" : "text-purple-700"}`}
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <span
                        className={`font-semibold ${darkMode ? "text-purple-300" : "text-purple-800"}`}
                      >
                        FPL Insight
                      </span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full ${
                          darkMode
                            ? "bg-purple-500/30 text-purple-300"
                            : "bg-purple-200 text-purple-700"
                        }`}
                      >
                        Manager Tip
                      </span>
                    </div>
                    <p
                      className={`text-sm leading-relaxed ${darkMode ? "text-purple-200" : "text-purple-700"}`}
                    >
                      {player.fpl_insight}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* FPL Value Assessment */}
          {player.fpl_value && (
            <div
              className={`px-4 sm:px-6 py-4 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}
            >
              <div
                className={`rounded-lg p-4 ${
                  player.fpl_value.tier === "Premium"
                    ? darkMode
                      ? "bg-yellow-500/10 border border-yellow-500/30"
                      : "bg-yellow-50 border border-yellow-200"
                    : player.fpl_value.tier === "Strong"
                      ? darkMode
                        ? "bg-green-500/10 border border-green-500/30"
                        : "bg-green-50 border border-green-200"
                      : player.fpl_value.tier === "Avoid"
                        ? darkMode
                          ? "bg-red-500/10 border border-red-500/30"
                          : "bg-red-50 border border-red-200"
                        : darkMode
                          ? "bg-blue-500/10 border border-blue-500/30"
                          : "bg-blue-50 border border-blue-200"
                }`}
              >
                <div className="flex items-start gap-3">
                  <Star
                    className={`flex-shrink-0 ${
                      player.fpl_value.tier === "Premium"
                        ? "text-yellow-400"
                        : player.fpl_value.tier === "Strong"
                          ? "text-green-400"
                          : player.fpl_value.tier === "Avoid"
                            ? "text-red-400"
                            : "text-blue-400"
                    }`}
                    size={20}
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                      <span
                        className={`font-semibold ${darkMode ? "text-white" : "text-gray-900"}`}
                      >
                        FPL Value: {player.fpl_value.tier}
                      </span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full ${
                          darkMode
                            ? "bg-white/10 text-gray-300"
                            : "bg-gray-200 text-gray-700"
                        }`}
                      >
                        {player.fpl_value.price}m
                      </span>
                    </div>
                    <p
                      className={`text-sm leading-relaxed ${darkMode ? "text-gray-300" : "text-gray-600"}`}
                    >
                      {player.fpl_value.verdict}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Empty state if no FPL data */}
          {!player.fpl_insight &&
            !player.fpl_value &&
            !player.fpl_points_projection && (
              <div className="px-4 sm:px-6 py-8 text-center">
                <p
                  className={`text-sm ${darkMode ? "text-gray-500" : "text-gray-400"}`}
                >
                  No FPL data available for this player.
                </p>
              </div>
            )}
        </>
      )}

      {/* ── MARKET TAB ── */}
      {activeTab === "market" && (
        <>
          {/* Yara's Response — commented out for now */}
          {/* {player.yara_response && (
            <div className={`px-4 sm:px-6 py-5 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}>
              ... Yara's Response card ...
            </div>
          )} */}

          {/* Scoring Odds */}
          {player.scoring_odds && (
            <div
              className={`px-4 sm:px-6 py-4 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}
            >
              <div className="flex items-start gap-3">
                <Target
                  className={
                    darkMode
                      ? "text-green-400 mt-0.5"
                      : "text-green-600 mt-0.5"
                  }
                  size={20}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-3 flex-wrap">
                    <span
                      className={`font-semibold ${darkMode ? "text-white" : "text-gray-900"}`}
                    >
                      Odds to Score
                    </span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        darkMode
                          ? "bg-green-500/20 text-green-400"
                          : "bg-green-100 text-green-700"
                      }`}
                    >
                      Injury Adjusted
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-3">
                    {[
                      {
                        name: "American",
                        value: player.scoring_odds.american,
                      },
                      {
                        name: "Decimal",
                        value: player.scoring_odds.decimal.toFixed(2),
                      },
                      {
                        name: "Probability",
                        value:
                          yaraModelProbability != null
                            ? `${Math.round(yaraModelProbability * 100)}%`
                            : "N/A",
                      },
                    ].map((fmt) => (
                      <div
                        key={fmt.name}
                        className={`text-center p-2 sm:p-3 rounded-lg ${
                          darkMode
                            ? "bg-[#0a0a0a] border border-[#1f1f1f] hover:border-green-500/40"
                            : "bg-gray-50 border border-gray-200 hover:border-green-300"
                        } transition-colors cursor-default`}
                      >
                        <div
                          className={`text-[10px] uppercase tracking-wider mb-1 ${darkMode ? "text-gray-600" : "text-gray-400"}`}
                        >
                          {fmt.name}
                        </div>
                        <div
                          className={`text-lg font-bold ${darkMode ? "text-green-400" : "text-green-600"}`}
                        >
                          {fmt.value}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div
                    className={`flex items-center gap-4 text-xs mb-3 ${darkMode ? "text-gray-400" : "text-gray-600"}`}
                  >
                    <span>
                      Goals/90:{" "}
                      <span
                        className={`font-medium ${darkMode ? "text-gray-200" : "text-gray-800"}`}
                      >
                        {player.scoring_odds.goals_per_90.toFixed(2)}
                      </span>
                    </span>
                    <span>
                      Assists/90:{" "}
                      <span
                        className={`font-medium ${darkMode ? "text-gray-200" : "text-gray-800"}`}
                      >
                        {player.scoring_odds.assists_per_90.toFixed(2)}
                      </span>
                    </span>
                  </div>
                  <p
                    className={`text-sm leading-relaxed ${darkMode ? "text-gray-300" : "text-gray-600"}`}
                  >
                    {player.scoring_odds.analysis ??
                      `Yara estimates ${player.name.split(" ").pop()}\'s chance to score at ${Math.round(player.scoring_odds.score_probability * 100)}% (${player.scoring_odds.decimal.toFixed(2)}) — adjusted for injury risk and recent form.`}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Clean Sheet Odds (Defenders/GK) */}
          {player.clean_sheet_odds && (
            <div
              className={`px-4 sm:px-6 py-4 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}
            >
              <div className="flex items-start gap-3">
                <Shield
                  className={darkMode ? "text-blue-400 mt-0.5" : "text-blue-600 mt-0.5"}
                  size={20}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-3 flex-wrap">
                    <span className={`font-semibold ${darkMode ? "text-white" : "text-gray-900"}`}>
                      Clean Sheet Odds
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      darkMode ? "bg-blue-500/20 text-blue-400" : "bg-blue-100 text-blue-700"
                    }`}>
                      Injury Adjusted
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-3">
                    <OddsBox
                      label="CS Prob"
                      value={`${Math.round(player.clean_sheet_odds.clean_sheet_probability * 100)}%`}
                      highlight darkMode={darkMode} highlightColor="blue"
                    />
                    <OddsBox
                      label="Odds"
                      value={player.clean_sheet_odds.american}
                      darkMode={darkMode}
                    />
                    <OddsBox
                      label="GA/Game"
                      value={player.clean_sheet_odds.goals_conceded_per_game.toString()}
                      darkMode={darkMode}
                    />
                  </div>
                  <p className={`text-sm ${darkMode ? "text-gray-300" : "text-gray-700"}`}>
                    Yara estimates clean-sheet probability at{" "}
                    {Math.round(player.clean_sheet_odds.clean_sheet_probability * 100)}%
                    ({player.clean_sheet_odds.decimal.toFixed(2)}).
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Market Lines — real bookmaker odds */}
          {player.bookmaker_consensus && player.bookmaker_consensus.lines.length > 0 && (
            <div className={`px-4 sm:px-6 py-4 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}>
              <div className="flex items-start gap-3">
                <BarChart3 className={darkMode ? "text-cyan-400 mt-0.5" : "text-cyan-600 mt-0.5"} size={20} />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-3 flex-wrap">
                    <span className={`font-semibold ${darkMode ? "text-white" : "text-gray-900"}`}>
                      Market Lines
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      darkMode ? "bg-cyan-500/20 text-cyan-400" : "bg-cyan-100 text-cyan-700"
                    }`}>
                      {player.bookmaker_consensus.market_label}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3 mb-3">
                    <OddsBox label="Avg Odds" value={player.bookmaker_consensus.average_decimal.toFixed(2)} darkMode={darkMode} />
                    <OddsBox label="Implied Prob" value={`${Math.round(player.bookmaker_consensus.average_probability * 100)}%`} highlight darkMode={darkMode} highlightColor="blue" />
                  </div>

                  <div className="space-y-1.5">
                    {player.bookmaker_consensus.lines.map((line, i) => (
                      <div key={i} className={`flex items-center justify-between py-1.5 px-3 rounded-lg ${
                        darkMode ? "bg-[#0a0a0a] border border-[#1f1f1f]" : "bg-gray-50 border border-gray-200"
                      }`}>
                        <span className={`text-xs font-medium ${darkMode ? "text-gray-300" : "text-gray-700"}`}>
                          {line.bookmaker}
                        </span>
                        <div className="flex items-center gap-3">
                          <span className={`text-sm font-bold ${darkMode ? "text-white" : "text-gray-900"}`}>
                            {line.decimal_odds.toFixed(2)}
                          </span>
                          <span className={`text-xs ${darkMode ? "text-gray-500" : "text-gray-400"}`}>
                            {Math.round(line.implied_probability * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>

                  <p className={`text-xs mt-2 ${darkMode ? "text-gray-600" : "text-gray-400"}`}>
                    {player.bookmaker_consensus.summary_text}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Injury Odds */}
          {player.implied_odds && (
            <div
              className={`px-4 sm:px-6 py-4 ${darkMode ? "border-b border-[#1f1f1f]" : "border-b border-gray-100"}`}
            >
              <div className="flex items-start gap-3">
                <Coins
                  className={
                    darkMode
                      ? "text-amber-400 mt-0.5"
                      : "text-amber-600 mt-0.5"
                  }
                  size={20}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-3 flex-wrap">
                    <span
                      className={`font-semibold ${darkMode ? "text-white" : "text-gray-900"}`}
                    >
                      Injury Odds
                    </span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        darkMode
                          ? "bg-amber-500/20 text-amber-400"
                          : "bg-amber-100 text-amber-700"
                      }`}
                    >
                      Implied
                    </span>
                  </div>
                  <p
                    className={`text-xs mb-3 ${darkMode ? "text-gray-500" : "text-gray-500"}`}
                  >
                    Implied odds for this player getting injured in the next 2
                    weeks:
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <OddsBox
                      label="American"
                      value={player.implied_odds.american}
                      darkMode={darkMode}
                    />
                    <OddsBox
                      label="Decimal"
                      value={player.implied_odds.decimal.toString()}
                      darkMode={darkMode}
                    />
                    <OddsBox
                      label="Fractional"
                      value={player.implied_odds.fractional}
                      darkMode={darkMode}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Empty state if no market data */}
          {!player.scoring_odds && !player.implied_odds && !player.bookmaker_consensus && !player.clean_sheet_odds && (
            <div className="px-4 sm:px-6 py-8 text-center">
              <p
                className={`text-sm ${darkMode ? "text-gray-500" : "text-gray-400"}`}
              >
                No market data available for this player.
              </p>
            </div>
          )}
        </>
      )}

      {/* Share card modal commented out for now */}
    </div>
  );
}

function KeeperGlovesIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M7.2 5.2 9.1 6l1.2 2.7-.9 2.9-2.8 1.2-2.4-1.8.7-3.4 2.3-2.4Z" />
      <path d="m16.8 5.2-1.9.8-1.2 2.7.9 2.9 2.8 1.2 2.4-1.8-.7-3.4-2.3-2.4Z" />
      <circle cx="12" cy="10" r="2.1" />
    </svg>
  );
}

function DefenderTackleIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="7.8" cy="5.7" r="1.4" />
      <path d="m8 7.4 3.3 2 2.2-1.2 1.1 1.2-2.4 1.5-1.8 2.5" />
      <path d="m11.3 9.5-2.3 3.1-3.1 1.9" />
      <path d="m13.8 15.4 3.8-1.2" />
      <circle cx="19.2" cy="14.1" r="1.7" />
    </svg>
  );
}

function MidfieldPassIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 15.2c2.9-.2 4.9-1.6 6-4.6L10.8 8h2.5l.6 2.9c.2 1.3 1.2 2.2 2.5 2.4l1.5.2V17H4.2A2.2 2.2 0 0 1 2 14.8v-.5l2-.1Z" />
      <path d="M15.6 6.5c2 .2 3.3 1 4.2 2.3" />
      <path d="m18.5 5.2 1.3 3.5-3.5 1.3" />
      <circle cx="21" cy="14.2" r="1.4" />
    </svg>
  );
}

function StrikerShotIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 15.2c3-.2 5-1.7 6.2-4.8L11 8h2.5l.6 3c.2 1.3 1.2 2.3 2.5 2.5l1.4.2V17H4.2A2.2 2.2 0 0 1 2 14.8V14l2-.2Z" />
      <path d="M7 18v2" />
      <path d="M10.2 18v2" />
      <path d="M13.4 18v2" />
      <path d="M16.6 18v2" />
      <path d="M17.2 6.5h3.8v3.8" />
      <path d="m21 6.5-4.8 4.8" />
      <circle cx="21" cy="14.4" r="1.3" />
    </svg>
  );
}

function StatCard({
  icon,
  label,
  value,
  description,
  darkMode = true,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  description?: string;
  darkMode?: boolean;
}) {
  return (
    <div
      className={`relative overflow-hidden rounded-xl p-3 sm:p-3.5 text-left ${
        darkMode
          ? "bg-[#101010] border border-[#222] shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]"
          : "bg-white border border-gray-200 shadow-sm"
      }`}
    >
      <div
        className={`absolute inset-x-0 top-0 h-px ${
          darkMode
            ? "bg-gradient-to-r from-transparent via-[#86efac]/35 to-transparent"
            : "bg-gradient-to-r from-transparent via-emerald-300 to-transparent"
        }`}
      />
      <div className="flex items-center gap-2">
        <div
          className={`h-7 w-7 rounded-md flex items-center justify-center ${
            darkMode ? "bg-[#171717] text-gray-400" : "bg-gray-100 text-gray-500"
          }`}
        >
          {icon}
        </div>
        <div
          className={`text-[10px] uppercase tracking-[0.1em] font-semibold ${
            darkMode ? "text-gray-500" : "text-gray-500"
          }`}
        >
          {label}
        </div>
      </div>
      <div
        className={`mt-2 text-[23px] sm:text-[26px] leading-none font-black ${
          darkMode ? "text-white" : "text-gray-900"
        }`}
      >
        {value}
      </div>
      {description && (
        <div
          className={`text-[11px] mt-1.5 leading-snug ${
            darkMode ? "text-gray-500" : "text-gray-500"
          }`}
        >
          {description}
        </div>
      )}
    </div>
  );
}

/**
 * Anatomical silhouette + injury hotspots.
 * Anchors each injury to side-aware body points so marks align with true body areas.
 */
function InjuryHeatmap({ player, darkMode }: { player: PlayerRisk; darkMode: boolean }) {
  const records = player.injury_records || [];
  const svgIdBase = `injury-map-${player.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;

  // Anatomical body regions as polygon point data (react-body-highlighter, MIT license)
  // viewBox="0 0 100 200"
  const bodyRegions: Record<string, string[]> = {
    head: [
      "42.45 2.86 40 11.84 42.04 19.59 46.12 23.27 49.80 25.31 54.69 22.45 57.55 19.18 59.18 10.20 57.14 2.45 49.80 0",
    ],
    neck: [
      "55.51 23.67 50.61 33.47 50.61 39.18 61.63 40 70.61 44.90 69.39 36.73 63.27 35.10 58.37 30.61",
      "28.98 44.90 30.20 37.14 36.33 35.10 41.22 30.20 44.49 24.49 48.98 33.88 48.57 39.18 37.96 39.59",
    ],
    chest: [
      "51.84 41.63 51.02 55.10 57.96 57.96 67.76 55.51 70.61 47.35 62.04 41.63",
      "29.80 46.53 31.43 55.51 40.82 57.96 48.16 55.10 47.76 42.04 37.55 42.04",
    ],
    frontDeltoids: [
      "78.37 53.06 79.59 47.76 79.18 41.22 75.92 37.96 71.02 36.33 72.24 42.86 71.43 47.35",
      "28.16 47.35 21.22 53.06 20 47.76 20.41 40.82 24.49 37.14 28.57 37.14 26.94 43.27",
    ],
    biceps: [
      "16.73 68.16 17.96 71.43 22.86 66.12 28.98 53.88 27.76 49.39 20.41 55.92",
      "71.43 49.39 70.20 54.69 76.33 66.12 81.63 71.84 82.86 68.98 78.78 55.51",
    ],
    triceps: [
      "69.39 55.51 69.39 61.63 75.92 72.65 77.55 70.20 75.51 67.35",
      "22.45 69.39 29.80 55.51 29.80 60.82 22.86 73.06",
    ],
    abs: [
      "56.33 59.18 57.96 64.08 58.37 77.96 58.37 92.65 56.33 98.37 55.10 104.08 51.43 107.76 51.02 84.49 50.61 67.35 51.02 57.14",
      "43.67 58.78 48.57 57.14 48.98 67.35 48.57 84.49 48.16 107.35 44.49 103.67 40.82 91.43 40.82 78.37 41.22 64.49",
    ],
    obliques: [
      "68.57 63.27 67.35 57.14 58.78 59.59 60 64.08 60.41 83.27 65.71 78.78 66.53 69.80",
      "33.88 78.37 33.06 71.84 31.02 63.27 32.24 57.14 40.82 59.18 39.18 63.27 39.18 83.67",
    ],
    abductors: [
      "52.65 110.20 54.29 124.90 60 110.20 62.04 100 64.90 94.29 60 92.65 56.73 104.49",
      "47.76 110.61 44.90 125.31 42.04 115.92 40.41 113.06 39.59 107.35 37.96 102.45 34.69 93.88 39.59 92.24 41.63 99.18 43.67 105.31",
    ],
    quadriceps: [
      "34.69 98.78 37.14 108.16 37.14 127.76 34.29 137.14 31.02 132.65 29.39 120 28.16 111.43 29.39 100.82 32.24 94.69",
      "63.27 105.71 64.49 100 66.94 94.69 70.20 101.22 71.02 111.84 68.16 133.06 65.31 137.55 62.45 128.57 62.04 111.43",
      "38.78 129.39 38.37 112.24 41.22 118.37 44.49 129.39 42.86 135.10 40 146.12 36.33 146.53 35.51 140",
      "59.59 145.71 55.51 128.98 60.82 113.88 61.22 130.20 64.08 139.59 62.86 146.53",
      "32.65 138.37 26.53 145.71 25.71 136.73 25.71 127.35 26.94 114.29 29.39 133.47",
      "71.84 113.06 73.88 124.08 73.88 140.41 72.65 145.71 66.53 138.37 70.20 133.47",
    ],
    knees: [
      "33.88 140 34.69 143.27 35.51 147.35 36.33 151.02 35.10 156.73 29.80 156.73 27.35 152.65 27.35 147.35 30.20 144.08",
      "65.71 140 72.24 147.76 72.24 152.24 69.80 157.14 64.90 156.73 62.86 151.02",
    ],
    calves: [
      "71.43 160.41 73.47 153.47 76.73 161.22 79.59 167.76 78.37 187.76 79.59 195.51 74.69 195.51",
      "24.90 194.69 27.76 164.90 28.16 160.41 26.12 154.29 24.90 157.55 22.45 161.63 20.82 167.76 22.04 188.16 20.82 195.51",
      "72.65 195.10 69.80 159.18 65.31 158.37 64.08 162.45 64.08 165.31 65.71 177.14",
      "35.51 158.37 35.92 162.45 35.92 166.94 35.10 172.24 35.10 176.73 32.24 182.04 30.61 187.35 26.94 194.69 27.35 187.76 28.16 180.41 28.57 175.51 28.98 169.80 29.80 164.08 30.20 158.78",
    ],
    forearm: [
      "6.12 88.57 10.20 75.10 14.69 70.20 16.33 74.29 19.18 73.47 4.49 97.55 0 100",
      "84.49 69.80 83.27 73.47 80 73.06 95.10 98.37 100 100.41 93.47 89.39 89.80 76.33",
      "77.55 72.24 77.55 77.55 80.41 84.08 85.31 89.80 92.24 101.22 94.69 99.59",
      "6.94 101.22 13.47 90.61 18.78 84.08 21.63 77.14 21.22 71.84 4.90 98.78",
    ],
  };

  type Side = "left" | "right" | "center" | "unknown";
  const inferSide = (areaRaw: string): Side => {
    const area = (areaRaw || "").toLowerCase();
    if (/(^|\b)(left|lt)\b/.test(area)) return "left";
    if (/(^|\b)(right|rt)\b/.test(area)) return "right";
    return "unknown";
  };

  const partMatchers: Array<{ keys: string[]; part: string }> = [
    { keys: ["head", "concussion", "skull"], part: "head" },
    { keys: ["neck"], part: "neck" },
    { keys: ["shoulder", "clavicle", "deltoid"], part: "shoulder" },
    { keys: ["chest", "rib"], part: "chest" },
    { keys: ["back", "spine"], part: "back" },
    { keys: ["hip", "pelvis"], part: "hip" },
    { keys: ["groin", "adductor"], part: "groin" },
    { keys: ["hamstring"], part: "hamstring" },
    { keys: ["thigh", "quadriceps", "quad"], part: "thigh" },
    { keys: ["knee", "patella", "acl", "mcl", "ligament"], part: "knee" },
    { keys: ["calf"], part: "calf" },
    { keys: ["shin"], part: "shin" },
    { keys: ["ankle", "achilles"], part: "ankle" },
    { keys: ["foot", "toe"], part: "foot" },
    { keys: ["elbow"], part: "elbow" },
    { keys: ["wrist", "forearm"], part: "wrist" },
    { keys: ["hand", "finger"], part: "hand" },
    { keys: ["arm", "biceps", "triceps"], part: "arm" },
    { keys: ["illness", "virus", "flu"], part: "illness" },
    { keys: ["soft", "tissue"], part: "soft_tissue" },
    { keys: ["muscle", "strain", "knock"], part: "muscle" },
  ];

  const resolvePart = (areaRaw: string): string => {
    const area = (areaRaw || "").toLowerCase();
    if (!area || area === "unknown" || area === "n/a" || area === "-") {
      return "unknown";
    }
    for (const matcher of partMatchers) {
      if (matcher.keys.some((k) => area.includes(k))) return matcher.part;
    }
    return "torso";
  };

  type AnchorDef = {
    cy: number;
    r: number;
    centerX?: number;
    leftX?: number;
    rightX?: number;
  };
  const anchorDefs: Record<string, AnchorDef> = {
    head: { centerX: 50, cy: 12, r: 4.2 },
    neck: { centerX: 50, cy: 34, r: 3.1 },
    shoulder: { leftX: 64, rightX: 36, cy: 44, r: 4.0 },
    chest: { leftX: 60, rightX: 40, cy: 52, r: 4.6 },
    back: { leftX: 60, rightX: 40, cy: 60, r: 4.6 },
    torso: { centerX: 50, cy: 80, r: 5.0 },
    hip: { leftX: 56, rightX: 44, cy: 107, r: 4.4 },
    groin: { centerX: 50, cy: 113, r: 4.0 },
    arm: { leftX: 74, rightX: 26, cy: 65, r: 4.0 },
    elbow: { leftX: 79, rightX: 21, cy: 76, r: 3.3 },
    wrist: { leftX: 85, rightX: 15, cy: 90, r: 3.0 },
    hand: { leftX: 91, rightX: 9, cy: 100, r: 2.8 },
    thigh: { leftX: 59, rightX: 41, cy: 126, r: 4.3 },
    hamstring: { leftX: 58, rightX: 42, cy: 133, r: 4.3 },
    knee: { leftX: 67, rightX: 33, cy: 150, r: 3.8 },
    calf: { leftX: 69, rightX: 31, cy: 171, r: 3.8 },
    shin: { leftX: 67, rightX: 33, cy: 177, r: 3.6 },
    ankle: { leftX: 69, rightX: 31, cy: 188, r: 3.1 },
    foot: { leftX: 72, rightX: 28, cy: 194, r: 3.1 },
    muscle: { centerX: 50, cy: 84, r: 4.8 },
    soft_tissue: { centerX: 50, cy: 86, r: 4.8 },
    illness: { centerX: 50, cy: 52, r: 5.0 },
  };

  const hashString = (value: string): number => {
    let h = 0;
    for (let i = 0; i < value.length; i++) {
      h = (h * 31 + value.charCodeAt(i)) >>> 0;
    }
    return h;
  };

  const resolveAnchorSide = (
    part: string,
    side: Side,
    seed: string,
  ): Exclude<Side, "unknown"> => {
    const def = anchorDefs[part] || anchorDefs.torso;
    const hasPair = def.leftX != null && def.rightX != null;
    if (side === "left" || side === "right" || side === "center") {
      return side;
    }
    if (hasPair) {
      return hashString(seed) % 2 === 0 ? "left" : "right";
    }
    return "center";
  };

  const getAnchor = (part: string, side: Exclude<Side, "unknown">) => {
    const def = anchorDefs[part] || anchorDefs.torso;
    let cx = def.centerX ?? 50;
    if (side !== "center" && def.leftX != null && def.rightX != null) {
      cx = side === "left" ? def.leftX : def.rightX;
    } else if (def.leftX != null && def.rightX != null && def.centerX == null) {
      cx = (def.leftX + def.rightX) / 2;
    }
    const cy = def.cy;
    const r = def.r;
    return { cx, cy, r, key: `${part}-${side}-${cx}-${cy}` };
  };

  const severityPalette = {
    mild: {
      fill: "rgba(52,211,153,0.82)",
      glow: "rgba(52,211,153,0.36)",
      ring: darkMode ? "rgba(167,243,208,0.92)" : "rgba(5,150,105,0.82)",
    },
    moderate: {
      fill: "rgba(245,158,11,0.84)",
      glow: "rgba(245,158,11,0.36)",
      ring: darkMode ? "rgba(253,230,138,0.95)" : "rgba(180,83,9,0.84)",
    },
    severe: {
      fill: "rgba(239,68,68,0.86)",
      glow: "rgba(239,68,68,0.44)",
      ring: darkMode ? "rgba(252,165,165,0.95)" : "rgba(185,28,28,0.85)",
    },
  } as const;

  // Aggregate injury data per hotspot anchor
  const anchorData: Record<
    string,
    {
      count: number;
      totalDays: number;
      maxSeverityDays: number;
      severeHits: number;
      moderateHits: number;
      types: string[];
      anchor: { cx: number; cy: number; r: number; key: string };
      part: string;
      side: Exclude<Side, "unknown">;
    }
  > = {};

  for (const injury of records) {
    const part = resolvePart(injury.body_area);
    if (part === "unknown") continue;
    const inferredSide = inferSide(injury.body_area);
    const anchorSide = resolveAnchorSide(
      part,
      inferredSide,
      `${injury.body_area || ""}|${injury.injury_type || ""}|${injury.date || ""}|${injury.injury_raw || ""}`,
    );
    const anchor = getAnchor(part, anchorSide);
    if (!anchorData[anchor.key]) {
      anchorData[anchor.key] = {
        count: 0,
        totalDays: 0,
        maxSeverityDays: 0,
        severeHits: 0,
        moderateHits: 0,
        types: [],
        anchor,
        part,
        side: anchorSide,
      };
    }
    const days = Number.isFinite(injury.severity_days)
      ? Math.max(0, injury.severity_days)
      : 0;
    anchorData[anchor.key].count += 1;
    anchorData[anchor.key].totalDays += days;
    anchorData[anchor.key].maxSeverityDays = Math.max(
      anchorData[anchor.key].maxSeverityDays,
      days,
    );
    if (days >= 60) anchorData[anchor.key].severeHits += 1;
    if (days >= 21) anchorData[anchor.key].moderateHits += 1;
    const label = injury.body_area
      ? injury.body_area.charAt(0).toUpperCase() + injury.body_area.slice(1)
      : "Unknown";
    if (!anchorData[anchor.key].types.includes(label)) {
      anchorData[anchor.key].types.push(label);
    }
  }

  const classifyAnchor = (z: {
    count: number;
    totalDays: number;
    maxSeverityDays: number;
    severeHits: number;
    moderateHits: number;
  }): "mild" | "moderate" | "severe" => {
    const avg = z.totalDays / Math.max(1, z.count);
    if (z.severeHits >= 2 || avg >= 75 || z.maxSeverityDays >= 120) return "severe";
    if (z.severeHits >= 1 || z.moderateHits >= 1 || avg >= 24) return "moderate";
    return "mild";
  };

  const hotspots = Object.values(anchorData)
    .map((z) => ({
      ...z,
      tier: classifyAnchor(z),
      activity: Math.min(1, (z.totalDays + z.count * 12) / 240),
    }))
    .sort((a, b) => {
      if (b.maxSeverityDays !== a.maxSeverityDays) {
        return b.maxSeverityDays - a.maxSeverityDays;
      }
      return b.totalDays - a.totalDays;
    });

  if (hotspots.length === 0) return null;

  return (
    <div
      className={`px-4 sm:px-6 py-4 ${darkMode ? "border-t border-[#1f1f1f]" : "border-t border-gray-100"}`}
    >
      <div className="flex items-center gap-2 mb-3">
        <Activity
          className={darkMode ? "text-purple-400" : "text-purple-600"}
          size={18}
        />
        <span
          className={`font-semibold text-sm ${darkMode ? "text-white" : "text-gray-900"}`}
        >
          Injury Map
        </span>
        <span className={`text-[10px] ${darkMode ? "text-gray-600" : "text-gray-400"}`}>
          {records.length} {records.length === 1 ? "injury" : "injuries"}
        </span>
      </div>

      <div className="flex flex-col sm:flex-row gap-3 sm:items-start">
        <div className="w-full sm:w-[178px] flex-shrink-0 flex flex-col items-center sm:items-start">
          <svg viewBox="0 0 100 200" className="w-[162px] sm:w-[170px] h-auto max-w-full">
            <defs>
              <radialGradient id={`${svgIdBase}-halo`} cx="50%" cy="34%" r="62%">
                <stop
                  offset="0%"
                  stopColor={
                    darkMode ? "rgba(134,239,172,0.12)" : "rgba(16,185,129,0.12)"
                  }
                />
                <stop
                  offset="72%"
                  stopColor={
                    darkMode ? "rgba(134,239,172,0.02)" : "rgba(16,185,129,0.03)"
                  }
                />
                <stop offset="100%" stopColor="transparent" />
              </radialGradient>
              {hotspots.map((h, i) => {
                const palette = severityPalette[h.tier];
                return (
                  <radialGradient key={i} id={`${svgIdBase}-hot-${i}`}>
                    <stop offset="0%" stopColor={palette.fill} />
                    <stop offset="52%" stopColor={palette.glow} />
                    <stop offset="100%" stopColor="transparent" />
                  </radialGradient>
                );
              })}
            </defs>

            <rect x="0" y="0" width="100" height="200" fill={`url(#${svgIdBase}-halo)`} />

            {/* Anatomical silhouette */}
            {Object.entries(bodyRegions).map(([region, polygons]) => {
              const baseFill = darkMode ? "rgba(220,252,231,0.12)" : "rgba(16,185,129,0.12)";
              const baseStroke = darkMode ? "rgba(167,243,208,0.22)" : "rgba(5,150,105,0.18)";
              return polygons.map((points, i) => (
                <polygon
                  key={`${region}-${i}`}
                  points={points}
                  fill={baseFill}
                  stroke={baseStroke}
                  strokeWidth="0.3"
                  strokeLinejoin="round"
                  opacity={0.7}
                />
              ));
            })}

            {/* Pulsing hotspot indicators on injuries */}
            {hotspots.map((h, i) => {
              const palette = severityPalette[h.tier];
              const core = Math.max(2.3, h.anchor.r * (0.75 + h.activity * 0.4));
              const auraStart = core * 1.8;
              const auraEnd = core * 2.4;
              return (
                <g key={`${h.anchor.key}-${i}`}>
                  <circle
                    cx={h.anchor.cx}
                    cy={h.anchor.cy}
                    r={auraStart}
                    fill={`url(#${svgIdBase}-hot-${i})`}
                    opacity={0.28 + h.activity * 0.34}
                  >
                    <animate
                      attributeName="r"
                      values={`${auraStart};${auraEnd};${auraStart}`}
                      dur={`${2.2 + (i % 4) * 0.35}s`}
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values={`${0.24 + h.activity * 0.2};${0.58 + h.activity * 0.22};${0.24 + h.activity * 0.2}`}
                      dur={`${2.2 + (i % 4) * 0.35}s`}
                      repeatCount="indefinite"
                    />
                  </circle>
                  <circle cx={h.anchor.cx} cy={h.anchor.cy} r={core * 0.52} fill={palette.fill} />
                  <circle
                    cx={h.anchor.cx}
                    cy={h.anchor.cy}
                    r={core * 0.76}
                    fill="none"
                    stroke={palette.ring}
                    strokeWidth="0.8"
                    opacity={0.9}
                  />
                </g>
              );
            })}
          </svg>

          <div
            className={`mt-1.5 flex items-center justify-between text-[9px] ${
              darkMode ? "text-gray-500" : "text-gray-400"
            }`}
          >
            <span className="inline-flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              Mild
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
              Moderate
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
              Severe
            </span>
          </div>
        </div>

        {/* Timeline */}
        {(() => {
          const timelineRecords = (player.injury_records || [])
            .filter((r) => r.date)
            .sort((a, b) => new Date(b.date!).getTime() - new Date(a.date!).getTime())
            .slice(0, 8);

          if (timelineRecords.length < 2) return null;

          const tlDotColor = (days: number) =>
            days >= 60
              ? "bg-red-500 shadow-red-500/40"
              : days >= 21
                ? "bg-amber-500 shadow-amber-500/40"
                : "bg-emerald-500 shadow-emerald-500/40";
          const tlSevText = (days: number) =>
            days >= 60
              ? darkMode
                ? "text-red-400"
                : "text-red-600"
              : days >= 21
                ? darkMode
                  ? "text-amber-400"
                  : "text-amber-600"
                : darkMode
                  ? "text-emerald-400"
                  : "text-emerald-600";
          const tlLineColor = darkMode ? "bg-[#1f1f1f]" : "bg-gray-200";
          const allRecords = (player.injury_records || []).filter((r) => r.date);

          return (
            <div className="w-full flex-1 min-w-0 overflow-hidden pt-1">
              {timelineRecords.map((r, i) => {
                const d = r.date ? new Date(r.date) : null;
                const month = d ? d.toLocaleDateString("en-GB", { month: "short" }) : "";
                const year = d ? d.getFullYear().toString() : "";
                const isLast = i === timelineRecords.length - 1;
                return (
                  <div key={i} className="flex gap-2" style={{ minHeight: 40 }}>
                    <div className="w-9 flex-shrink-0 text-right pt-0.5">
                      <div
                        className={`text-[10px] font-medium leading-tight ${
                          darkMode ? "text-gray-500" : "text-gray-400"
                        }`}
                      >
                        {month}
                      </div>
                      <div
                        className={`text-[10px] leading-tight ${
                          darkMode ? "text-gray-600" : "text-gray-400"
                        }`}
                      >
                        {year}
                      </div>
                    </div>
                    <div className="flex flex-col items-center flex-shrink-0" style={{ width: 14 }}>
                      <div
                        className={`w-2.5 h-2.5 rounded-full flex-shrink-0 shadow-sm ${tlDotColor(r.severity_days)}`}
                      />
                      {!isLast && <div className={`w-px flex-1 ${tlLineColor}`} />}
                    </div>
                    <div className="flex-1 pb-3 min-w-0">
                      <div
                        className={`text-xs font-medium ${darkMode ? "text-gray-200" : "text-gray-800"}`}
                      >
                        {r.injury_type}
                      </div>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className={`text-[11px] font-bold ${tlSevText(r.severity_days)}`}>
                          {r.severity_days}d out
                        </span>
                        <span className={`text-[10px] ${darkMode ? "text-gray-600" : "text-gray-400"}`}>
                          {r.body_area}
                        </span>
                        {r.games_missed > 0 && (
                          <span className={`text-[10px] ${darkMode ? "text-gray-600" : "text-gray-400"}`}>
                            · {r.games_missed} games missed
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              {allRecords.length > 8 && (
                <div className="flex gap-2">
                  <div className="w-9" />
                  <div className="flex flex-col items-center" style={{ width: 14 }}>
                    <div
                      className={`w-1.5 h-1.5 rounded-full ${
                        darkMode ? "bg-gray-700" : "bg-gray-300"
                      }`}
                    />
                  </div>
                  <p className={`text-[10px] ${darkMode ? "text-gray-600" : "text-gray-400"}`}>
                    +{allRecords.length - 8} earlier
                  </p>
                </div>
              )}
            </div>
          );
        })()}
      </div>
    </div>
  );
}

function OddsBox({
  label,
  value,
  darkMode = true,
  highlight = false,
  highlightColor = "green",
}: {
  label: string;
  value: string;
  darkMode?: boolean;
  highlight?: boolean;
  highlightColor?: string;
}) {
  const colorMap: Record<string, string> = {
    green: darkMode ? "text-green-400" : "text-green-600",
    blue: darkMode ? "text-blue-400" : "text-blue-600",
  };
  return (
    <div
      className={`text-center p-2 sm:p-3 rounded-lg ${
        darkMode
          ? "bg-[#0a0a0a] border border-[#1f1f1f]"
          : "bg-gray-50 border border-gray-200"
      }`}
    >
      <div
        className={`text-base sm:text-lg font-bold ${
          highlight
            ? colorMap[highlightColor] || colorMap.green
            : darkMode
              ? "text-white"
              : "text-gray-900"
        }`}
      >
        {value}
      </div>
      <div
        className={`text-xs ${darkMode ? "text-gray-500" : "text-gray-500"}`}
      >
        {label}
      </div>
    </div>
  );
}
