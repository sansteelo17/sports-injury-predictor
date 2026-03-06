"use client";

import { useState } from "react";
import { PlayerRisk } from "@/types/api";
import { RiskBadge } from "./RiskBadge";
import { RiskMeter } from "./RiskMeter";
import { ShareCard } from "./ShareCard";
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
  Share2,
} from "lucide-react";

interface PlayerCardProps {
  player: PlayerRisk;
  darkMode?: boolean;
}

const FPL_LOGO_SOURCES = [
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
    { id: "overview" as const, label: "Risk", icon: <Shield size={14} /> },
    { id: "fpl" as const, label: "FPL", icon: <Star size={14} /> },
    { id: "market" as const, label: "Odds", icon: <Coins size={14} /> },
  ];
  type TabId = (typeof tabs)[number]["id"];
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [showShareCard, setShowShareCard] = useState(false);

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
                    <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30 uppercase tracking-wide flex-shrink-0">
                      <Ambulance size={12} />
                      Injured
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
                  <span className="hidden sm:inline">·</span>
                  <span className="text-sm">Age {player.age}</span>
                </div>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <button
                  onClick={() => setShowShareCard(true)}
                  className={`p-1.5 rounded-lg transition-colors ${
                    darkMode
                      ? "hover:bg-white/10 text-gray-400 hover:text-white"
                      : "hover:bg-gray-100 text-gray-400 hover:text-gray-700"
                  }`}
                  title="Share card"
                >
                  <Share2 size={16} />
                </button>
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
          label="Previous Injuries"
          value={player.factors.previous_injuries.toString()}
          darkMode={darkMode}
        />
        <StatCard
          icon={<Clock size={18} />}
          label="Days Lost"
          value={player.factors.total_days_lost.toString()}
          darkMode={darkMode}
        />
        <StatCard
          icon={<Calendar size={18} />}
          label="Days Since Last"
          value={player.factors.days_since_last_injury.toString()}
          darkMode={darkMode}
        />
        <StatCard
          icon={<TrendingUp size={18} />}
          label="Avg Days/Injury"
          value={player.factors.avg_days_per_injury.toFixed(1)}
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
              <Shield
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
            const riskScore = Math.round(player.risk_probability * 100);

            return (
              <div className={`grid grid-cols-3 gap-3 p-4 sm:p-6 ${darkMode ? "bg-[#0a0a0a]" : "bg-gray-50"}`}>
                <StatCard
                  icon={<Shield size={18} />}
                  label={`in ${player.team}`}
                  value={squadLabel}
                  darkMode={darkMode}
                />
                <StatCard
                  icon={<Target size={18} />}
                  label={`PL ${rc.position_group}s`}
                  value={posLabel}
                  darkMode={darkMode}
                />
                <StatCard
                  icon={<Activity size={18} />}
                  label="Risk Score"
                  value={riskScore.toString()}
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
                    <img
                      src={FPL_LOGO_SOURCES[0]}
                      alt="Fantasy Premier League"
                      className="h-5 w-auto object-contain"
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
                    <span className={`text-xs uppercase tracking-wider font-semibold ${darkMode ? "text-[#86efac]/70" : "text-emerald-600"}`}>
                      Expected Points
                    </span>
                  </div>
                  <span className={`text-[10px] px-2 py-0.5 rounded-full ${
                    player.fpl_points_projection.confidence === "high"
                      ? darkMode ? "bg-[#86efac]/20 text-[#86efac]" : "bg-emerald-200 text-emerald-800"
                      : darkMode ? "bg-amber-500/20 text-amber-400" : "bg-amber-200 text-amber-800"
                  }`}>
                    {player.fpl_points_projection.confidence} confidence
                  </span>
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
                    <div className={`text-xs ${darkMode ? "text-gray-500" : "text-gray-400"}`}>Fixture</div>
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
                  <div className="grid grid-cols-3 gap-2 mb-3">
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
                  <div className="grid grid-cols-3 gap-3 mb-3">
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
                  <div className="grid grid-cols-3 gap-3">
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

      {/* Share Card Modal */}
      {showShareCard && (
        <ShareCard
          player={player}
          darkMode={darkMode}
          onClose={() => setShowShareCard(false)}
        />
      )}
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
    <div
      className={`rounded-lg p-2 sm:p-3 text-center ${
        darkMode ? "bg-[#141414] border border-[#1f1f1f]" : "bg-white shadow-sm"
      }`}
    >
      <div
        className={`flex justify-center mb-1 ${darkMode ? "text-gray-500" : "text-gray-400"}`}
      >
        {icon}
      </div>
      <div
        className={`text-lg sm:text-xl font-bold ${darkMode ? "text-white" : "text-gray-900"}`}
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

/**
 * Body silhouette with glowing injury hotspots.
 * Maps real injury records to body regions with pulsing radial glows.
 */
function InjuryHeatmap({ player, darkMode }: { player: PlayerRisk; darkMode: boolean }) {
  const records = player.injury_records || [];
  const svgIdBase = `injury-map-${player.name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;

  // Aggregate by body zone — coordinates mapped to viewBox 0 0 724 1448
  type Zone = { cx: number; cy: number; r: number };
  const bodyZones: Record<string, Zone> = {
    head: { cx: 362, cy: 185, r: 55 },
    neck: { cx: 362, cy: 235, r: 35 },
    shoulder: { cx: 362, cy: 305, r: 90 },
    arm: { cx: 195, cy: 430, r: 50 },
    elbow: { cx: 170, cy: 510, r: 40 },
    wrist: { cx: 135, cy: 660, r: 30 },
    hand: { cx: 125, cy: 730, r: 30 },
    back: { cx: 362, cy: 400, r: 75 },
    torso: { cx: 362, cy: 400, r: 75 },
    hip: { cx: 362, cy: 530, r: 65 },
    groin: { cx: 362, cy: 580, r: 55 },
    thigh: { cx: 300, cy: 700, r: 65 },
    quadriceps: { cx: 425, cy: 700, r: 65 },
    hamstring: { cx: 300, cy: 760, r: 65 },
    knee: { cx: 310, cy: 870, r: 50 },
    calf: { cx: 310, cy: 990, r: 50 },
    shin: { cx: 415, cy: 990, r: 50 },
    ankle: { cx: 305, cy: 1130, r: 40 },
    achilles: { cx: 420, cy: 1130, r: 40 },
    foot: { cx: 362, cy: 1280, r: 45 },
    muscle: { cx: 440, cy: 500, r: 70 },
    soft_tissue: { cx: 285, cy: 500, r: 70 },
    illness: { cx: 362, cy: 350, r: 65 },
  };

  const resolveZone = (area: string): Zone => {
    const a = area.toLowerCase().replace(/[^a-z_]/g, "");
    for (const [key, zone] of Object.entries(bodyZones)) {
      if (a.includes(key) || key.includes(a)) return zone;
    }
    return bodyZones.muscle; // fallback
  };

  const severityTier = (days: number): "mild" | "moderate" | "severe" => {
    if (days >= 60) return "severe";
    if (days >= 21) return "moderate";
    return "mild";
  };

  const severityPalette = {
    mild: {
      fill: "rgba(52,211,153,0.68)",
      glow: "rgba(52,211,153,0.3)",
      ring: darkMode ? "rgba(167,243,208,0.85)" : "rgba(5,150,105,0.7)",
    },
    moderate: {
      fill: "rgba(245,158,11,0.72)",
      glow: "rgba(245,158,11,0.32)",
      ring: darkMode ? "rgba(253,230,138,0.9)" : "rgba(180,83,9,0.72)",
    },
    severe: {
      fill: "rgba(239,68,68,0.75)",
      glow: "rgba(239,68,68,0.38)",
      ring: darkMode ? "rgba(252,165,165,0.9)" : "rgba(185,28,28,0.75)",
    },
  } as const;

  // Aggregate injuries per zone
  const zoneData: Record<
    string,
    {
      count: number;
      totalDays: number;
      maxSeverityDays: number;
      zone: Zone;
      types: string[];
      mildCount: number;
      moderateCount: number;
      severeCount: number;
    }
  > = {};
  for (const r of records) {
    const z = resolveZone(r.body_area);
    const key = `${z.cx}-${z.cy}`;
    if (!zoneData[key]) {
      zoneData[key] = {
        count: 0,
        totalDays: 0,
        maxSeverityDays: 0,
        zone: z,
        types: [],
        mildCount: 0,
        moderateCount: 0,
        severeCount: 0,
      };
    }
    const days = Number.isFinite(r.severity_days) ? Math.max(0, r.severity_days) : 0;
    const tier = severityTier(days);
    zoneData[key].count++;
    zoneData[key].totalDays += days;
    zoneData[key].maxSeverityDays = Math.max(zoneData[key].maxSeverityDays, days);
    if (tier === "severe") zoneData[key].severeCount += 1;
    else if (tier === "moderate") zoneData[key].moderateCount += 1;
    else zoneData[key].mildCount += 1;
    const label = r.body_area.charAt(0).toUpperCase() + r.body_area.slice(1);
    if (!zoneData[key].types.includes(label)) zoneData[key].types.push(label);
  }

  const hotspots = Object.values(zoneData)
    .map((zone) => ({
      ...zone,
      tier: severityTier(zone.maxSeverityDays),
      activity: Math.min(1, (zone.totalDays + zone.count * 8) / 170),
    }))
    .sort((a, b) => {
      if (b.maxSeverityDays !== a.maxSeverityDays) return b.maxSeverityDays - a.maxSeverityDays;
      return b.totalDays - a.totalDays;
    });

  if (hotspots.length === 0) return null;

  return (
    <div className={`px-4 sm:px-6 py-4 ${darkMode ? "border-t border-[#1f1f1f]" : "border-t border-gray-100"}`}>
      <div className="flex items-center gap-2 mb-3">
        <Activity className={darkMode ? "text-purple-400" : "text-purple-600"} size={18} />
        <span className={`font-semibold text-sm ${darkMode ? "text-white" : "text-gray-900"}`}>
          Injury Map
        </span>
        <span className={`text-[10px] ${darkMode ? "text-gray-600" : "text-gray-400"}`}>
          {records.length} {records.length === 1 ? "injury" : "injuries"}
        </span>
      </div>

      <div className="flex gap-3 items-start">
        {/* Body silhouette with upgraded skeleton + hotspot pulses */}
        <div className="flex-shrink-0">
          <svg viewBox="0 0 724 1448" width="160" height="312">
            <defs>
              <radialGradient id={`${svgIdBase}-backdrop`} cx="50%" cy="38%" r="58%">
                <stop offset="0%" stopColor={darkMode ? "rgba(134,239,172,0.10)" : "rgba(16,185,129,0.10)"} />
                <stop offset="65%" stopColor={darkMode ? "rgba(134,239,172,0.03)" : "rgba(16,185,129,0.04)"} />
                <stop offset="100%" stopColor="transparent" />
              </radialGradient>
              <linearGradient id={`${svgIdBase}-bone-grad`} x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor={darkMode ? "rgba(217,249,231,0.85)" : "rgba(22,101,52,0.85)"} />
                <stop offset="100%" stopColor={darkMode ? "rgba(134,239,172,0.42)" : "rgba(34,197,94,0.45)"} />
              </linearGradient>
              <linearGradient id={`${svgIdBase}-scan-line`} x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="transparent" />
                <stop offset="40%" stopColor={darkMode ? "rgba(134,239,172,0.04)" : "rgba(16,185,129,0.06)"} />
                <stop offset="52%" stopColor={darkMode ? "rgba(167,243,208,0.16)" : "rgba(5,150,105,0.18)"} />
                <stop offset="64%" stopColor={darkMode ? "rgba(134,239,172,0.04)" : "rgba(16,185,129,0.06)"} />
                <stop offset="100%" stopColor="transparent" />
              </linearGradient>
              <filter id={`${svgIdBase}-bone-glow`} x="-40%" y="-40%" width="180%" height="180%">
                <feGaussianBlur stdDeviation="2.8" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <pattern id={`${svgIdBase}-injury-grid`} width="44" height="44" patternUnits="userSpaceOnUse">
                <path
                  d="M 44 0 L 0 0 0 44"
                  fill="none"
                  stroke={darkMode ? "rgba(134,239,172,0.06)" : "rgba(16,185,129,0.08)"}
                  strokeWidth="1"
                />
              </pattern>
              {hotspots.map((h, i) => {
                const palette = severityPalette[h.tier];
                return (
                  <radialGradient key={i} id={`${svgIdBase}-glow-${i}`}>
                    <stop offset="0%" stopColor={palette.fill} />
                    <stop offset="45%" stopColor={palette.glow} />
                    <stop offset="100%" stopColor="transparent" />
                  </radialGradient>
                );
              })}
            </defs>

            <rect x="0" y="0" width="724" height="1448" fill={`url(#${svgIdBase}-backdrop)`} />
            <rect x="0" y="0" width="724" height="1448" fill={`url(#${svgIdBase}-injury-grid)`} opacity={0.72} />
            <rect x="0" y="-360" width="724" height="360" fill={`url(#${svgIdBase}-scan-line)`} opacity={0.85}>
              <animate attributeName="y" values="-360;1448" dur="7.5s" repeatCount="indefinite" />
            </rect>

            <g stroke={darkMode ? "rgba(134,239,172,0.08)" : "rgba(15,23,42,0.05)"} strokeWidth="1">
              <line x1="90" y1="210" x2="634" y2="210" />
              <line x1="90" y1="470" x2="634" y2="470" />
              <line x1="90" y1="760" x2="634" y2="760" />
              <line x1="90" y1="1040" x2="634" y2="1040" />
              <line x1="180" y1="90" x2="180" y2="1320" />
              <line x1="544" y1="90" x2="544" y2="1320" />
            </g>

            <g
              fill="none"
              stroke={darkMode ? "rgba(251,191,36,0.22)" : "rgba(146,64,14,0.2)"}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity={0.8}
            >
              <animate attributeName="opacity" values="0.62;0.86;0.62" dur="8s" repeatCount="indefinite" />
              <circle cx="580" cy="112" r="36" />
              <circle cx="580" cy="112" r="16" />
              <line x1="580" y1="56" x2="580" y2="38" />
              <line x1="580" y1="168" x2="580" y2="186" />
              <line x1="524" y1="112" x2="506" y2="112" />
              <line x1="636" y1="112" x2="654" y2="112" />
              <line x1="540" y1="72" x2="526" y2="58" />
              <line x1="620" y1="152" x2="634" y2="166" />
              <line x1="620" y1="72" x2="634" y2="58" />
              <line x1="540" y1="152" x2="526" y2="166" />
              <path d="M78 330 Q132 300 168 344 Q126 380 176 422 Q124 470 188 516" />
              <path d="M646 332 Q592 302 556 346 Q598 382 548 424 Q600 468 536 514" />
              <path d="M112 1182 L188 1116 L262 1182 Z" />
              <path d="M462 1206 L544 1138 L622 1206 Z" />
              <path d="M90 1248 L182 1184 L274 1248 Z" />
              <path d="M438 1274 L538 1200 L640 1274 Z" />
              <g transform="translate(122 612)">
                <circle cx="0" cy="0" r="12" />
                <line x1="0" y1="12" x2="0" y2="56" />
                <line x1="-22" y1="34" x2="22" y2="34" />
                <line x1="0" y1="56" x2="-18" y2="88" />
                <line x1="0" y1="56" x2="18" y2="88" />
              </g>
              <g transform="translate(602 622)">
                <circle cx="0" cy="0" r="12" />
                <line x1="0" y1="12" x2="0" y2="56" />
                <line x1="-22" y1="34" x2="22" y2="34" />
                <line x1="0" y1="56" x2="-18" y2="88" />
                <line x1="0" y1="56" x2="18" y2="88" />
              </g>
            </g>

            <path
              d="M362 120c-92 0-150 78-150 168 0 88 36 168 36 236 0 58-35 118-35 202 0 80 40 136 92 166v162c0 58-26 130-26 190 0 63 40 116 83 116s83-53 83-116c0-60-26-132-26-190V892c52-30 92-86 92-166 0-84-35-144-35-202 0-68 36-148 36-236 0-90-58-168-150-168Z"
              fill={darkMode ? "rgba(134,239,172,0.04)" : "rgba(16,185,129,0.05)"}
            />

            <g
              fill="none"
              stroke={`url(#${svgIdBase}-bone-grad)`}
              strokeLinecap="round"
              strokeLinejoin="round"
              filter={`url(#${svgIdBase}-bone-glow)`}
            >
              <animate attributeName="opacity" values="0.88;1;0.88" dur="4.8s" repeatCount="indefinite" />
              <g strokeWidth="3.1">
                <ellipse cx="362" cy="108" rx="56" ry="62" />
                <path d="M332 142 Q362 158 392 142 Q392 174 362 184 Q332 174 332 142Z" />
                <circle cx="344" cy="102" r="8" />
                <circle cx="380" cy="102" r="8" />
                <path d="M362 170 L362 584" />
                <path d="M362 202 Q304 198 258 232" />
                <path d="M362 202 Q420 198 466 232" />
                <path d="M362 676 Q306 676 272 646 Q274 608 300 594 L362 620 L424 594 Q450 608 452 646 Q418 676 362 676Z" />
              </g>

              <g strokeWidth="2.3">
                {[246, 276, 308, 340, 372, 404].map((y) => (
                  <g key={y}>
                    <path d={`M362 ${y} Q ${362 - 88} ${y + 18} ${362 - 58} ${y + 42}`} />
                    <path d={`M362 ${y} Q ${362 + 88} ${y + 18} ${362 + 58} ${y + 42}`} />
                  </g>
                ))}
                {[202, 230, 258, 286, 314, 342, 370, 398, 426, 454, 482, 510, 538, 566].map((y) => (
                  <circle key={`v-${y}`} cx="362" cy={y} r="5.2" />
                ))}
              </g>

              <g strokeWidth="3.4">
                <line x1="286" y1="254" x2="224" y2="424" />
                <line x1="224" y1="424" x2="178" y2="622" />
                <line x1="438" y1="254" x2="500" y2="424" />
                <line x1="500" y1="424" x2="546" y2="622" />

                <line x1="328" y1="686" x2="302" y2="908" />
                <line x1="302" y1="924" x2="286" y2="1188" />
                <line x1="396" y1="686" x2="422" y2="908" />
                <line x1="422" y1="924" x2="438" y2="1188" />
              </g>

              <g strokeWidth="2.4">
                <ellipse cx="222" cy="432" rx="18" ry="24" />
                <ellipse cx="502" cy="432" rx="18" ry="24" />
                <ellipse cx="302" cy="914" rx="18" ry="22" />
                <ellipse cx="422" cy="914" rx="18" ry="22" />
                <circle cx="286" cy="1188" r="15" />
                <circle cx="438" cy="1188" r="15" />
                <path d="M286 1202 L252 1234 L242 1268 L268 1274 L304 1268 L304 1202" />
                <path d="M438 1202 L472 1234 L482 1268 L456 1274 L420 1268 L420 1202" />
              </g>
            </g>

            {/* Hotspot glows */}
            {hotspots.map((h, i) => {
              const palette = severityPalette[h.tier];
              const radius = h.zone.r * (0.68 + h.activity * 1.05);
              const pulseMax = radius * 1.92;
              const pulseMin = radius * 1.58;
              return (
                <g key={i}>
                  <circle
                    cx={h.zone.cx}
                    cy={h.zone.cy}
                    r={pulseMin}
                    fill={`url(#${svgIdBase}-glow-${i})`}
                    opacity={0.34 + h.activity * 0.34}
                  >
                    <animate
                      attributeName="r"
                      values={`${pulseMin};${pulseMax};${pulseMin}`}
                      dur={`${2.2 + (i % 3) * 0.35}s`}
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values={`${0.3 + h.activity * 0.16};${0.6 + h.activity * 0.22};${0.3 + h.activity * 0.16}`}
                      dur={`${2.2 + (i % 3) * 0.35}s`}
                      repeatCount="indefinite"
                    />
                  </circle>
                  <circle
                    cx={h.zone.cx}
                    cy={h.zone.cy}
                    r={radius * 0.45}
                    fill={palette.fill}
                    opacity={0.96}
                  />
                  <circle
                    cx={h.zone.cx}
                    cy={h.zone.cy}
                    r={radius * 0.62}
                    fill="none"
                    stroke={palette.ring}
                    strokeWidth="2"
                    opacity={0.78}
                  />
                </g>
              );
            })}
          </svg>

          <div className={`mt-1.5 flex items-center justify-between text-[9px] ${darkMode ? "text-gray-500" : "text-gray-400"}`}>
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

        {/* Injury timeline — git-log style beside the skeleton */}
        {(() => {
          const timelineRecords = (player.injury_records || [])
            .filter((r) => r.date)
            .sort((a, b) => new Date(b.date!).getTime() - new Date(a.date!).getTime())
            .slice(0, 8);

          if (timelineRecords.length < 2) return null;

          const tlDotColor = (days: number) =>
            days >= 60 ? "bg-red-500 shadow-red-500/40" : days >= 21 ? "bg-amber-500 shadow-amber-500/40" : "bg-emerald-500 shadow-emerald-500/40";
          const tlSevText = (days: number) =>
            days >= 60 ? (darkMode ? "text-red-400" : "text-red-600")
            : days >= 21 ? (darkMode ? "text-amber-400" : "text-amber-600")
            : (darkMode ? "text-emerald-400" : "text-emerald-600");
          const tlLineColor = darkMode ? "bg-[#1f1f1f]" : "bg-gray-200";
          const allRecords = (player.injury_records || []).filter((r) => r.date);

          return (
            <div className="flex-1 min-w-0 overflow-hidden pt-1">
              {timelineRecords.map((r, i) => {
                const d = r.date ? new Date(r.date) : null;
                const month = d ? d.toLocaleDateString("en-GB", { month: "short" }) : "";
                const year = d ? d.getFullYear().toString() : "";
                const isLast = i === timelineRecords.length - 1;
                return (
                  <div key={i} className="flex gap-2" style={{ minHeight: 40 }}>
                    <div className="w-9 flex-shrink-0 text-right pt-0.5">
                      <div className={`text-[10px] font-medium leading-tight ${darkMode ? "text-gray-500" : "text-gray-400"}`}>
                        {month}
                      </div>
                      <div className={`text-[10px] leading-tight ${darkMode ? "text-gray-600" : "text-gray-400"}`}>
                        {year}
                      </div>
                    </div>
                    <div className="flex flex-col items-center flex-shrink-0" style={{ width: 14 }}>
                      <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 shadow-sm ${tlDotColor(r.severity_days)}`} />
                      {!isLast && <div className={`w-px flex-1 ${tlLineColor}`} />}
                    </div>
                    <div className="flex-1 pb-3 min-w-0">
                      <div className={`text-xs font-medium ${darkMode ? "text-gray-200" : "text-gray-800"}`}>
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
                            · {r.games_missed}gm
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
                    <div className={`w-1.5 h-1.5 rounded-full ${darkMode ? "bg-gray-700" : "bg-gray-300"}`} />
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
