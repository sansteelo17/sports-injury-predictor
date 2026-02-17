"use client";

import { useState, useEffect } from "react";
import {
  getTeams,
  getTeamOverview,
  getPlayerRisk,
  getFPLInsights,
  getStandingsSummary,
  getTeamBadges,
} from "@/lib/api";
import {
  TeamOverview as TeamOverviewType,
  PlayerRisk,
  FPLInsights as FPLInsightsType,
  StandingsSummary,
} from "@/types/api";
import { TeamSelector } from "@/components/TeamSelector";
import { TeamOverview } from "@/components/TeamOverview";
import { PlayerList } from "@/components/PlayerList";
import { PlayerCard } from "@/components/PlayerCard";
import { LabNotes } from "@/components/LabNotes";
import { FPLInsights } from "@/components/FPLInsights";
import { StandingsCards } from "@/components/StandingsCards";
import { Activity, Shield, Info, Moon, Sun, Zap, Microscope } from "lucide-react";

export default function Home() {
  const [teams, setTeams] = useState<string[]>([]);
  const [selectedTeam, setSelectedTeam] = useState("");
  const [teamOverview, setTeamOverview] = useState<TeamOverviewType | null>(
    null,
  );
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null);
  const [playerRisk, setPlayerRisk] = useState<PlayerRisk | null>(null);
  const [fplInsights, setFplInsights] = useState<FPLInsightsType | null>(null);
  const [standings, setStandings] = useState<StandingsSummary | null>(null);
  const [teamBadges, setTeamBadges] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(true);
  const [view, setView] = useState<'overview' | 'lab'>('overview');

  // Load teams, FPL data, and badges on mount
  useEffect(() => {
    getTeams()
      .then(setTeams)
      .catch(() => setError("Failed to load teams. Is the API running?"));

    getFPLInsights()
      .then(setFplInsights)
      .catch(() => console.log("FPL insights unavailable"));

    getTeamBadges()
      .then(setTeamBadges)
      .catch(() => console.log("Team badges unavailable"));
  }, []);

  // Load team overview when team selected
  useEffect(() => {
    if (!selectedTeam) {
      setTeamOverview(null);
      setSelectedPlayer(null);
      setPlayerRisk(null);
      return;
    }

    setLoading(true);
    setError(null);

    Promise.all([
      getTeamOverview(selectedTeam),
      getStandingsSummary(selectedTeam).catch(() => null),
    ])
      .then(([teamData, standingsData]) => {
        setTeamOverview(teamData);
        setStandings(standingsData);
        setSelectedPlayer(null);
        setPlayerRisk(null);
      })
      .catch(() => setError("Failed to load team data"))
      .finally(() => setLoading(false));
  }, [selectedTeam]);

  // Load player risk when player selected
  useEffect(() => {
    if (!selectedPlayer) {
      setPlayerRisk(null);
      return;
    }

    setLoading(true);
    setView('overview');
    getPlayerRisk(selectedPlayer)
      .then(setPlayerRisk)
      .catch(() => setError("Failed to load player data"))
      .finally(() => setLoading(false));
  }, [selectedPlayer]);

  const bgClass = darkMode ? "bg-[#0a0a0a]" : "bg-gray-50";
  const textClass = darkMode ? "text-white" : "text-gray-900";
  const mutedClass = darkMode ? "text-gray-500" : "text-gray-500";
  const cardClass = darkMode
    ? "bg-[#141414] border-[#1f1f1f]"
    : "bg-white border-gray-200";

  return (
    <div className={`app-shell min-h-screen flex flex-col ${bgClass} ${textClass} ${darkMode ? "matrix-theme" : "light-theme"}`}>
      {/* Header */}
      <header
        className={`holo-header mobile-square-header ${darkMode ? "bg-[#141414] border-b border-[#1f1f1f]" : "bg-white border-b border-gray-200"} py-3 sm:py-4 px-3 sm:px-4`}
      >
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="relative">
              <Activity
                size={28}
                className={darkMode ? "text-[#86efac]" : "text-emerald-600"}
              />
              <Zap
                size={12}
                className={`absolute -top-1 -right-1 ${darkMode ? "text-[#86efac]" : "text-emerald-600"}`}
              />
            </div>
            <div>
              <h1 className="text-lg sm:text-xl font-bold tracking-tight">
                Yara
                <span
                  className={darkMode ? "text-[#86efac]" : "text-emerald-600"}
                >
                  Sports
                </span>
              </h1>
              <p className={`text-[10px] sm:text-xs leading-tight max-w-[170px] sm:max-w-none ${mutedClass}`}>
                Risk-aware match intelligence for fans and analysts.
              </p>
            </div>
          </div>

          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 rounded-lg transition-colors ${
              darkMode
                ? "bg-[#1f1f1f] hover:bg-[#86efac]/20"
                : "bg-gray-100 hover:bg-gray-200"
            }`}
          >
            {darkMode ? (
              <Sun size={18} className="text-[#86efac]" />
            ) : (
              <Moon size={18} />
            )}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-3 sm:px-4 py-4 sm:py-8">
        {/* League Notice */}
        <div
          className={`holo-panel mobile-square-league ${darkMode ? "bg-[#86efac]/10 border-[#86efac]/30" : "bg-emerald-50 border-emerald-200"} border rounded-xl p-3 sm:p-4 mb-4 sm:mb-6`}
        >
          <div className="flex items-start gap-2 sm:gap-3">
            <Info
              className={`flex-shrink-0 ${darkMode ? "text-[#86efac]" : "text-emerald-600"}`}
              size={16}
            />
            <div
              className={`text-xs sm:text-sm ${darkMode ? "text-gray-300" : "text-gray-700"}`}
            >
              <strong
                className={darkMode ? "text-[#86efac]" : "text-emerald-600"}
              >
                Currently covering Premier League.
              </strong>{" "}
              <span className="hidden sm:inline">
                More leagues coming soon. Our ML model analyzes injury history,
                recovery patterns, and severity to predict injury risk over the
                next 2 weeks.
              </span>
              <span className="sm:hidden">
                ML-powered injury risk predictions.
              </span>
            </div>
          </div>
        </div>

        {/* Team Selector */}
        <div className="mb-4 sm:mb-6">
          <label className={`block text-sm font-medium mb-2 ${mutedClass}`}>
            Select Team
          </label>
          <TeamSelector
            teams={teams}
            selectedTeam={selectedTeam}
            onSelectTeam={setSelectedTeam}
            darkMode={darkMode}
            teamBadges={teamBadges}
          />
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 sm:p-4 mb-4 sm:mb-6 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-8 sm:py-12">
            <div
              className={`animate-spin rounded-full h-8 w-8 border-b-2 ${darkMode ? "border-[#86efac]" : "border-emerald-600"}`}
            ></div>
          </div>
        )}

        {/* Content Grid */}
        {teamOverview && !loading && (
          <div className="grid lg:grid-cols-3 gap-4 sm:gap-6">
            {/* Left Column - Team Overview & Player List */}
            <div className={`lg:col-span-1 space-y-4 sm:space-y-6 ${playerRisk ? 'order-2 lg:order-1' : ''}`}>
              <TeamOverview team={teamOverview} darkMode={darkMode} />

              {standings && (
                <StandingsCards standings={standings} darkMode={darkMode} teamBadges={teamBadges} />
              )}

              {fplInsights && (
                <FPLInsights
                  insights={fplInsights}
                  selectedTeam={selectedTeam}
                  darkMode={darkMode}
                />
              )}

              <div className={`holo-panel ${cardClass} border rounded-xl p-3 sm:p-4`}>
                <h3
                  className={`font-semibold mb-3 flex items-center gap-2 text-sm sm:text-base ${textClass}`}
                >
                  <Shield
                    size={16}
                    className={darkMode ? "text-[#86efac]" : "text-emerald-600"}
                  />
                  Squad
                </h3>
                <div className="max-h-[50vh] sm:max-h-96 overflow-y-auto">
                  <PlayerList
                    players={teamOverview.players}
                    onSelectPlayer={setSelectedPlayer}
                    selectedPlayer={selectedPlayer || undefined}
                    darkMode={darkMode}
                  />
                </div>
              </div>
            </div>

            {/* Right Column - Player Card / Lab Notes */}
            <div className={`lg:col-span-2 ${playerRisk ? 'order-1 lg:order-2' : ''}`}>
              {playerRisk ? (
                <div className="space-y-4">
                  {/* View Toggle */}
                  <div className={`holo-panel flex gap-1 p-1 rounded-xl ${darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-gray-100'}`}>
                    <button
                      onClick={() => setView('overview')}
                      className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                        view === 'overview'
                          ? darkMode ? 'bg-[#1f1f1f] text-white' : 'bg-white text-gray-900 shadow-sm'
                          : darkMode ? 'text-gray-500 hover:text-gray-300' : 'text-gray-500 hover:text-gray-700'
                      }`}
                    >
                      <Shield size={14} />
                      Overview
                    </button>
                    <button
                      onClick={() => setView('lab')}
                      className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
                        view === 'lab'
                          ? darkMode ? 'bg-[#1f1f1f] text-white' : 'bg-white text-gray-900 shadow-sm'
                          : darkMode ? 'text-gray-500 hover:text-gray-300' : 'text-gray-500 hover:text-gray-700'
                      }`}
                    >
                      <Microscope size={14} />
                      <span className="leading-tight">
                        Yara&apos;s Lab Notes{" "}
                        <span className="text-[10px] sm:text-xs opacity-80">(for builders)</span>
                      </span>
                    </button>
                  </div>

                  {/* Content */}
                  {view === 'overview' ? (
                    <PlayerCard player={playerRisk} darkMode={darkMode} />
                  ) : (
                    <LabNotes player={playerRisk} darkMode={darkMode} />
                  )}
                </div>
              ) : (
                <div
                  className={`holo-panel ${cardClass} border rounded-2xl p-8 sm:p-12 text-center`}
                >
                  <Shield
                    size={48}
                    className={`mx-auto mb-4 ${darkMode ? "text-[#1f1f1f]" : "text-gray-300"}`}
                  />
                  <h3 className={`text-lg font-medium mb-2 ${textClass}`}>
                    Select a Player
                  </h3>
                  <p className={`text-sm ${mutedClass}`}>
                    Click on any player to view their injury risk analysis
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!selectedTeam && !loading && (
          <div className="text-center py-10 sm:py-16">
            <div className="relative inline-block mb-4 sm:mb-6">
              <Activity
                size={48}
                className={darkMode ? "text-[#1f1f1f]" : "text-gray-200"}
              />
              <Zap
                size={20}
                className={`absolute -top-2 -right-2 animate-pulse ${darkMode ? "text-[#86efac]" : "text-emerald-600"}`}
              />
            </div>
            <h2 className={`text-lg sm:text-xl font-semibold mb-2 ${textClass}`}>
              Welcome to YaraSports
            </h2>
            <p className={`text-sm max-w-md mx-auto ${mutedClass}`}>
              Select a Premier League team to view squad injury risk analysis
              and player predictions.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer
        className={`holo-header mobile-square-header ${darkMode ? "bg-[#141414] border-t border-[#1f1f1f]" : "bg-gray-100 border-t border-gray-200"} py-4 sm:py-6`}
      >
        <div className="max-w-6xl mx-auto px-3 sm:px-4 text-center">
          <p
            className={`text-xs sm:text-sm ${darkMode ? "text-gray-600" : "text-gray-500"}`}
          >
            Predictions estimate injury probability over the next 2 weeks.
            Powered by ensemble ML models. For educational purposes only.
          </p>
        </div>
      </footer>
    </div>
  );
}
