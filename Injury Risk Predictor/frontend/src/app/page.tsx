"use client";

import { useState, useEffect } from "react";
import {
  getTeams,
  getTeamOverview,
  getPlayerRisk,
  getFPLInsights,
  getStandingsSummary,
  getTeamBadges,
  getFPLSquad,
  getLaLigaStandings,
} from "@/lib/api";
import {
  TeamOverview as TeamOverviewType,
  PlayerRisk,
  FPLInsights as FPLInsightsType,
  StandingsSummary,
  FPLSquadSync,
  LaLigaStandingRow,
} from "@/types/api";
import { TeamSelector } from "@/components/TeamSelector";
import { TeamOverview } from "@/components/TeamOverview";
import { PlayerList } from "@/components/PlayerList";
import { PlayerCard } from "@/components/PlayerCard";
import { LabNotes } from "@/components/LabNotes";
import { FPLInsights } from "@/components/FPLInsights";
import { StandingsCards } from "@/components/StandingsCards";
import { LaLigaStandingsCards } from "@/components/LaLigaStandingsCards";
import { FPLSquadInput } from "@/components/FPLSquadInput";
import { FPLSquadView } from "@/components/FPLSquadView";
import {
  trackYaraFplSquadSyncCompleted,
  trackYaraLabNotesOpened,
  trackYaraPlayerSelected,
  trackYaraTeamSelected,
} from "@/analytics/sygna";
import {
  Activity,
  Shield,
  Info,
  Moon,
  Sun,
  Zap,
  Microscope,
  Users,
  Search,
} from "lucide-react";

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
  const [laLigaStandings, setLaLigaStandings] = useState<LaLigaStandingRow[]>([]);
  const [teamBadges, setTeamBadges] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(true);
  const [view, setView] = useState<"overview" | "lab">("overview");

  // League
  const [league, setLeague] = useState<"Premier League" | "La Liga">("Premier League");

  // Squad sync state
  const [mode, setMode] = useState<"browse" | "squad">("browse");
  const [fplSquad, setFplSquad] = useState<FPLSquadSync | null>(null);
  const [squadLoading, setSquadLoading] = useState(false);
  const [squadError, setSquadError] = useState<string | null>(null);
  const [lastSyncedId, setLastSyncedId] = useState<string | null>(null);

  // Reload teams when league changes
  useEffect(() => {
    setSelectedTeam("");
    setTeamOverview(null);
    setSelectedPlayer(null);
    setPlayerRisk(null);
    setStandings(null);
    setLaLigaStandings([]);
    getTeams(league)
      .then(setTeams)
      .catch(() => setError("Failed to load teams. Is the API running?"));
  }, [league]);

  // Load FPL data only when EPL/FPL-specific context is active.
  useEffect(() => {
    const needsFplInsights = league === "Premier League" || mode === "squad";
    if (!needsFplInsights || fplInsights) return;

    getFPLInsights()
      .then(setFplInsights)
      .catch(() => console.log("FPL insights unavailable"));
  }, [league, mode, fplInsights]);

  // Team badges are league-agnostic and cheap enough to load once.
  useEffect(() => {
    getTeamBadges()
      .then(setTeamBadges)
      .catch(() => console.log("Team badges unavailable"));
  }, []);

  // Load La Liga standings once per league switch, not on every team click.
  useEffect(() => {
    if (league !== "La Liga") return;

    getLaLigaStandings()
      .then(setLaLigaStandings)
      .catch(() => console.log("La Liga standings unavailable"));
  }, [league]);

  const handleLeagueSwitch = (l: "Premier League" | "La Liga") => {
    if (l !== league) {
      setLeague(l);
      // FPL squad mode is EPL-only; switch to browse when moving to La Liga
      if (l === "La Liga" && mode === "squad") setMode("browse");
    }
  };

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

    const isLaLiga = league === "La Liga";

    const standingsPromise = isLaLiga
      ? Promise.resolve(null)
      : getStandingsSummary(selectedTeam).catch(() => null);

    Promise.all([getTeamOverview(selectedTeam), standingsPromise])
      .then(([teamData, standingsData]) => {
        setTeamOverview(teamData);
        if (isLaLiga) {
          setStandings(null);
        } else {
          setStandings(standingsData as StandingsSummary | null);
          setLaLigaStandings([]);
        }
        setSelectedPlayer(null);
        setPlayerRisk(null);
      })
      .catch(() => setError("Failed to load team data"))
      .finally(() => setLoading(false));
  }, [selectedTeam, league]);

  // Load player risk when player selected
  useEffect(() => {
    if (!selectedPlayer) {
      setPlayerRisk(null);
      return;
    }

    setLoading(true);
    setView("overview");
    getPlayerRisk(selectedPlayer)
      .then(setPlayerRisk)
      .catch(() => setError("Failed to load player data"))
      .finally(() => setLoading(false));
  }, [selectedPlayer]);

  // Squad sync handler
  const handleSquadSync = (teamId: string) => {
    setSquadLoading(true);
    setSquadError(null);
    getFPLSquad(teamId)
      .then((data) => {
        setFplSquad(data);
        setLastSyncedId(teamId);
        setSelectedPlayer(null);
        setPlayerRisk(null);
        trackYaraFplSquadSyncCompleted(teamId, data.players.length);
      })
      .catch((err) => {
        const raw = String(err?.message || "");
        const msg = raw.includes("404")
          ? "FPL team not found. Check your Team ID."
          : raw.includes("temporarily unavailable")
            ? raw.replace(/^API error:\s*\d+\s*/, "")
            : raw.includes("503")
              ? "FPL servers are currently unavailable. Try again shortly."
              : raw || "Failed to sync squad. Try again.";
        setSquadError(msg);
      })
      .finally(() => setSquadLoading(false));
  };

  const handleTeamSelected = (team: string) => {
    setSelectedTeam(team);
    trackYaraTeamSelected(team);
  };

  const handlePlayerSelected = (playerName: string) => {
    setSelectedPlayer(playerName);
    trackYaraPlayerSelected(playerName, {
      team: mode === "browse" ? selectedTeam : fplSquad?.entry.team_name || null,
      mode,
    });
  };

  const handleModeSwitch = (newMode: "browse" | "squad") => {
    setMode(newMode);
    setSelectedPlayer(null);
    setPlayerRisk(null);
    setError(null);
  };

  const bgClass = darkMode ? "bg-[#0a0a0a]" : "bg-gray-50";
  const textClass = darkMode ? "text-white" : "text-gray-900";
  const mutedClass = darkMode ? "text-gray-500" : "text-gray-500";
  const cardClass = darkMode
    ? "bg-[#141414] border-[#1f1f1f]"
    : "bg-white border-gray-200";

  const hasContent = mode === "browse" ? !!teamOverview : !!fplSquad;

  return (
    <div
      className={`app-shell min-h-screen flex flex-col ${bgClass} ${textClass} ${darkMode ? "matrix-theme" : "light-theme"}`}
    >
      {/* Header */}
      <header
        className={`holo-header mobile-square-header ${darkMode ? "bg-[#141414] border-b border-[#1f1f1f]" : "bg-white border-b border-gray-200"} py-3 sm:py-4 px-3 sm:px-4`}
      >
        <div className="max-w-6xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
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
            <div className="min-w-0">
              <h1 className="text-lg sm:text-xl font-bold tracking-tight truncate">
                Yara
                <span
                  className={darkMode ? "text-[#86efac]" : "text-emerald-600"}
                >
                  Sports
                </span>
              </h1>
              <p
                className={`text-[10px] sm:text-xs leading-tight max-w-[170px] sm:max-w-none ${mutedClass}`}
              >
                Risk-aware match intelligence for fans and analysts.
              </p>
            </div>
          </div>

          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`shrink-0 p-2 rounded-lg transition-colors ${
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
              className={`text-xs sm:text-sm break-words ${darkMode ? "text-gray-300" : "text-gray-700"}`}
            >
              <strong
                className={darkMode ? "text-[#86efac]" : "text-emerald-600"}
              >
                Premier League and La Liga are now live
              </strong>{" "}
              — more leagues coming.{" "}
              <span className="hidden sm:inline">
                Yara explains why a player might get injured in the next 2 weeks
                by blending injury history, workload patterns, and fixture
                context into risk narratives.
              </span>
              <span className="sm:hidden">
                Yara explains why a player might get injured in the next 2 weeks
                by blending injury history, workload patterns, and fixture
                context into risk narratives.
              </span>
            </div>
          </div>
        </div>

        {/* League Switcher + Mode Toggle + Input */}
        <div className="mb-4 sm:mb-6">
          {/* League switcher — always visible first */}
          <div className="flex gap-1 mb-3">
            {(["Premier League", "La Liga"] as const).map((l) => (
              <button
                key={l}
                onClick={() => handleLeagueSwitch(l)}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                  league === l
                    ? darkMode
                      ? "bg-[#86efac]/15 text-[#86efac] border border-[#86efac]/30"
                      : "bg-emerald-50 text-emerald-700 border border-emerald-300"
                    : darkMode
                      ? "text-gray-500 hover:text-gray-300 border border-transparent"
                      : "text-gray-500 hover:text-gray-700 border border-transparent"
                }`}
              >
                {l === "Premier League" ? "🏴󠁧󠁢󠁥󠁮󠁧󠁿 EPL" : "🇪🇸 La Liga"}
              </button>
            ))}
          </div>

          {/* Mode tabs — Browse Teams always; My FPL Squad only for EPL */}
          <div className="flex gap-1 mb-3">
            <button
              onClick={() => handleModeSwitch("browse")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
                mode === "browse"
                  ? darkMode
                    ? "bg-[#86efac]/15 text-[#86efac] border border-[#86efac]/30"
                    : "bg-emerald-50 text-emerald-700 border border-emerald-300"
                  : darkMode
                    ? "text-gray-500 hover:text-gray-300"
                    : "text-gray-500 hover:text-gray-700"
              }`}
            >
              <Search size={13} />
              Browse Teams
            </button>
            {league === "Premier League" && (
              <button
                onClick={() => handleModeSwitch("squad")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
                  mode === "squad"
                    ? darkMode
                      ? "bg-[#86efac]/15 text-[#86efac] border border-[#86efac]/30"
                      : "bg-emerald-50 text-emerald-700 border border-emerald-300"
                    : darkMode
                      ? "text-gray-500 hover:text-gray-300"
                      : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <Users size={13} />
                My FPL Squad
              </button>
            )}
          </div>

          {/* Conditional input */}
          {mode === "squad" && league === "Premier League" ? (
            <FPLSquadInput
              onSync={handleSquadSync}
              loading={squadLoading}
              error={squadError}
              darkMode={darkMode}
            />
          ) : (
            <>
              <label className={`block text-sm font-medium mb-2 ${mutedClass}`}>
                Select Team
              </label>
              <TeamSelector
                teams={teams}
                selectedTeam={selectedTeam}
                onSelectTeam={handleTeamSelected}
                darkMode={darkMode}
                teamBadges={teamBadges}
              />
            </>
          )}
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
        {hasContent && !loading && (
          <div className="grid lg:grid-cols-3 gap-4 sm:gap-6">
            {/* Left Column */}
            <div
              className={`lg:col-span-1 min-w-0 space-y-4 sm:space-y-6 ${playerRisk ? "order-2 lg:order-1" : ""}`}
            >
              {mode === "squad" && fplSquad ? (
                <FPLSquadView
                  squad={fplSquad}
                  onSelectPlayer={handlePlayerSelected}
                  selectedPlayer={selectedPlayer || undefined}
                  onRefresh={() => lastSyncedId && handleSquadSync(lastSyncedId)}
                  darkMode={darkMode}
                />
              ) : teamOverview ? (
                <>
                  <TeamOverview team={teamOverview} darkMode={darkMode} />

                  {standings && league === "Premier League" && (
                    <StandingsCards
                      standings={standings}
                      darkMode={darkMode}
                      teamBadges={teamBadges}
                    />
                  )}

                  {laLigaStandings.length > 0 && league === "La Liga" && (
                    <LaLigaStandingsCards
                      standings={laLigaStandings}
                      selectedTeam={selectedTeam}
                      darkMode={darkMode}
                    />
                  )}

                  {fplInsights && league === "Premier League" && (
                    <FPLInsights
                      insights={fplInsights}
                      selectedTeam={selectedTeam}
                      darkMode={darkMode}
                    />
                  )}

                  <div
                    className={`holo-panel ${cardClass} border rounded-xl p-3 sm:p-4`}
                  >
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
                        onSelectPlayer={handlePlayerSelected}
                        selectedPlayer={selectedPlayer || undefined}
                        darkMode={darkMode}
                      />
                    </div>
                  </div>
                </>
              ) : null}
            </div>

            {/* Right Column - Player Card / Lab Notes */}
            <div
              className={`lg:col-span-2 min-w-0 ${playerRisk ? "order-1 lg:order-2" : ""}`}
            >
              {playerRisk ? (
                <div className="space-y-4">
                  {/* View Toggle */}
                  <div
                    className={`holo-panel flex gap-1 p-1 rounded-xl ${darkMode ? "bg-[#141414] border border-[#1f1f1f]" : "bg-gray-100"}`}
                  >
                    <button
                      onClick={() => setView("overview")}
                      className={`flex-1 flex items-center justify-center gap-2 px-2.5 sm:px-4 py-2 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
                        view === "overview"
                          ? darkMode
                            ? "bg-[#1f1f1f] text-white"
                            : "bg-white text-gray-900 shadow-sm"
                          : darkMode
                            ? "text-gray-500 hover:text-gray-300"
                            : "text-gray-500 hover:text-gray-700"
                      }`}
                    >
                      <Shield size={14} />
                      Overview
                    </button>
                    <button
                      onClick={() => {
                        setView("lab");
                        if (playerRisk) {
                          trackYaraLabNotesOpened(playerRisk.name, {
                            team: playerRisk.team,
                            mode,
                          });
                        }
                      }}
                      className={`flex-1 flex items-center justify-center gap-2 px-2.5 sm:px-4 py-2 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
                        view === "lab"
                          ? darkMode
                            ? "bg-[#1f1f1f] text-white"
                            : "bg-white text-gray-900 shadow-sm"
                          : darkMode
                            ? "text-gray-500 hover:text-gray-300"
                            : "text-gray-500 hover:text-gray-700"
                      }`}
                    >
                      <Microscope size={14} />
                      <span className="leading-tight">
                        <span>Yara&apos;s Lab Notes</span>
                        <span className="block sm:inline text-[10px] sm:text-xs opacity-80 sm:ml-1">
                          (for builders)
                        </span>
                      </span>
                    </button>
                  </div>

                  {/* Content */}
                  {view === "overview" ? (
                    <PlayerCard player={playerRisk} darkMode={darkMode} />
                  ) : (
                    <LabNotes player={playerRisk} darkMode={darkMode} />
                  )}
                </div>
              ) : (
                <div className={`holo-panel ${cardClass} border rounded-2xl p-6 sm:p-12 text-center`}>
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
        {!hasContent && !loading && (
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
            <h2
              className={`text-lg sm:text-xl font-semibold mb-2 ${textClass}`}
            >
              Welcome to YaraSports
            </h2>
            <p className={`text-sm max-w-md mx-auto ${mutedClass}`}>
              {mode === "squad"
                ? "Enter your FPL Team ID above to see injury risk for your squad."
                : "Select a Premier League team to view squad injury risk analysis and player predictions."}
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
            className={`text-xs sm:text-sm max-w-3xl mx-auto ${darkMode ? "text-gray-600" : "text-gray-500"}`}
          >
            Predictions estimate injury probability over the next 2 weeks.
            Powered by ensemble ML models. For educational purposes only.
          </p>
        </div>
      </footer>
    </div>
  );
}
