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
  getWinnerOdds,
} from "@/lib/api";
import {
  TeamOverview as TeamOverviewType,
  PlayerRisk,
  FPLInsights as FPLInsightsType,
  StandingsSummary,
  FPLSquadSync,
  LaLigaStandingRow,
  WinnerOdds,
} from "@/types/api";
import { TeamSelector } from "@/components/TeamSelector";
import { TeamOverview } from "@/components/TeamOverview";
import { PlayerList } from "@/components/PlayerList";
import { PlayerCard } from "@/components/PlayerCard";
import { LabNotes } from "@/components/LabNotes";
import { FPLInsights } from "@/components/FPLInsights";
import { StandingsCards } from "@/components/StandingsCards";
import { LaLigaStandingsCards } from "@/components/LaLigaStandingsCards";
import { WinnerOddsCard } from "@/components/WinnerOddsCard";
import { FPLSquadInput } from "@/components/FPLSquadInput";
import { FPLSquadView } from "@/components/FPLSquadView";
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
  ChevronDown,
} from "lucide-react";

type LeagueStatus = "active" | "upcoming" | "offseason";

// Single source for the switcher: display order is the in-season default; the
// component re-sorts active/upcoming first at render. ``id`` matches the
// CompetitionChoice union used everywhere else.
const LEAGUE_META: { id: string; label: string; flag: string }[] = [
  { id: "FIFA World Cup 2026", label: "World Cup 2026", flag: "🏆" },
  { id: "Premier League", label: "Premier League", flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿" },
  { id: "La Liga", label: "La Liga", flag: "🇪🇸" },
  { id: "Bundesliga", label: "Bundesliga", flag: "🇩🇪" },
  { id: "Serie A", label: "Serie A", flag: "🇮🇹" },
  { id: "Ligue 1", label: "Ligue 1", flag: "🇫🇷" },
  { id: "Champions League", label: "Champions League", flag: "🇪🇺" },
];

// Coarse season calendar — the client has no per-league fixture feed, so this
// only orders the switcher and tags off-season competitions; it never gates
// anything. Adjust the windows (or replace with a backend status) as needed.
function leagueStatus(id: string, now: Date): LeagueStatus {
  if (id === "FIFA World Cup 2026") {
    const start = new Date("2026-06-11T00:00:00Z");
    const end = new Date("2026-07-20T00:00:00Z");
    if (now < start) return "upcoming";
    if (now <= end) return "active";
    return "offseason";
  }
  const m = now.getUTCMonth(); // 0=Jan … 11=Dec
  if (id === "Champions League") {
    // League phase Sep, knockouts to late May; off June–August.
    return m >= 8 || m <= 4 ? "active" : "offseason";
  }
  // European club leagues run ~August through May; June/July is the off-season.
  return m >= 7 || m <= 4 ? "active" : "offseason";
}

const STATUS_RANK: Record<LeagueStatus, number> = { active: 0, upcoming: 1, offseason: 2 };
const STATUS_LABEL: Record<LeagueStatus, string> = { active: "live", upcoming: "soon", offseason: "off-season" };

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
  const [winnerOdds, setWinnerOdds] = useState<WinnerOdds | null>(null);
  const [teamBadges, setTeamBadges] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(true);
  const [view, setView] = useState<"overview" | "lab">("overview");
  const [leagueMenuOpen, setLeagueMenuOpen] = useState(false);

  // Competition (international tournament leads while the WC is active).
  type CompetitionChoice = "FIFA World Cup 2026" | "Premier League" | "La Liga" | "Bundesliga" | "Serie A" | "Ligue 1" | "Champions League";
  const [league, setLeague] = useState<CompetitionChoice>("FIFA World Cup 2026");
  const isInternational = league === "FIFA World Cup 2026";
  const competitionId =
    league === "Premier League"
      ? "premier-league"
      : league === "La Liga"
        ? "la-liga"
        : league === "Bundesliga"
          ? "bundesliga"
          : league === "Serie A"
            ? "serie-a"
            : league === "Ligue 1"
              ? "ligue-1"
              : league === "Champions League"
                ? "champions-league"
                : "world-cup-2026";

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
    getTeams(undefined, competitionId)
      .then(setTeams)
      .catch(() => setError("Failed to load teams. Is the API running?"));
  }, [league, competitionId]);

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

  // Tournament-winner odds for the World Cup view (cleared otherwise).
  useEffect(() => {
    if (!isInternational) {
      setWinnerOdds(null);
      return;
    }
    getWinnerOdds(competitionId)
      .then(setWinnerOdds)
      .catch(() => setWinnerOdds(null));
  }, [isInternational, competitionId]);

  const handleLeagueSwitch = (l: CompetitionChoice) => {
    if (l !== league) {
      setLeague(l);
      // FPL squad mode is EPL-only; switch to browse when leaving EPL.
      if (l !== "Premier League" && mode === "squad") setMode("browse");
    }
  };

  // Load team overview when team selected
  useEffect(() => {
    // Clear the previous team's overview, player, and standings immediately so
    // nothing stale lingers while the new team loads (e.g. switching from an
    // England WC view to Barcelona). Without this, the old content stays on
    // screen until the new fetch resolves.
    setTeamOverview(null);
    setSelectedPlayer(null);
    setPlayerRisk(null);
    setStandings(null);

    if (!selectedTeam) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    const skipFplStandings = league === "La Liga" || league === "FIFA World Cup 2026";

    const standingsPromise = skipFplStandings
      ? Promise.resolve(null)
      : getStandingsSummary(selectedTeam).catch(() => null);

    Promise.all([getTeamOverview(selectedTeam, competitionId), standingsPromise])
      .then(([teamData, standingsData]) => {
        if (cancelled) return; // a newer team was selected; drop this response
        setTeamOverview(teamData);
        if (!skipFplStandings) {
          setStandings(standingsData as StandingsSummary | null);
        }
      })
      .catch(() => {
        if (!cancelled) setError("Failed to load team data");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedTeam, league]);

  // Load player risk when player selected
  useEffect(() => {
    if (!selectedPlayer) {
      setPlayerRisk(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setView("overview");
    setPlayerRisk(null); // drop the previous player's card before the new one loads
    getPlayerRisk(selectedPlayer, competitionId)
      .then((data) => {
        if (!cancelled) setPlayerRisk(data); // ignore a stale player's response
      })
      .catch(() => {
        if (!cancelled) setError("Failed to load player data");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedPlayer, competitionId]);

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
  };

  const handlePlayerSelected = (playerName: string) => {
    setSelectedPlayer(playerName);
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

  // Club season is "over" once the leader has played the league's full
  // matchday count. Market panels, fixture odds, and Fantasy tabs all become
  // noise post-final-day. International is never "over" in this sense (the
  // tournament is itself an active competition).
  // Bundesliga + Ligue 1 (18 teams in 2025-26) are 34 matchdays; EPL + La Liga + Serie A are 38.
  const clubMatchdays = league === "Bundesliga" || league === "Ligue 1" ? 34 : 38;
  const seasonOver = isInternational
    ? false
    : league === "La Liga"
      ? (laLigaStandings[0]?.played ?? 0) >= clubMatchdays
      : (standings?.leader?.played ?? 0) >= clubMatchdays;

  // Switcher ordering: active/upcoming competitions lead, off-season trail,
  // each group keeping its default order. Off-season leagues stay reachable,
  // just visibly secondary.
  const _now = new Date();
  const orderedLeagues = LEAGUE_META.map((m) => ({ ...m, status: leagueStatus(m.id, _now) })).sort(
    (a, b) => STATUS_RANK[a.status] - STATUS_RANK[b.status],
  );
  const currentLeagueMeta = LEAGUE_META.find((m) => m.id === league) ?? LEAGUE_META[0];
  const currentStatus = leagueStatus(String(league), _now);

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
                Predicts injury risk. Tells the story behind it.
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
                Yara predicts injury risk and writes the story behind it.
              </strong>{" "}
              She reads workload, injury history, fixtures, odds, and the latest news into one injury-driven narrative, for footballers worldwide. The World Cup, Premier League, La Liga, Bundesliga, Serie A, and Ligue 1.
            </div>
          </div>
        </div>

        {/* League Switcher + Mode Toggle + Input */}
        <div className="mb-4 sm:mb-6">
          {/* League switcher — dropdown on mobile, wrapping chips on desktop.
              Active/upcoming competitions lead; off-season ones are dimmed and
              tagged but still one tap away. */}
          {/* Mobile: single compact dropdown (scales to any number of comps) */}
          <div className="relative sm:hidden mb-3">
            <button
              onClick={() => setLeagueMenuOpen((o) => !o)}
              aria-haspopup="listbox"
              aria-expanded={leagueMenuOpen}
              className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm font-medium border ${
                darkMode
                  ? "bg-[#86efac]/10 text-[#86efac] border-[#86efac]/30"
                  : "bg-emerald-50 text-emerald-700 border-emerald-300"
              }`}
            >
              <span className="flex items-center gap-2">
                <span>{currentLeagueMeta.flag} {currentLeagueMeta.label}</span>
                <span className={`text-[10px] uppercase tracking-wide ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
                  {STATUS_LABEL[currentStatus]}
                </span>
              </span>
              <ChevronDown size={16} className={`transition-transform ${leagueMenuOpen ? "rotate-180" : ""}`} />
            </button>
            {leagueMenuOpen && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setLeagueMenuOpen(false)} />
                <div
                  role="listbox"
                  className={`absolute left-0 right-0 mt-1 z-20 rounded-lg border overflow-hidden shadow-lg ${
                    darkMode ? "bg-[#0b1220] border-gray-700" : "bg-white border-gray-200"
                  }`}
                >
                  {orderedLeagues.map((m) => {
                    const selected = league === m.id;
                    const off = m.status === "offseason";
                    return (
                      <button
                        key={m.id}
                        role="option"
                        aria-selected={selected}
                        onClick={() => {
                          handleLeagueSwitch(m.id as typeof league);
                          setLeagueMenuOpen(false);
                        }}
                        className={`w-full flex items-center justify-between px-3 py-2 text-sm text-left ${
                          selected
                            ? darkMode
                              ? "bg-[#86efac]/15 text-[#86efac]"
                              : "bg-emerald-50 text-emerald-700"
                            : darkMode
                              ? "text-gray-300 hover:bg-white/5"
                              : "text-gray-700 hover:bg-gray-50"
                        } ${off && !selected ? "opacity-60" : ""}`}
                      >
                        <span>{m.flag} {m.label}</span>
                        <span
                          className={`text-[10px] uppercase tracking-wide ${
                            m.status === "active"
                              ? darkMode ? "text-[#86efac]" : "text-emerald-600"
                              : darkMode ? "text-gray-500" : "text-gray-400"
                          }`}
                        >
                          {STATUS_LABEL[m.status]}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </>
            )}
          </div>

          {/* Desktop: wrapping chips, no horizontal scroll */}
          <div className="hidden sm:flex gap-1 mb-3 flex-wrap">
            {orderedLeagues.map((m) => {
              const selected = league === m.id;
              const off = m.status === "offseason";
              return (
                <button
                  key={m.id}
                  onClick={() => handleLeagueSwitch(m.id as typeof league)}
                  title={m.status === "active" ? "In season" : m.status === "upcoming" ? "Starts soon" : "Off-season"}
                  className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors whitespace-nowrap inline-flex items-center gap-1.5 ${
                    selected
                      ? darkMode
                        ? "bg-[#86efac]/15 text-[#86efac] border border-[#86efac]/30"
                        : "bg-emerald-50 text-emerald-700 border border-emerald-300"
                      : darkMode
                        ? "text-gray-500 hover:text-gray-300 border border-transparent"
                        : "text-gray-500 hover:text-gray-700 border border-transparent"
                  } ${off && !selected ? "opacity-60" : ""}`}
                >
                  <span>{m.flag} {m.label}</span>
                  {m.status === "active" && (
                    <span className={`w-1.5 h-1.5 rounded-full ${darkMode ? "bg-[#86efac]" : "bg-emerald-500"}`} />
                  )}
                  {off && <span className="text-[10px] uppercase tracking-wide opacity-70">off</span>}
                </button>
              );
            })}
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

        {/* Tournament winner odds (World Cup view) */}
        {isInternational && winnerOdds && (
          <WinnerOddsCard data={winnerOdds} darkMode={darkMode} />
        )}

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
                  <TeamOverview team={teamOverview} darkMode={darkMode} seasonOver={seasonOver} />

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
                    <PlayerCard player={playerRisk} darkMode={darkMode} seasonOver={seasonOver} />
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
                : isInternational
                  ? "Pick a nation to see their World Cup 2026 squad. Players with club history get a full risk score; the rest show identity and tournament context."
                  : `Select a ${league} team to view squad injury risk analysis and player predictions.`}
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
