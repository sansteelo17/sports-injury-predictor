'use client';

import { useState, useEffect } from 'react';
import { getTeams, getTeamOverview, getPlayerRisk, getFPLInsights } from '@/lib/api';
import { TeamOverview as TeamOverviewType, PlayerRisk, FPLInsights as FPLInsightsType } from '@/types/api';
import { TeamSelector } from '@/components/TeamSelector';
import { TeamOverview } from '@/components/TeamOverview';
import { PlayerList } from '@/components/PlayerList';
import { PlayerCard } from '@/components/PlayerCard';
import { FPLInsights } from '@/components/FPLInsights';
import { Activity, Shield, Info, Moon, Sun, Zap } from 'lucide-react';

export default function Home() {
  const [teams, setTeams] = useState<string[]>([]);
  const [selectedTeam, setSelectedTeam] = useState('');
  const [teamOverview, setTeamOverview] = useState<TeamOverviewType | null>(null);
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null);
  const [playerRisk, setPlayerRisk] = useState<PlayerRisk | null>(null);
  const [fplInsights, setFplInsights] = useState<FPLInsightsType | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(true);

  // Load teams and FPL data on mount
  useEffect(() => {
    getTeams()
      .then(setTeams)
      .catch((err) => setError('Failed to load teams. Is the API running?'));

    getFPLInsights()
      .then(setFplInsights)
      .catch((err) => console.log('FPL insights unavailable'));
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
    getTeamOverview(selectedTeam)
      .then((data) => {
        setTeamOverview(data);
        setSelectedPlayer(null);
        setPlayerRisk(null);
      })
      .catch((err) => setError('Failed to load team data'))
      .finally(() => setLoading(false));
  }, [selectedTeam]);

  // Load player risk when player selected
  useEffect(() => {
    if (!selectedPlayer) {
      setPlayerRisk(null);
      return;
    }

    setLoading(true);
    getPlayerRisk(selectedPlayer)
      .then(setPlayerRisk)
      .catch((err) => setError('Failed to load player data'))
      .finally(() => setLoading(false));
  }, [selectedPlayer]);

  const bgClass = darkMode ? 'bg-[#0a0a0a]' : 'bg-gray-50';
  const textClass = darkMode ? 'text-white' : 'text-gray-900';
  const mutedClass = darkMode ? 'text-gray-500' : 'text-gray-500';
  const cardClass = darkMode ? 'bg-[#141414] border-[#1f1f1f]' : 'bg-white border-gray-200';

  return (
    <div className={`min-h-screen ${bgClass} ${textClass}`}>
      {/* Header */}
      <header className={`${darkMode ? 'bg-[#141414] border-b border-[#1f1f1f]' : 'bg-white border-b border-gray-200'} py-4 px-4`}>
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Activity size={32} className={darkMode ? 'text-[#86efac]' : 'text-emerald-600'} />
              <Zap size={14} className={`absolute -top-1 -right-1 ${darkMode ? 'text-[#86efac]' : 'text-emerald-600'}`} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                Injury<span className={darkMode ? 'text-[#86efac]' : 'text-emerald-600'}>Watch</span>
              </h1>
              <p className={`text-xs ${mutedClass}`}>
                ML-powered injury prediction
              </p>
            </div>
          </div>

          {/* Dark mode toggle */}
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 rounded-lg transition-colors ${
              darkMode ? 'bg-[#1f1f1f] hover:bg-[#86efac]/20' : 'bg-gray-100 hover:bg-gray-200'
            }`}
          >
            {darkMode ? <Sun size={20} className="text-[#86efac]" /> : <Moon size={20} />}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* League Notice */}
        <div className={`${darkMode ? 'bg-[#86efac]/10 border-[#86efac]/30' : 'bg-emerald-50 border-emerald-200'} border rounded-xl p-4 mb-6`}>
          <div className="flex items-start gap-3">
            <Info className={darkMode ? 'text-[#86efac]' : 'text-emerald-600'} size={18} />
            <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              <strong className={darkMode ? 'text-[#86efac]' : 'text-emerald-600'}>Currently covering Premier League.</strong>
              {' '}More leagues coming soon. Our ML model analyzes injury history, recovery patterns, and severity to predict injury risk over the next 2 weeks.
            </div>
          </div>
        </div>

        {/* Team Selector */}
        <div className="mb-6">
          <label className={`block text-sm font-medium mb-2 ${mutedClass}`}>
            Select Team
          </label>
          <TeamSelector
            teams={teams}
            selectedTeam={selectedTeam}
            onSelectTeam={setSelectedTeam}
            darkMode={darkMode}
          />
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-6 text-red-400">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className={`animate-spin rounded-full h-8 w-8 border-b-2 ${darkMode ? 'border-[#86efac]' : 'border-emerald-600'}`}></div>
          </div>
        )}

        {/* Content Grid */}
        {teamOverview && !loading && (
          <div className="grid lg:grid-cols-3 gap-6">
            {/* Left Column - Team Overview & Player List */}
            <div className="lg:col-span-1 space-y-6">
              <TeamOverview team={teamOverview} darkMode={darkMode} />

              {fplInsights && (
                <FPLInsights
                  insights={fplInsights}
                  selectedTeam={selectedTeam}
                  darkMode={darkMode}
                />
              )}

              <div className={`${cardClass} border rounded-xl p-4`}>
                <h3 className={`font-semibold mb-3 flex items-center gap-2 ${textClass}`}>
                  <Shield size={18} className={darkMode ? 'text-[#86efac]' : 'text-emerald-600'} />
                  Squad
                </h3>
                <div className="max-h-96 overflow-y-auto">
                  <PlayerList
                    players={teamOverview.players}
                    onSelectPlayer={setSelectedPlayer}
                    selectedPlayer={selectedPlayer || undefined}
                    darkMode={darkMode}
                  />
                </div>
              </div>
            </div>

            {/* Right Column - Player Card */}
            <div className="lg:col-span-2">
              {playerRisk ? (
                <PlayerCard player={playerRisk} darkMode={darkMode} />
              ) : (
                <div className={`${cardClass} border rounded-2xl p-12 text-center`}>
                  <Shield size={48} className={`mx-auto mb-4 ${darkMode ? 'text-[#1f1f1f]' : 'text-gray-300'}`} />
                  <h3 className={`text-lg font-medium mb-2 ${textClass}`}>
                    Select a Player
                  </h3>
                  <p className={mutedClass}>
                    Click on any player to view their injury risk analysis
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!selectedTeam && !loading && (
          <div className="text-center py-16">
            <div className="relative inline-block mb-6">
              <Activity size={64} className={darkMode ? 'text-[#1f1f1f]' : 'text-gray-200'} />
              <Zap size={24} className={`absolute -top-2 -right-2 animate-pulse ${darkMode ? 'text-[#86efac]' : 'text-emerald-600'}`} />
            </div>
            <h2 className={`text-xl font-semibold mb-2 ${textClass}`}>
              Welcome to InjuryWatch
            </h2>
            <p className={`max-w-md mx-auto ${mutedClass}`}>
              Select a Premier League team to view squad injury risk analysis and player predictions.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className={`${darkMode ? 'bg-[#141414] border-t border-[#1f1f1f]' : 'bg-gray-100 border-t border-gray-200'} py-6 mt-12`}>
        <div className="max-w-6xl mx-auto px-4 text-center">
          <p className={`text-sm ${darkMode ? 'text-gray-600' : 'text-gray-500'}`}>
            Predictions estimate injury probability over the next 2 weeks.
            Powered by ensemble ML models. For educational purposes only.
          </p>
        </div>
      </footer>
    </div>
  );
}
