'use client';

import { useState, useEffect } from 'react';
import { getTeams, getTeamOverview, getPlayerRisk } from '@/lib/api';
import { TeamOverview as TeamOverviewType, PlayerRisk } from '@/types/api';
import { TeamSelector } from '@/components/TeamSelector';
import { TeamOverview } from '@/components/TeamOverview';
import { PlayerList } from '@/components/PlayerList';
import { PlayerCard } from '@/components/PlayerCard';
import { Activity, Shield, Info } from 'lucide-react';

export default function Home() {
  const [teams, setTeams] = useState<string[]>([]);
  const [selectedTeam, setSelectedTeam] = useState('');
  const [teamOverview, setTeamOverview] = useState<TeamOverviewType | null>(null);
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null);
  const [playerRisk, setPlayerRisk] = useState<PlayerRisk | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load teams on mount
  useEffect(() => {
    getTeams()
      .then(setTeams)
      .catch((err) => setError('Failed to load teams. Is the API running?'));
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

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-gradient-to-r from-pl-purple to-purple-800 text-white py-6 px-4 shadow-lg">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center gap-3">
            <Activity size={32} className="text-pl-green" />
            <div>
              <h1 className="text-2xl font-bold">EPL Injury Predictor</h1>
              <p className="text-purple-200 text-sm">ML-powered risk analysis for Premier League players</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* How it Works Banner */}
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-6 flex items-start gap-3">
          <Info className="text-blue-500 flex-shrink-0 mt-0.5" size={20} />
          <div className="text-sm text-blue-800">
            <strong>How it works:</strong> Our ML model analyzes injury history patterns,
            recovery times, and injury severity to predict injury risk over the <strong>next 2 weeks</strong>.
            Select a team to see squad analysis, then click any player for detailed insights.
          </div>
        </div>

        {/* Team Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Choose a Premier League Team
          </label>
          <TeamSelector
            teams={teams}
            selectedTeam={selectedTeam}
            onSelectTeam={setSelectedTeam}
          />
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6 text-red-700">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
          </div>
        )}

        {/* Content Grid */}
        {teamOverview && !loading && (
          <div className="grid lg:grid-cols-3 gap-6">
            {/* Left Column - Team Overview & Player List */}
            <div className="lg:col-span-1 space-y-6">
              <TeamOverview team={teamOverview} />

              <div className="bg-white rounded-xl shadow-lg p-4">
                <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                  <Shield size={18} className="text-gray-400" />
                  Squad Players
                </h3>
                <div className="max-h-96 overflow-y-auto">
                  <PlayerList
                    players={teamOverview.players}
                    onSelectPlayer={setSelectedPlayer}
                    selectedPlayer={selectedPlayer || undefined}
                  />
                </div>
              </div>
            </div>

            {/* Right Column - Player Card */}
            <div className="lg:col-span-2">
              {playerRisk ? (
                <PlayerCard player={playerRisk} />
              ) : (
                <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                  <Shield size={48} className="mx-auto text-gray-300 mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Select a Player
                  </h3>
                  <p className="text-gray-500">
                    Click on any player from the squad list to view their detailed injury risk analysis
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!selectedTeam && !loading && (
          <div className="text-center py-16">
            <Activity size={64} className="mx-auto text-gray-300 mb-4" />
            <h2 className="text-xl font-semibold text-gray-700 mb-2">
              Welcome to EPL Injury Predictor
            </h2>
            <p className="text-gray-500 max-w-md mx-auto">
              Select a Premier League team above to view squad injury risk analysis
              and individual player predictions.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-100 border-t border-gray-200 py-6 mt-12">
        <div className="max-w-6xl mx-auto px-4 text-center text-sm text-gray-500">
          <p>
            Predictions estimate injury probability over the next 2 weeks based on historical patterns.
            Powered by ensemble ML models (CatBoost, LightGBM, XGBoost). For educational purposes only.
          </p>
        </div>
      </footer>
    </div>
  );
}
