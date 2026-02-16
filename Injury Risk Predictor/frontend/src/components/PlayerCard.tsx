'use client';

import { PlayerRisk } from '@/types/api';
import { RiskBadge } from './RiskBadge';
import { RiskMeter } from './RiskMeter';
import {
  Activity,
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
  FlaskConical,
} from 'lucide-react';

interface PlayerCardProps {
  player: PlayerRisk;
  darkMode?: boolean;
}

export function PlayerCard({ player, darkMode = true }: PlayerCardProps) {
  const archetypeColors: Record<string, { dark: string; light: string }> = {
    'Durable': {
      dark: 'bg-[#86efac]/20 border-[#86efac]/30 text-[#86efac]',
      light: 'bg-green-50 border-green-200 text-green-800',
    },
    'Fragile': {
      dark: 'bg-red-500/20 border-red-500/30 text-red-400',
      light: 'bg-red-50 border-red-200 text-red-800',
    },
    'Currently Vulnerable': {
      dark: 'bg-orange-500/20 border-orange-500/30 text-orange-400',
      light: 'bg-orange-50 border-orange-200 text-orange-800',
    },
    'Injury Prone': {
      dark: 'bg-amber-500/20 border-amber-500/30 text-amber-400',
      light: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    },
    'Recurring': {
      dark: 'bg-purple-500/20 border-purple-500/30 text-purple-400',
      light: 'bg-purple-50 border-purple-200 text-purple-800',
    },
    'Recurring Issues': {
      dark: 'bg-purple-500/20 border-purple-500/30 text-purple-400',
      light: 'bg-purple-50 border-purple-200 text-purple-800',
    },
    'Unpredictable': {
      dark: 'bg-orange-500/20 border-orange-500/30 text-orange-400',
      light: 'bg-orange-50 border-orange-200 text-orange-800',
    },
    'Moderate Risk': {
      dark: 'bg-blue-500/20 border-blue-500/30 text-blue-400',
      light: 'bg-blue-50 border-blue-200 text-blue-800',
    },
    'Clean Record': {
      dark: 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400',
      light: 'bg-emerald-50 border-emerald-200 text-emerald-800',
    },
  };

  const getArchetypeClasses = (archetype: string) => {
    const colors = archetypeColors[archetype] || {
      dark: 'bg-gray-500/20 border-gray-500/30 text-gray-400',
      light: 'bg-gray-50 border-gray-200 text-gray-800',
    };
    return darkMode ? colors.dark : colors.light;
  };

  return (
    <div className={`rounded-2xl overflow-hidden mx-auto ${
      darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-xl'
    }`}>
      {/* Header with player image and club badge */}
      <div className={`px-4 sm:px-6 py-5 ${
        darkMode
          ? 'bg-gradient-to-r from-[#1f1f1f] to-[#141414] border-b border-[#1f1f1f]'
          : 'bg-gradient-to-r from-emerald-600 to-emerald-800 text-white'
      }`}>
        <div className="flex items-start gap-3 sm:gap-4">
          {/* Player Image */}
          {player.player_image_url && (
            <img
              src={player.player_image_url}
              alt={player.name}
              className="w-14 h-14 sm:w-16 sm:h-16 rounded-full object-cover object-top border-2 border-white/20 flex-shrink-0"
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <h2 className={`text-xl sm:text-2xl font-bold truncate ${darkMode ? 'text-white' : ''}`}>{player.name}</h2>
                <div className={`flex items-center gap-2 mt-1 flex-wrap ${darkMode ? 'text-gray-400' : 'text-emerald-100'}`}>
                  {player.team_badge_url && (
                    <img src={player.team_badge_url} alt="" className="w-4 h-4 flex-shrink-0" />
                  )}
                  <span className="text-sm">{player.team}</span>
                  <span className="hidden sm:inline">·</span>
                  <span className="text-sm">{player.position}</span>
                  <span className="hidden sm:inline">·</span>
                  <span className="text-sm">Age {player.age}</span>
                </div>
              </div>
              <div className="flex-shrink-0">
                <RiskBadge level={player.risk_level} probability={player.risk_probability} size="lg" darkMode={darkMode} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Meter */}
      <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
        <RiskMeter probability={player.risk_probability} darkMode={darkMode} />
        {player.risk_percentile != null && player.risk_percentile >= 0.7 && (
          <p className={`text-xs mt-2 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            Higher risk than {Math.round(player.risk_percentile * 100)}% of Premier League players tracked
          </p>
        )}
      </div>

      {/* Stats Grid */}
      <div className={`grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 sm:p-6 ${darkMode ? 'bg-[#0a0a0a]' : 'bg-gray-50'}`}>
        <StatCard icon={<Activity size={18} />} label="Previous Injuries" value={player.factors.previous_injuries.toString()} darkMode={darkMode} />
        <StatCard icon={<Clock size={18} />} label="Days Lost" value={player.factors.total_days_lost.toString()} darkMode={darkMode} />
        <StatCard icon={<Calendar size={18} />} label="Days Since Last" value={player.factors.days_since_last_injury.toString()} darkMode={darkMode} />
        <StatCard icon={<TrendingUp size={18} />} label="Avg Days/Injury" value={player.factors.avg_days_per_injury.toFixed(1)} darkMode={darkMode} />
      </div>

      {/* Archetype */}
      <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
        <div className="flex items-start gap-3">
          <Shield className={darkMode ? 'text-gray-500 mt-0.5' : 'text-gray-400 mt-0.5'} size={20} />
          <div className="flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Player Profile</span>
              <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${getArchetypeClasses(player.archetype)}`}>
                {player.archetype}
              </span>
            </div>
            <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              {player.archetype_description}
            </p>
          </div>
        </div>
      </div>

      {/* Risk Story */}
      {player.story && (
        <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className="flex items-start gap-3">
            <FileText className={darkMode ? 'text-[#86efac] mt-0.5' : 'text-emerald-600 mt-0.5'} size={20} />
            <div className="flex-1">
              <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Risk Analysis</span>
              <p className={`text-sm mt-2 leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                {player.story}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Next Fixture */}
      {player.next_fixture && (
        <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className={`rounded-lg p-4 ${
            darkMode ? 'bg-cyan-500/10 border border-cyan-500/30' : 'bg-cyan-50 border border-cyan-200'
          }`}>
            <div className="flex items-start gap-3">
              <Calendar className={darkMode ? 'text-cyan-400 mt-0.5' : 'text-cyan-600 mt-0.5'} size={20} />
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className={`font-semibold ${darkMode ? 'text-cyan-300' : 'text-cyan-800'}`}>Next Fixture</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    darkMode ? 'bg-cyan-500/30 text-cyan-300' : 'bg-cyan-200 text-cyan-700'
                  }`}>
                    {player.next_fixture.is_home ? 'HOME' : 'AWAY'}
                  </span>
                </div>
                <div className={`text-base sm:text-lg font-bold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {player.next_fixture.is_home ? `vs ${player.next_fixture.opponent}` : `@ ${player.next_fixture.opponent}`}
                </div>
                <div className="flex gap-4 text-sm flex-wrap">
                  {player.next_fixture.clean_sheet_odds && (
                    <div className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                      CS: <span className={darkMode ? 'text-cyan-400 font-medium' : 'text-cyan-700 font-medium'}>{player.next_fixture.clean_sheet_odds}</span>
                    </div>
                  )}
                  {player.next_fixture.win_probability != null && (
                    <div className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                      Win: <span className={darkMode ? 'text-green-400 font-medium' : 'text-green-700 font-medium'}>{Math.round(player.next_fixture.win_probability * 100)}%</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* FPL Insight */}
      {player.fpl_insight && (
        <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className={`rounded-lg p-4 ${
            darkMode ? 'bg-purple-500/10 border border-purple-500/30' : 'bg-purple-50 border border-purple-200'
          }`}>
            <div className="flex items-start gap-3">
              <span className="text-lg flex-shrink-0">⚽</span>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className={`font-semibold ${darkMode ? 'text-purple-300' : 'text-purple-800'}`}>FPL Insight</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    darkMode ? 'bg-purple-500/30 text-purple-300' : 'bg-purple-200 text-purple-700'
                  }`}>Manager Tip</span>
                </div>
                <p className={`text-sm leading-relaxed ${darkMode ? 'text-purple-200' : 'text-purple-700'}`}>
                  {player.fpl_insight}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Yara's Prediction — injury-adjusted scoring probability */}
      {player.scoring_odds && (
        <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className="flex items-start gap-3">
            <FlaskConical className={darkMode ? 'text-[#86efac] mt-0.5' : 'text-emerald-600 mt-0.5'} size={20} />
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-3 flex-wrap">
                <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Yara&apos;s Prediction</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  darkMode ? 'bg-[#86efac]/20 text-[#86efac]' : 'bg-emerald-100 text-emerald-700'
                }`}>Injury-Adjusted</span>
              </div>
              <div className={`rounded-lg p-4 mb-3 ${
                darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f]' : 'bg-gray-50 border border-gray-200'
              }`}>
                <div className="flex items-center justify-between mb-1">
                  <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Chance to score</span>
                  <span className={`text-2xl font-bold ${darkMode ? 'text-[#86efac]' : 'text-emerald-600'}`}>
                    {Math.round(player.scoring_odds.score_probability * 100)}%
                  </span>
                </div>
                <div className={`w-full h-2 rounded-full ${darkMode ? 'bg-[#1f1f1f]' : 'bg-gray-200'}`}>
                  <div
                    className={`h-full rounded-full ${darkMode ? 'bg-[#86efac]' : 'bg-emerald-500'}`}
                    style={{ width: `${Math.min(player.scoring_odds.score_probability * 100, 100)}%` }}
                  />
                </div>
                <div className={`flex gap-2 sm:gap-4 mt-3 text-xs flex-wrap ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  <span>Goals/90: <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>{player.scoring_odds.goals_per_90}</span></span>
                  <span>Assists/90: <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>{player.scoring_odds.assists_per_90}</span></span>
                  <span>Avail: <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>{Math.round(player.scoring_odds.availability_factor * 100)}%</span></span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Odds Formats */}
      {player.scoring_odds && (
        <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className="flex items-start gap-3">
            <Target className={darkMode ? 'text-green-400 mt-0.5' : 'text-green-600 mt-0.5'} size={20} />
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-3 flex-wrap">
                <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Scorer Odds</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  darkMode ? 'bg-green-500/20 text-green-400' : 'bg-green-100 text-green-700'
                }`}>Model-Derived</span>
              </div>
              <div className="grid grid-cols-3 gap-2 mb-3">
                {[
                  { name: 'American', value: player.scoring_odds.american },
                  { name: 'Decimal', value: player.scoring_odds.decimal.toFixed(2) },
                  { name: 'Fractional', value: player.scoring_odds.fractional },
                ].map((fmt) => (
                  <div key={fmt.name} className={`text-center p-2 sm:p-3 rounded-lg ${
                    darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f] hover:border-green-500/40' : 'bg-gray-50 border border-gray-200 hover:border-green-300'
                  } transition-colors cursor-default`}>
                    <div className={`text-[10px] uppercase tracking-wider mb-1 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>{fmt.name}</div>
                    <div className={`text-lg font-bold ${darkMode ? 'text-green-400' : 'text-green-600'}`}>{fmt.value}</div>
                  </div>
                ))}
              </div>
              <p className={`text-sm leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                Yara estimates {player.name.split(' ').pop()}&apos;s chance to score at <span className={`font-semibold ${darkMode ? 'text-green-400' : 'text-green-600'}`}>{Math.round(player.scoring_odds.score_probability * 100)}%</span> ({player.scoring_odds.decimal.toFixed(2)}) — adjusted for injury risk and recent form.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Yara's Response — analysis comparing Yara vs bookies */}
      {player.yara_response && (
        <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className="flex items-start gap-3">
            <FlaskConical className={darkMode ? 'text-[#86efac] mt-0.5' : 'text-emerald-600 mt-0.5'} size={20} />
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-3 flex-wrap">
                <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Yara&apos;s Response</span>
                {player.yara_response.bookmaker && (
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    darkMode ? 'bg-[#1f1f1f] text-gray-400' : 'bg-gray-100 text-gray-500'
                  }`}>vs {player.yara_response.bookmaker}</span>
                )}
              </div>
              <p className={`text-sm leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                {player.yara_response.response_text}
              </p>
              {/* Yara vs Market side-by-side */}
              {player.yara_response.market_probability != null && (
                <div className="grid grid-cols-2 gap-3 mt-3">
                  <div className={`rounded-lg p-3 text-center ${darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f]' : 'bg-gray-50 border border-gray-200'}`}>
                    <div className={`text-[10px] uppercase tracking-wider mb-1 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>Yara</div>
                    <div className={`text-lg font-bold ${darkMode ? 'text-[#86efac]' : 'text-emerald-600'}`}>{Math.round(player.yara_response.yara_probability * 100)}%</div>
                  </div>
                  <div className={`rounded-lg p-3 text-center ${darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f]' : 'bg-gray-50 border border-gray-200'}`}>
                    <div className={`text-[10px] uppercase tracking-wider mb-1 ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>Market</div>
                    <div className={`text-lg font-bold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>{Math.round(player.yara_response.market_probability * 100)}%</div>
                  </div>
                </div>
              )}
              {player.yara_response.market_probability == null && (
                <div className={`mt-3 text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  Yara projection: <span className={`font-medium ${darkMode ? 'text-[#86efac]' : 'text-emerald-600'}`}>{Math.round(player.yara_response.yara_probability * 100)}%</span> — no market line available for comparison.
                </div>
              )}
              {/* FPL tip */}
              <div className={`mt-3 rounded-lg p-3 ${darkMode ? 'bg-purple-500/10 border border-purple-500/20' : 'bg-purple-50 border border-purple-100'}`}>
                <p className={`text-xs leading-relaxed ${darkMode ? 'text-purple-300' : 'text-purple-700'}`}>
                  {player.yara_response.fpl_tip}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Clean Sheet Odds (Defenders/GK) */}
      {player.clean_sheet_odds && (
        <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className="flex items-start gap-3">
            <Shield className={darkMode ? 'text-blue-400 mt-0.5' : 'text-blue-600 mt-0.5'} size={20} />
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-3 flex-wrap">
                <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Clean Sheet Odds</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  darkMode ? 'bg-blue-500/20 text-blue-400' : 'bg-blue-100 text-blue-700'
                }`}>Defender</span>
              </div>
              <div className="grid grid-cols-3 gap-3 mb-3">
                <OddsBox label="CS Prob" value={`${Math.round(player.clean_sheet_odds.clean_sheet_probability * 100)}%`} highlight darkMode={darkMode} highlightColor="blue" />
                <OddsBox label="Odds" value={player.clean_sheet_odds.american} darkMode={darkMode} />
                <OddsBox label="GA/Game" value={player.clean_sheet_odds.goals_conceded_per_game.toString()} darkMode={darkMode} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* FPL Value Assessment */}
      {player.fpl_value && (
        <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className={`rounded-lg p-4 ${
            player.fpl_value.tier === 'Premium'
              ? darkMode ? 'bg-yellow-500/10 border border-yellow-500/30' : 'bg-yellow-50 border border-yellow-200'
              : player.fpl_value.tier === 'Strong'
              ? darkMode ? 'bg-green-500/10 border border-green-500/30' : 'bg-green-50 border border-green-200'
              : player.fpl_value.tier === 'Avoid'
              ? darkMode ? 'bg-red-500/10 border border-red-500/30' : 'bg-red-50 border border-red-200'
              : darkMode ? 'bg-blue-500/10 border border-blue-500/30' : 'bg-blue-50 border border-blue-200'
          }`}>
            <div className="flex items-start gap-3">
              <Star className={`flex-shrink-0 ${
                player.fpl_value.tier === 'Premium' ? 'text-yellow-400'
                : player.fpl_value.tier === 'Strong' ? 'text-green-400'
                : player.fpl_value.tier === 'Avoid' ? 'text-red-400'
                : 'text-blue-400'
              }`} size={20} />
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    FPL Value: {player.fpl_value.tier}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    darkMode ? 'bg-white/10 text-gray-300' : 'bg-gray-200 text-gray-700'
                  }`}>{player.fpl_value.price}m</span>
                </div>
                <p className={`text-sm leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  {player.fpl_value.verdict}
                </p>
                {player.fpl_value.position_insight && (
                  <p className={`text-sm mt-2 leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    {player.fpl_value.position_insight}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Injury Odds */}
      {player.implied_odds && (
        <div className={`px-4 sm:px-6 py-4 ${darkMode ? 'border-b border-[#1f1f1f]' : 'border-b border-gray-100'}`}>
          <div className="flex items-start gap-3">
            <Coins className={darkMode ? 'text-amber-400 mt-0.5' : 'text-amber-600 mt-0.5'} size={20} />
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-3 flex-wrap">
                <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Injury Odds</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  darkMode ? 'bg-amber-500/20 text-amber-400' : 'bg-amber-100 text-amber-700'
                }`}>Implied</span>
              </div>
              <p className={`text-xs mb-3 ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                Implied odds for this player getting injured in the next 2 weeks:
              </p>
              <div className="grid grid-cols-3 gap-3">
                <OddsBox label="American" value={player.implied_odds.american} darkMode={darkMode} />
                <OddsBox label="Decimal" value={player.implied_odds.decimal.toString()} darkMode={darkMode} />
                <OddsBox label="Fractional" value={player.implied_odds.fractional} darkMode={darkMode} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="px-4 sm:px-6 py-4">
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle className="text-amber-500" size={20} />
          <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Key Factors & Recommendations</span>
        </div>
        <ul className="space-y-2">
          {player.recommendations.map((rec, i) => (
            <li key={i} className={`flex items-start gap-2 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              <ChevronRight className={`mt-0.5 flex-shrink-0 ${darkMode ? 'text-[#86efac]' : 'text-emerald-600'}`} size={16} />
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Model Details (Collapsible) */}
      <details className={darkMode ? 'border-t border-[#1f1f1f]' : 'border-t border-gray-100'}>
        <summary className={`px-4 sm:px-6 py-3 cursor-pointer text-sm ${
          darkMode ? 'text-gray-500 hover:bg-[#1f1f1f]' : 'text-gray-500 hover:bg-gray-50'
        }`}>Model Details</summary>
        <div className="px-4 sm:px-6 pb-4">
          <div className="grid grid-cols-3 gap-4 text-center">
            <ModelScore label="LightGBM" value={player.model_predictions.lgb} darkMode={darkMode} />
            <ModelScore label="XGBoost" value={player.model_predictions.xgb} darkMode={darkMode} />
            <ModelScore label="CatBoost" value={player.model_predictions.catboost} darkMode={darkMode} />
          </div>
        </div>
      </details>
    </div>
  );
}

function StatCard({ icon, label, value, darkMode = true }: {
  icon: React.ReactNode; label: string; value: string; darkMode?: boolean;
}) {
  return (
    <div className={`rounded-lg p-2 sm:p-3 text-center ${
      darkMode ? 'bg-[#141414] border border-[#1f1f1f]' : 'bg-white shadow-sm'
    }`}>
      <div className={`flex justify-center mb-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{icon}</div>
      <div className={`text-lg sm:text-xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{value}</div>
      <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{label}</div>
    </div>
  );
}

function OddsBox({ label, value, darkMode = true, highlight = false, highlightColor = 'green' }: {
  label: string; value: string; darkMode?: boolean; highlight?: boolean; highlightColor?: string;
}) {
  const colorMap: Record<string, string> = {
    green: darkMode ? 'text-green-400' : 'text-green-600',
    blue: darkMode ? 'text-blue-400' : 'text-blue-600',
  };
  return (
    <div className={`text-center p-2 sm:p-3 rounded-lg ${
      darkMode ? 'bg-[#0a0a0a] border border-[#1f1f1f]' : 'bg-gray-50 border border-gray-200'
    }`}>
      <div className={`text-base sm:text-lg font-bold ${
        highlight ? colorMap[highlightColor] || colorMap.green : darkMode ? 'text-white' : 'text-gray-900'
      }`}>{value}</div>
      <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{label}</div>
    </div>
  );
}

function ModelScore({ label, value, darkMode = true }: {
  label: string; value: number; darkMode?: boolean;
}) {
  return (
    <div>
      <div className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
        {Math.round(value * 100)}%
      </div>
      <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{label}</div>
    </div>
  );
}
