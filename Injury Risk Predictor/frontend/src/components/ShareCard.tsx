"use client";

import { useRef, useState, useCallback } from "react";
import { PlayerRisk } from "@/types/api";
import { X, Download, Twitter } from "lucide-react";
import html2canvas from "html2canvas";

type Format = "twitter" | "instagram";

interface ShareCardProps {
  player: PlayerRisk;
  darkMode?: boolean;
  onClose: () => void;
}

const FORMAT_SIZES: Record<Format, { w: number; h: number; label: string }> = {
  twitter: { w: 1200, h: 628, label: "Twitter (1200×628)" },
  instagram: { w: 1080, h: 1920, label: "Stories (1080×1920)" },
};

export function ShareCard({ player, onClose }: ShareCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [format, setFormat] = useState<Format>("twitter");
  const [downloading, setDownloading] = useState(false);

  const size = FORMAT_SIZES[format];
  const riskColor =
    player.risk_level === "High"
      ? "#ef4444"
      : player.risk_level === "Medium"
        ? "#f59e0b"
        : "#22c55e";

  const yaraLine =
    player.yara_response?.response_text?.slice(0, 140) ??
    player.story?.slice(0, 140) ??
    null;

  const fixtureLabel = player.next_fixture
    ? `${player.next_fixture.is_home ? "vs" : "@"} ${player.next_fixture.opponent}`
    : null;

  const handleDownload = useCallback(async () => {
    if (!cardRef.current || downloading) return;
    setDownloading(true);
    try {
      const canvas = await html2canvas(cardRef.current, {
        scale: 2,
        backgroundColor: "#141414",
        useCORS: true,
        logging: false,
      });
      const link = document.createElement("a");
      link.download = `${player.name.replace(/\s+/g, "_")}_risk_card.png`;
      link.href = canvas.toDataURL("image/png");
      link.click();
    } finally {
      setDownloading(false);
    }
  }, [downloading, player.name]);

  // Scale factor for preview (fit in modal)
  const previewScale = format === "twitter" ? 0.42 : 0.22;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="bg-[#1a1a1a] rounded-2xl border border-[#2a2a2a] max-w-lg w-full p-5 space-y-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between">
          <h3 className="text-white font-semibold text-sm">Share Card</h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-white transition-colors p-1"
          >
            <X size={18} />
          </button>
        </div>

        {/* Format Toggle */}
        <div className="flex gap-2">
          {(Object.keys(FORMAT_SIZES) as Format[]).map((f) => (
            <button
              key={f}
              onClick={() => setFormat(f)}
              className={`text-xs px-3 py-1.5 rounded-lg transition-colors ${
                format === f
                  ? "bg-green-500/20 text-green-400 border border-green-500/30"
                  : "bg-[#222] text-gray-400 border border-[#333] hover:border-[#444]"
              }`}
            >
              {FORMAT_SIZES[f].label}
            </button>
          ))}
        </div>

        {/* Preview Container */}
        <div className="flex justify-center overflow-hidden rounded-xl bg-[#111] p-3">
          <div
            style={{
              width: size.w * previewScale,
              height: size.h * previewScale,
              overflow: "hidden",
            }}
          >
            <div
              ref={cardRef}
              style={{
                width: size.w,
                height: size.h,
                transform: `scale(${previewScale})`,
                transformOrigin: "top left",
                background: "linear-gradient(160deg, #141414 0%, #0d1f0d 50%, #141414 100%)",
                fontFamily: "system-ui, -apple-system, sans-serif",
                position: "relative",
                display: "flex",
                flexDirection: "column",
                padding: format === "twitter" ? "40px 48px" : "80px 56px",
              }}
            >
              {/* Top Bar */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: format === "twitter" ? 24 : 48,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  {player.team_badge_url && (
                    <img
                      src={player.team_badge_url}
                      alt=""
                      style={{ width: 32, height: 32 }}
                      crossOrigin="anonymous"
                    />
                  )}
                  <span
                    style={{
                      color: "#9ca3af",
                      fontSize: format === "twitter" ? 14 : 18,
                      fontWeight: 500,
                    }}
                  >
                    {player.team} · {player.position}
                  </span>
                </div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                  }}
                >
                  <div
                    style={{
                      width: 20,
                      height: 20,
                      borderRadius: "50%",
                      background: "#22c55e",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "white",
                      fontSize: 11,
                      fontWeight: 700,
                    }}
                  >
                    Y
                  </div>
                  <span
                    style={{
                      color: "#6b7280",
                      fontSize: 12,
                      fontWeight: 600,
                      letterSpacing: "0.05em",
                    }}
                  >
                    YARASPORTS
                  </span>
                </div>
              </div>

              {/* Player Name + Risk */}
              <div
                style={{
                  display: "flex",
                  alignItems: "flex-end",
                  justifyContent: "space-between",
                  marginBottom: format === "twitter" ? 20 : 40,
                  gap: 16,
                }}
              >
                <div>
                  <div
                    style={{
                      color: "white",
                      fontSize: format === "twitter" ? 36 : 52,
                      fontWeight: 800,
                      lineHeight: 1.1,
                      letterSpacing: "-0.02em",
                    }}
                  >
                    {player.name}
                  </div>
                  {fixtureLabel && (
                    <div
                      style={{
                        color: "#6b7280",
                        fontSize: format === "twitter" ? 14 : 18,
                        marginTop: 6,
                      }}
                    >
                      Next: {fixtureLabel}
                    </div>
                  )}
                </div>
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div
                    style={{
                      color: riskColor,
                      fontSize: format === "twitter" ? 48 : 64,
                      fontWeight: 800,
                      lineHeight: 1,
                    }}
                  >
                    {Math.round(player.risk_probability * 100)}%
                  </div>
                  <div
                    style={{
                      color: riskColor,
                      fontSize: format === "twitter" ? 12 : 15,
                      fontWeight: 600,
                      opacity: 0.8,
                      textTransform: "uppercase" as const,
                      letterSpacing: "0.08em",
                    }}
                  >
                    {player.risk_level} Risk
                  </div>
                </div>
              </div>

              {/* Yara's Response */}
              {yaraLine && (
                <div
                  style={{
                    background: "rgba(255,255,255,0.04)",
                    borderRadius: 12,
                    padding: format === "twitter" ? "14px 18px" : "20px 24px",
                    borderLeft: "3px solid #22c55e",
                    marginBottom: format === "twitter" ? 20 : 40,
                    flex: format === "instagram" ? 1 : undefined,
                  }}
                >
                  <div
                    style={{
                      color: "#d1d5db",
                      fontSize: format === "twitter" ? 15 : 20,
                      lineHeight: 1.5,
                      fontStyle: "italic",
                    }}
                  >
                    &ldquo;{yaraLine}&rdquo;
                  </div>
                </div>
              )}

              {/* Stats Row */}
              <div
                style={{
                  display: "flex",
                  gap: format === "twitter" ? 24 : 32,
                  marginTop: "auto",
                  paddingTop: format === "twitter" ? 0 : 24,
                }}
              >
                {player.archetype && (
                  <StatPill
                    label="Archetype"
                    value={player.archetype}
                    format={format}
                  />
                )}
                {player.fpl_points_projection && (
                  <StatPill
                    label="xPts"
                    value={player.fpl_points_projection.expected_points.toFixed(1)}
                    format={format}
                  />
                )}
                {player.yara_response && (
                  <StatPill
                    label="Yara Prob"
                    value={`${Math.round(player.yara_response.yara_probability * 100)}%`}
                    format={format}
                  />
                )}
                {player.implied_odds && (
                  <StatPill
                    label="Odds"
                    value={player.implied_odds.american}
                    format={format}
                  />
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={handleDownload}
            disabled={downloading}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-green-500/15 text-green-400 text-sm font-medium border border-green-500/20 hover:bg-green-500/25 transition-colors disabled:opacity-50"
          >
            <Download size={15} />
            {downloading ? "Generating..." : "Download PNG"}
          </button>
          <button
            onClick={() => {
              const text = `${player.name} — ${Math.round(player.risk_probability * 100)}% injury risk (${player.risk_level})\n${yaraLine ? `"${yaraLine}"` : ""}\n\nvia @YaraSports`;
              window.open(
                `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`,
                "_blank"
              );
            }}
            className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-[#222] text-gray-300 text-sm font-medium border border-[#333] hover:border-[#444] transition-colors"
          >
            <Twitter size={15} />
            Tweet
          </button>
        </div>
      </div>
    </div>
  );
}

function StatPill({
  label,
  value,
  format,
}: {
  label: string;
  value: string;
  format: Format;
}) {
  return (
    <div>
      <div
        style={{
          color: "#6b7280",
          fontSize: format === "twitter" ? 10 : 13,
          fontWeight: 600,
          textTransform: "uppercase" as const,
          letterSpacing: "0.08em",
          marginBottom: 2,
        }}
      >
        {label}
      </div>
      <div
        style={{
          color: "white",
          fontSize: format === "twitter" ? 15 : 20,
          fontWeight: 700,
        }}
      >
        {value}
      </div>
    </div>
  );
}
