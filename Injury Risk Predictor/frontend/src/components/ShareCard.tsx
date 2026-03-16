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
  twitter: { w: 1200, h: 628, label: "Twitter (1200x628)" },
  instagram: { w: 1080, h: 1920, label: "Stories (1080x1920)" },
};

function riskColor(level: string): string {
  if (level === "High") return "#ef4444";
  if (level === "Medium") return "#f59e0b";
  return "#22c55e";
}

function tierColor(tier: string): string {
  const t = (tier || "").toLowerCase();
  if (t === "premium") return "#a855f7";
  if (t === "strong") return "#22c55e";
  if (t === "decent") return "#3b82f6";
  if (t === "rotation") return "#f59e0b";
  return "#ef4444"; // avoid
}

export function ShareCard({ player, onClose }: ShareCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [format, setFormat] = useState<Format>("twitter");
  const [downloading, setDownloading] = useState(false);

  const size = FORMAT_SIZES[format];
  const rc = riskColor(player.risk_level);
  const isStories = format === "instagram";

  const riskPct = Math.round(player.risk_probability * 100);

  const fixtureLabel = player.next_fixture
    ? `${player.next_fixture.is_home ? "vs" : "@"} ${player.next_fixture.opponent}`
    : null;

  // Best available narrative line
  const insightLine = player.fpl_insight || null;
  const storyLine = player.story?.slice(0, 180) || null;
  const displayLine = insightLine || storyLine;

  const valueTier = player.fpl_value?.tier || null;
  const valuePrice = player.fpl_value?.price || null;

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
      link.download = `${player.name.replace(/\s+/g, "_")}_yara.png`;
      link.href = canvas.toDataURL("image/png");
      link.click();
    } finally {
      setDownloading(false);
    }
  }, [downloading, player.name]);

  const previewScale = isStories ? 0.22 : 0.42;
  const pad = isStories ? 56 : 44;
  const gap = isStories ? 40 : 20;

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
        <div className="flex flex-wrap gap-2">
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
        <div className="flex justify-center overflow-x-auto rounded-xl bg-[#111] p-3">
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
                background:
                  "linear-gradient(160deg, #141414 0%, #0d1f0d 50%, #141414 100%)",
                fontFamily: "system-ui, -apple-system, sans-serif",
                position: "relative",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                padding: `${pad}px ${pad + 4}px`,
              }}
            >
              {/* ── Top: Branding + Team ── */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  {player.team_badge_url && (
                    <img
                      src={player.team_badge_url}
                      alt=""
                      style={{ width: isStories ? 36 : 28, height: isStories ? 36 : 28 }}
                      crossOrigin="anonymous"
                    />
                  )}
                  <span
                    style={{
                      color: "#9ca3af",
                      fontSize: isStories ? 18 : 14,
                      fontWeight: 500,
                    }}
                  >
                    {player.team} · {player.position}
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div
                    style={{
                      width: isStories ? 24 : 20,
                      height: isStories ? 24 : 20,
                      borderRadius: "50%",
                      background: "#22c55e",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "white",
                      fontSize: isStories ? 13 : 11,
                      fontWeight: 700,
                    }}
                  >
                    Y
                  </div>
                  <span
                    style={{
                      color: "#6b7280",
                      fontSize: isStories ? 14 : 11,
                      fontWeight: 600,
                      letterSpacing: "0.05em",
                    }}
                  >
                    YARA
                  </span>
                </div>
              </div>

              {/* ── Middle: Name + Risk + Fixture ── */}
              <div style={{ marginTop: gap }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-end",
                    justifyContent: "space-between",
                    gap: 16,
                  }}
                >
                  <div>
                    <div
                      style={{
                        color: "white",
                        fontSize: isStories ? 52 : 38,
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
                          fontSize: isStories ? 18 : 14,
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
                        color: rc,
                        fontSize: isStories ? 64 : 48,
                        fontWeight: 800,
                        lineHeight: 1,
                      }}
                    >
                      {riskPct}%
                    </div>
                    <div
                      style={{
                        color: rc,
                        fontSize: isStories ? 15 : 12,
                        fontWeight: 600,
                        opacity: 0.8,
                        textTransform: "uppercase",
                        letterSpacing: "0.08em",
                      }}
                    >
                      Injury Risk
                    </div>
                  </div>
                </div>

                {/* Insight quote */}
                {displayLine && (
                  <div
                    style={{
                      background: "rgba(255,255,255,0.04)",
                      borderRadius: 12,
                      padding: isStories ? "20px 24px" : "14px 18px",
                      borderLeft: "3px solid #22c55e",
                      marginTop: gap,
                    }}
                  >
                    <div
                      style={{
                        color: "#d1d5db",
                        fontSize: isStories ? 20 : 15,
                        lineHeight: 1.5,
                      }}
                    >
                      {displayLine}
                    </div>
                  </div>
                )}
              </div>

              {/* ── Bottom: Pills ── */}
              <div
                style={{
                  display: "flex",
                  gap: isStories ? 20 : 16,
                  marginTop: "auto",
                  paddingTop: gap,
                  flexWrap: "wrap",
                }}
              >
                {player.archetype && (
                  <Pill
                    label="Profile"
                    value={player.archetype}
                    isStories={isStories}
                  />
                )}
                {valueTier && (
                  <Pill
                    label="FPL Value"
                    value={
                      valuePrice
                        ? `${valueTier} (${"\u00A3"}${valuePrice.toFixed(1)}m)`
                        : valueTier
                    }
                    color={tierColor(valueTier)}
                    isStories={isStories}
                  />
                )}
                {player.factors && (
                  <Pill
                    label="Injuries"
                    value={`${player.factors.previous_injuries} career`}
                    isStories={isStories}
                  />
                )}
                {player.acwr != null && player.acwr > 0 && (
                  <Pill
                    label="ACWR"
                    value={player.acwr.toFixed(2)}
                    color={player.acwr >= 1.5 ? "#ef4444" : undefined}
                    isStories={isStories}
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
              const text = `${player.name} — ${riskPct}% injury risk (${player.risk_level})${displayLine ? `\n"${displayLine.slice(0, 140)}"` : ""}\n\nvia @YaraSports`;
              window.open(
                `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`,
                "_blank"
              );
            }}
            className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-[#222] text-gray-300 text-sm font-medium border border-[#333] hover:border-[#444] transition-colors"
          >
            <Twitter size={15} />
            Post
          </button>
        </div>
      </div>
    </div>
  );
}

function Pill({
  label,
  value,
  color,
  isStories,
}: {
  label: string;
  value: string;
  color?: string;
  isStories: boolean;
}) {
  return (
    <div>
      <div
        style={{
          color: "#6b7280",
          fontSize: isStories ? 12 : 10,
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          marginBottom: 2,
        }}
      >
        {label}
      </div>
      <div
        style={{
          color: color || "white",
          fontSize: isStories ? 18 : 14,
          fontWeight: 700,
        }}
      >
        {value}
      </div>
    </div>
  );
}
