"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { PlayerRisk } from "@/types/api";
import { X, Download, Twitter } from "lucide-react";

interface ShareCardProps {
  player: PlayerRisk;
  darkMode?: boolean;
  onClose: () => void;
}

const W = 1200;
const H = 630;
const FONT = "'Inter', system-ui, -apple-system, sans-serif";

const GREEN = "#00ff87";
const AMBER = "#f59e0b";
const RED = "#ef4444";

function riskColor(prob: number): string {
  if (prob >= 0.6) return RED;
  if (prob >= 0.3) return AMBER;
  return GREEN;
}

function riskLabel(level: string): string {
  return level.toUpperCase();
}

function hexToRgba(hex: string, a: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawArc(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  progress: number,
  color: string,
  lineWidth: number
) {
  const startAngle = -Math.PI * 0.75;
  const totalAngle = Math.PI * 1.5;

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, radius, startAngle, startAngle + totalAngle);
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.stroke();

  // Progress
  ctx.beginPath();
  ctx.arc(cx, cy, radius, startAngle, startAngle + totalAngle * progress);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.stroke();
}

function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
  maxLines: number
): string[] {
  const words = text.split(" ");
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const test = current ? `${current} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && current) {
      lines.push(current);
      current = word;
      if (lines.length >= maxLines) {
        lines[lines.length - 1] += "...";
        return lines;
      }
    } else {
      current = test;
    }
  }
  if (current && lines.length < maxLines) lines.push(current);
  return lines;
}

function loadImage(src: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

async function drawCard(
  canvas: HTMLCanvasElement,
  player: PlayerRisk
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  canvas.width = W;
  canvas.height = H;

  const rc = riskColor(player.risk_probability);
  const riskPct = Math.round(player.risk_probability * 100);
  const f = player.factors;

  // ── Solid dark background ──
  ctx.fillStyle = "#0d0d0d";
  ctx.fillRect(0, 0, W, H);

  // ── Left panel: player image zone (420px wide) ──
  const imgPanelW = 420;

  // Risk-colored gradient behind player image
  const imgGrad = ctx.createLinearGradient(0, 0, imgPanelW, 0);
  imgGrad.addColorStop(0, hexToRgba(rc, 0.15));
  imgGrad.addColorStop(0.7, hexToRgba(rc, 0.04));
  imgGrad.addColorStop(1, "rgba(13,13,13,0)");
  ctx.fillStyle = imgGrad;
  ctx.fillRect(0, 0, imgPanelW + 60, H);

  // Bottom fade for image
  const bottomFade = ctx.createLinearGradient(0, H - 180, 0, H);
  bottomFade.addColorStop(0, "rgba(13,13,13,0)");
  bottomFade.addColorStop(1, "rgba(13,13,13,1)");

  // Load and draw player image
  let playerImg: HTMLImageElement | null = null;
  if (player.player_image_url) {
    playerImg = await loadImage(player.player_image_url);
  }

  if (playerImg) {
    // Draw player image — centered in left panel, from waist up
    const imgH = H * 0.88;
    const imgW = imgH * (playerImg.width / playerImg.height);
    const imgX = (imgPanelW - imgW) / 2 + 20;
    const imgY = H - imgH;

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, imgPanelW + 40, H);
    ctx.clip();
    ctx.drawImage(playerImg, imgX, imgY, imgW, imgH);
    ctx.restore();

    // Fade overlay at bottom
    ctx.fillStyle = bottomFade;
    ctx.fillRect(0, H - 180, imgPanelW + 40, 180);

    // Fade on right edge of image
    const rightFade = ctx.createLinearGradient(imgPanelW - 40, 0, imgPanelW + 60, 0);
    rightFade.addColorStop(0, "rgba(13,13,13,0)");
    rightFade.addColorStop(1, "rgba(13,13,13,1)");
    ctx.fillStyle = rightFade;
    ctx.fillRect(imgPanelW - 40, 0, 100, H);
  } else {
    // No image: draw large shirt number or initials
    ctx.fillStyle = hexToRgba(rc, 0.08);
    ctx.font = `800 200px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const fallback = player.shirt_number != null ? `${player.shirt_number}` : player.name.split(" ").map(w => w[0]).join("");
    ctx.fillText(fallback, imgPanelW / 2 + 20, H / 2);
  }

  // ── Right panel content ──
  const rx = imgPanelW + 60; // right content start x
  const rw = W - rx - 48; // available width

  // ── Top: YARA branding + position badge ──
  const topY = 44;
  ctx.textBaseline = "alphabetic";

  // YARA mark
  ctx.fillStyle = GREEN;
  ctx.beginPath();
  ctx.arc(rx + 11, topY + 2, 11, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#0d0d0d";
  ctx.font = `bold 13px ${FONT}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Y", rx + 11, topY + 3);

  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "#555";
  ctx.font = `600 11px ${FONT}`;
  ctx.textAlign = "left";
  ctx.fillText("YARA", rx + 28, topY + 7);

  // Position badge — right aligned
  ctx.textAlign = "right";
  ctx.fillStyle = hexToRgba(rc, 0.15);
  const posText = player.position;
  ctx.font = `700 11px ${FONT}`;
  const posW = ctx.measureText(posText).width + 16;
  roundRect(ctx, W - 48 - posW, topY - 9, posW, 22, 11);
  ctx.fill();
  ctx.fillStyle = rc;
  ctx.textAlign = "center";
  ctx.fillText(posText, W - 48 - posW / 2, topY + 5);

  // ── Player name ──
  const nameY = topY + 56;
  ctx.textAlign = "left";
  ctx.fillStyle = "#ffffff";
  ctx.font = `800 42px ${FONT}`;

  // Split name for two-line layout if long
  const nameParts = player.name.split(" ");
  if (nameParts.length >= 2 && ctx.measureText(player.name).width > rw) {
    const firstName = nameParts[0];
    const lastName = nameParts.slice(1).join(" ");
    ctx.font = `400 24px ${FONT}`;
    ctx.fillStyle = "#888";
    ctx.fillText(firstName.toUpperCase(), rx, nameY - 10);
    ctx.font = `800 42px ${FONT}`;
    ctx.fillStyle = "#fff";
    ctx.fillText(lastName.toUpperCase(), rx, nameY + 34);
  } else {
    ctx.fillText(player.name, rx, nameY);
  }

  // Team + age line
  const metaY = nameY + (nameParts.length >= 2 && ctx.measureText(player.name).width > rw ? 56 : 24);
  ctx.fillStyle = "#666";
  ctx.font = `500 14px ${FONT}`;
  ctx.textAlign = "left";

  // Team badge inline if available
  let teamBadge: HTMLImageElement | null = null;
  if (player.team_badge_url) {
    teamBadge = await loadImage(player.team_badge_url);
  }
  let teamTextX = rx;
  if (teamBadge) {
    ctx.drawImage(teamBadge, rx, metaY - 12, 16, 16);
    teamTextX = rx + 22;
  }
  ctx.fillText(`${player.team}  ·  Age ${player.age}`, teamTextX, metaY);

  // ── Risk gauge — circular arc ──
  const gaugeR = 56;
  const gaugeCx = W - 48 - gaugeR - 10;
  const gaugeCy = nameY + 20;
  drawArc(ctx, gaugeCx, gaugeCy, gaugeR, player.risk_probability, rc, 8);

  // Risk number in center of arc
  ctx.fillStyle = rc;
  ctx.font = `800 38px ${FONT}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(`${riskPct}`, gaugeCx, gaugeCy - 2);

  // % below number
  ctx.font = `600 14px ${FONT}`;
  ctx.fillText("%", gaugeCx, gaugeCy + 22);

  // RISK label below arc
  ctx.font = `700 10px ${FONT}`;
  ctx.fillStyle = hexToRgba(rc, 0.7);
  ctx.fillText(`${riskLabel(player.risk_level)} RISK`, gaugeCx, gaugeCy + gaugeR + 16);

  ctx.textBaseline = "alphabetic";

  // ── Stats grid — 2x2 ──
  const statsStartY = metaY + 36;
  const statBoxW = (rw - 16) / 2;
  const statBoxH = 72;
  const statGap = 12;

  const statItems = [
    { label: "INJURIES", value: String(f.previous_injuries), color: f.previous_injuries >= 5 ? RED : f.previous_injuries >= 3 ? AMBER : "#fff" },
    { label: "DAYS LOST", value: String(f.total_days_lost), color: f.total_days_lost >= 200 ? RED : f.total_days_lost >= 80 ? AMBER : "#fff" },
    {
      label: "DAYS SINCE LAST",
      value: f.days_since_last_injury >= 999 ? "365+" : String(f.days_since_last_injury),
      color: f.days_since_last_injury < 60 ? AMBER : "#fff",
    },
    ...(player.acwr != null && player.acwr > 0
      ? [{ label: "ACWR", value: player.acwr.toFixed(2), color: player.acwr >= 1.5 ? RED : player.acwr >= 1.2 ? AMBER : "#fff" }]
      : [{ label: "ARCHETYPE", value: player.archetype, color: GREEN }]),
  ];

  statItems.forEach((stat, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const sx = rx + col * (statBoxW + statGap);
    const sy = statsStartY + row * (statBoxH + statGap);

    roundRect(ctx, sx, sy, statBoxW, statBoxH, 8);
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Label
    ctx.fillStyle = "#555";
    ctx.font = `600 10px ${FONT}`;
    ctx.textAlign = "left";
    ctx.fillText(stat.label, sx + 14, sy + 24);

    // Value
    ctx.fillStyle = stat.color;
    const isLongVal = stat.value.length > 6;
    ctx.font = `700 ${isLongVal ? 18 : 26}px ${FONT}`;
    ctx.fillText(stat.value, sx + 14, sy + isLongVal ? 52 : 56);
  });

  // ── Narrative strip at bottom ──
  const narY = H - 100;
  const narrativeLine = player.fpl_insight || player.story || "";
  if (narrativeLine) {
    // Subtle divider
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.fillRect(rx, narY - 12, rw, 1);

    // Quote mark
    ctx.fillStyle = hexToRgba(rc, 0.5);
    ctx.font = `700 28px ${FONT}`;
    ctx.textAlign = "left";
    ctx.fillText("\u201C", rx, narY + 14);

    // Text
    ctx.fillStyle = "#999";
    ctx.font = `400 13px ${FONT}`;
    const lines = wrapText(ctx, narrativeLine, rw - 24, 2);
    lines.forEach((line, i) => {
      ctx.fillText(line, rx + 18, narY + 10 + i * 20);
    });
  }

  // ── Bottom bar ──
  ctx.fillStyle = "#333";
  ctx.font = `500 11px ${FONT}`;
  ctx.textAlign = "left";
  ctx.fillText("yaraspeaks.com", 40, H - 22);

  ctx.textAlign = "right";
  ctx.fillStyle = "#2a2a2a";
  ctx.font = `600 10px ${FONT}`;
  ctx.fillText("INJURY RISK PREDICTION", W - 40, H - 22);

  // Next fixture pill at bottom-left of image panel
  if (player.next_fixture) {
    const venue = player.next_fixture.is_home ? "HOME" : "AWAY";
    const fixText = `${venue} vs ${player.next_fixture.opponent}`.toUpperCase();
    ctx.font = `600 10px ${FONT}`;
    const fw = ctx.measureText(fixText).width + 20;
    const fy = H - 50;
    const fx = 40;
    roundRect(ctx, fx, fy, fw, 22, 11);
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "#aaa";
    ctx.textAlign = "center";
    ctx.fillText(fixText, fx + fw / 2, fy + 14);
  }
}

export function ShareCard({ player, onClose }: ShareCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [downloading, setDownloading] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setReady(false);
    if (canvasRef.current) {
      drawCard(canvasRef.current, player).then(() => setReady(true));
    }
  }, [player]);

  const handleDownload = useCallback(() => {
    if (!canvasRef.current || downloading) return;
    setDownloading(true);
    try {
      const link = document.createElement("a");
      link.download = `${player.name.replace(/\s+/g, "_")}_yara.png`;
      link.href = canvasRef.current.toDataURL("image/png");
      link.click();
    } finally {
      setDownloading(false);
    }
  }, [downloading, player.name]);

  const handleTweet = useCallback(() => {
    const riskPct = Math.round(player.risk_probability * 100);
    const text = `${player.name} | ${riskPct}% injury risk\n\nyaraspeaks.com`;
    window.open(
      `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`,
      "_blank"
    );
  }, [player]);

  const previewScale = 0.46;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4"
      onClick={onClose}
    >
      <div
        className="bg-[#161616] rounded-2xl border border-[#252525] max-w-[580px] w-full p-5 space-y-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between">
          <h3 className="text-white/60 font-medium text-xs tracking-widest uppercase">
            Share
          </h3>
          <button
            onClick={onClose}
            className="text-gray-600 hover:text-white transition-colors p-1"
          >
            <X size={18} />
          </button>
        </div>

        {/* Preview */}
        <div className="flex justify-center rounded-xl bg-[#0a0a0a] border border-[#1a1a1a] p-3 overflow-hidden">
          <div
            style={{
              width: W * previewScale,
              height: H * previewScale,
              overflow: "hidden",
              opacity: ready ? 1 : 0.3,
              transition: "opacity 0.3s",
            }}
          >
            <canvas
              ref={canvasRef}
              style={{
                width: W,
                height: H,
                transform: `scale(${previewScale})`,
                transformOrigin: "top left",
              }}
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={handleDownload}
            disabled={downloading || !ready}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-[#00ff87]/10 text-[#00ff87] text-sm font-medium border border-[#00ff87]/20 hover:bg-[#00ff87]/15 transition-colors disabled:opacity-50"
          >
            <Download size={15} />
            {downloading ? "Saving..." : "Download"}
          </button>
          <button
            onClick={handleTweet}
            className="flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-[#1a1a1a] text-gray-400 text-sm font-medium border border-[#252525] hover:border-[#333] hover:text-white transition-colors"
          >
            <Twitter size={15} />
            Post
          </button>
        </div>
      </div>
    </div>
  );
}
