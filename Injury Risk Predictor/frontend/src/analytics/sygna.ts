"use client";

const SYGNA_API_URL = process.env.NEXT_PUBLIC_SYGNA_API_URL?.trim() || "";
const SYGNA_INGEST_KEY = process.env.NEXT_PUBLIC_SYGNA_INGEST_KEY?.trim() || "";
const VISITOR_KEY = "__sygna_yara_visitor_id";
const SESSION_KEY = "__sygna_yara_session_id";
const LAST_SEEN_KEY = "__sygna_yara_last_seen_at";
const SESSION_TTL_MS = 30 * 60 * 1000;

type EventPayload = Record<string, unknown>;

function isEnabled(): boolean {
  return typeof window !== "undefined" && SYGNA_API_URL.length > 0;
}

function makeId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function readStorage(storage: Storage, key: string): string | null {
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(storage: Storage, key: string, value: string): void {
  try {
    storage.setItem(key, value);
  } catch {
    // Ignore storage failures and keep analytics best-effort.
  }
}

function getIdentity(): { visitorId: string; sessionId: string } {
  const visitorId = readStorage(window.localStorage, VISITOR_KEY) || makeId();
  const lastSeen = Number(readStorage(window.sessionStorage, LAST_SEEN_KEY) || 0);
  const expired = !lastSeen || Date.now() - lastSeen > SESSION_TTL_MS;
  const sessionId = expired ? makeId() : readStorage(window.sessionStorage, SESSION_KEY) || makeId();

  writeStorage(window.localStorage, VISITOR_KEY, visitorId);
  writeStorage(window.sessionStorage, SESSION_KEY, sessionId);
  writeStorage(window.sessionStorage, LAST_SEEN_KEY, String(Date.now()));

  return { visitorId, sessionId };
}

function resolveIngestUrl(): string {
  const base = SYGNA_API_URL.replace(/\/+$/, "");
  return base.endsWith("/api") ? `${base}/ingest` : `${base}/api/ingest`;
}

function buildHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json"
  };

  if (SYGNA_INGEST_KEY) {
    headers["x-sygna-key"] = SYGNA_INGEST_KEY;
  }

  return headers;
}

function send(type: "pageview" | "event", name: string, payload: EventPayload = {}): void {
  if (!isEnabled()) {
    return;
  }

  const identity = getIdentity();
  const body = JSON.stringify({
    projectKey: "yara",
    projectName: "Yara",
    type,
    name,
    sessionId: identity.sessionId,
    visitorId: identity.visitorId,
    url: window.location.href,
    path: `${window.location.pathname}${window.location.search}`,
    referrer: document.referrer || undefined,
    timestamp: new Date().toISOString(),
    payload
  });

  const sendBeaconFallback = (): boolean => {
    if (SYGNA_INGEST_KEY || typeof navigator.sendBeacon !== "function") {
      return false;
    }

    return navigator.sendBeacon(resolveIngestUrl(), new Blob([body], { type: "application/json" }));
  };

  try {
    void fetch(resolveIngestUrl(), {
      method: "POST",
      headers: buildHeaders(),
      body,
      keepalive: true,
      credentials: "omit"
    }).catch(() => {
      sendBeaconFallback();
    });
  } catch {
    sendBeaconFallback();
  }
}

export function trackYaraPageView(): void {
  send("pageview", "page_view");
}

export function trackYaraTeamSelected(team: string): void {
  send("event", "team_selected", { team });
}

export function trackYaraPlayerSelected(player: string, context: { team?: string | null; mode?: string }): void {
  send("event", "player_selected", {
    player,
    team: context.team || undefined,
    mode: context.mode
  });
}

export function trackYaraLabNotesOpened(player: string, context: { team?: string | null; mode?: string }): void {
  send("event", "lab_notes_opened", {
    player,
    team: context.team || undefined,
    mode: context.mode
  });
}

export function trackYaraFplSquadSyncCompleted(teamId: string, playersSynced: number): void {
  send("event", "fpl_squad_sync_completed", {
    teamIdLength: teamId.trim().length,
    playersSynced
  });
}
