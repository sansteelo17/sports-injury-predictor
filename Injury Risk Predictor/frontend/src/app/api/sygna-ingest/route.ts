import { NextRequest, NextResponse } from 'next/server';

/**
 * Same-origin proxy for Sygna ingest. Browsers cannot POST to ingest.usesygna.com
 * from yaraspeaks.com without that host allowing CORS; forwarding server-side avoids it.
 *
 * Render / production: set SYGNA_INGEST_KEY (secret) and optionally SYGNA_UPSTREAM_INGEST_URL.
 * Do not set NEXT_PUBLIC_SYGNA_API_URL to the external ingest host in production.
 */
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const UPSTREAM =
  process.env.SYGNA_UPSTREAM_INGEST_URL?.trim() ||
  'https://ingest.usesygna.com/api/ingest';
const SYGNA_KEY = process.env.SYGNA_INGEST_KEY?.trim() || '';

export async function POST(request: NextRequest) {
  let body: string;
  try {
    body = await request.text();
  } catch {
    return NextResponse.json({ error: 'Bad request' }, { status: 400 });
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (SYGNA_KEY) {
    headers['x-sygna-key'] = SYGNA_KEY;
  }

  try {
    const res = await fetch(UPSTREAM, {
      method: 'POST',
      headers,
      body,
    });
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: {
        'Content-Type': res.headers.get('content-type') || 'application/json',
      },
    });
  } catch (e) {
    console.error('Sygna ingest proxy failed:', e);
    return NextResponse.json({ error: 'Upstream unavailable' }, { status: 502 });
  }
}
