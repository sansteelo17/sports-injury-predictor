"""Email the finished draft to my inbox via Gmail SMTP.

Image is embedded as a CID attachment (shows inline in Gmail, not a broken
link). One email per fired trigger; nothing auto-posts.
"""
from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Dict

from . import config

FORMAT_LABEL = {"matchday_board": "Matchday board"}


def _html_body(payload: Dict, x_post: str, reddit_post: str, cid: str) -> str:
    top5 = payload.get("top_5", [])
    lead = top5[0] if top5 else {}
    meta = (f"{payload.get('post_type')} &middot; {payload.get('league')} &middot; "
            f"{payload.get('matchday')} &middot; headline {lead.get('risk_score_pct','?')}%")
    pre = "white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:13px;" \
          "background:#f6f6f6;border:1px solid #e5e5e5;border-radius:8px;padding:12px;"
    return f"""\
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#111;max-width:680px;margin:0 auto">
  <p style="color:#666;font-size:12px;margin:0 0 4px">{meta}</p>
  <p style="color:#444;font-size:14px;margin:0 0 14px"><em>{payload.get('narrative_spine','')}</em></p>
  <img src="cid:{cid}" alt="card" style="width:100%;border:1px solid #e5e5e5;border-radius:10px"/>
  <h3 style="margin:20px 0 6px;font-size:14px">X post ({len(x_post)} chars)</h3>
  <div style="{pre}">{_esc(x_post)}</div>
  <h3 style="margin:20px 0 6px;font-size:14px">Reddit</h3>
  <div style="{pre}">{_esc(reddit_post)}</div>
  <p style="color:#999;font-size:12px;margin-top:18px">Approve / Edit / Kill. Nothing posts automatically.</p>
</div>"""


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_draft(payload: Dict, x_post: str, reddit_post: str, image_path: Path) -> None:
    if not config.GMAIL_APP_PASSWORD:
        raise RuntimeError(
            "GMAIL_APP_PASSWORD not set. Create a Google App Password "
            "(myaccount.google.com -> Security -> App passwords) and add it to .env."
        )
    top5 = payload.get("top_5", [])
    headline = (top5[0].get("player_name") if top5 else "").split(" ")[-1] if top5 else ""
    fmt = FORMAT_LABEL.get(payload.get("post_type", ""), payload.get("post_type", "Draft"))
    subject = f"Yara · {fmt} · {payload.get('matchday','')} · {headline}".strip()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.DRAFT_EMAIL_FROM
    msg["To"] = config.DRAFT_EMAIL_TO
    cid = "yaracard"
    msg.set_content(f"{subject}\n\nX post:\n{x_post}\n\nReddit:\n{reddit_post}\n")
    msg.add_alternative(_html_body(payload, x_post, reddit_post, cid), subtype="html")
    # Attach the image to the HTML alternative so the cid: reference resolves.
    img = msg.get_payload()[1]
    img.add_related(image_path.read_bytes(), maintype="image", subtype="png", cid=f"<{cid}>")

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(config.GMAIL_USER, config.GMAIL_APP_PASSWORD)
        server.send_message(msg)
    print(f"[email] sent: {subject}  ->  {config.DRAFT_EMAIL_TO}")
