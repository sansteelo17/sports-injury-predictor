"""
Yara Card Renderer
==================
Parameterized PIL renderer derived from node3_risk_card.py (which was a
hardcoded Salah prototype). build_risk_card() takes a data dict and an output
path and degrades gracefully:

  - Missing fonts  -> falls back to PIL's bundled font (sized when supported).
  - Missing cutout -> renders a solid accent panel with the player's initials
                      instead of a transparent player photo.

So the pipeline never crashes on a missing asset; it just renders a simpler
card. Auto-generating per-player transparent cutouts is a separate problem and
intentionally NOT attempted here.
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np

FONT_DIR = os.getenv("YARA_FONT_DIR", "/home/user/fonts")

_INTER = {
    "black": "Inter-Black.ttf",
    "extrabold": "Inter-ExtraBold.ttf",
    "bold": "Inter-Bold.ttf",
    "semibold": "Inter-SemiBold.ttf",
    "medium": "Inter-Medium.ttf",
    "regular": "Inter-Regular.ttf",
    "light": "Inter-Light.ttf",
}


def _font(weight: str, size: int):
    """Load Inter at the given weight/size, falling back to PIL default."""
    path = os.path.join(FONT_DIR, _INTER.get(weight, _INTER["regular"]))
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.load_default(size)  # Pillow >= 10
        except TypeError:
            return ImageFont.load_default()


def _rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# Palette
BG = _rgb("#FFFFFF")
ACCENT = _rgb("#00C060")
DANGER = _rgb("#E01535")
AMBER = _rgb("#D97706")
TEXT1 = _rgb("#0D0D0D")
TEXT2 = _rgb("#5A5A5A")
TEXT3 = _rgb("#ABABAB")
DIVIDER = _rgb("#E6E6E6")

W, H = 1600, 900
SPLIT = int(W * 0.455)


def _risk_color(pct: float):
    if pct >= 65:
        return DANGER
    if pct >= 35:
        return AMBER
    return ACCENT


def _initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _left_panel(canvas, data, club_rgb):
    """Render player cutout if available, else an accent block with initials."""
    cutout = data.get("cutout_path")
    if cutout and os.path.exists(cutout):
        try:
            raw = Image.open(cutout).convert("RGBA")
            scale = H / raw.height
            player = raw.resize((int(raw.width * scale), H), Image.LANCZOS)
            rr, gg, bb, aa = player.split()
            rc = ImageEnhance.Contrast(
                ImageEnhance.Sharpness(Image.merge("RGB", (rr, gg, bb))).enhance(2.0)
            ).enhance(1.05)
            rr, gg, bb = rc.split()
            player = Image.merge("RGBA", (rr, gg, bb, aa))
            canvas.paste(player, (-60, 0), player)
            return
        except Exception:
            pass  # fall through to placeholder

    # Placeholder: tinted block + large initials
    block = Image.new("RGB", (SPLIT, H), club_rgb)
    canvas.paste(block, (0, 0))
    d = ImageDraw.Draw(canvas)
    initials = _initials(data.get("name", "?"))
    f = _font("black", 300)
    tw = int(d.textlength(initials, font=f))
    d.text(((SPLIT - tw) // 2, H // 2 - 200), initials, font=f, fill=(255, 255, 255))


def build_risk_card(data: dict, output_path: str) -> str:
    """Render a risk card PNG. Returns output_path.

    data keys (all optional except name/risk_pct):
      name, team, position, league, gw, risk_pct (0-100), archetype,
      club_color (#hex), cutout_path, signals [(label,value,delta,color_name)],
      footer_url, strip_text
    """
    club_rgb = _rgb(data.get("club_color", "#1F2937"))
    pct = float(data.get("risk_pct", 0))
    rcol = _risk_color(pct)

    canvas = Image.new("RGB", (W, H), BG)
    _left_panel(canvas, data, club_rgb)

    # Gradient fade from left panel into white
    gm = np.ones((H, W), dtype=np.float32)
    FS, FE = int(SPLIT * 0.70), SPLIT + 45
    gm[:, FE:] = 0.0
    for x in range(FS, FE):
        t = (x - FS) / (FE - FS)
        gm[:, x] = 1.0 - (3 * t ** 2 - 2 * t ** 3)
    mask = Image.fromarray((gm * 255).astype(np.uint8))
    canvas = Image.composite(canvas, Image.new("RGB", (W, H), BG), mask)

    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, 5], fill=ACCENT)
    d.rectangle([0, 5, 12, H], fill=club_rgb)
    d.rectangle([SPLIT + 44, 56, SPLIT + 45, H - 48], fill=DIVIDER)

    RX = SPLIT + 70
    RM = 72
    STRIP_Y = H - 50
    CW = W - RX - RM

    def div(y):
        d.rectangle([RX, y, W - RM, y + 1], fill=DIVIDER)

    def lbl(t, y, c=None):
        d.text((RX, y), t, font=_font("medium", 12), fill=c or TEXT3)

    y = 52
    d.text((RX, y), "INJURY RISK CARD", font=_font("medium", 13), fill=ACCENT)
    header = f"GW{data.get('gw', '')}  ·  {data.get('league', 'Premier League').upper()}"
    d.text((W - RM - int(d.textlength(header, font=_font("medium", 13))), y),
           header, font=_font("medium", 13), fill=TEXT3)
    y += 26
    div(y)
    y += 28

    # Name (split into two lines on the space if it fits the card better)
    name = (data.get("name") or "Unknown").strip()
    parts = name.split()
    if len(parts) >= 2:
        first, last = parts[0], " ".join(parts[1:])
    else:
        first, last = name, ""
    d.text((RX, y), first.upper(), font=_font("black", 72), fill=TEXT1)
    y += 76
    if last:
        d.text((RX, y), last.upper(), font=_font("black", 72), fill=TEXT1)
        y += 86
    else:
        y += 10

    tf = _font("semibold", 17)
    team = data.get("team", "")
    d.text((RX, y), team, font=tf, fill=TEXT2)
    sx = RX + int(d.textlength(team, font=tf)) + 14
    d.ellipse([sx, y + 7, sx + 4, y + 11], fill=TEXT3)
    d.text((sx + 14, y), data.get("position", "").upper(), font=tf, fill=ACCENT)
    y += 34
    div(y)
    y += 22

    lbl("INJURY RISK", y, c=TEXT2)
    y += 18
    nf = _font("black", 110)
    pf = _font("extrabold", 54)
    pct_str = str(int(round(pct)))
    d.text((RX, y), pct_str, font=nf, fill=rcol)
    nw = int(d.textlength(pct_str, font=nf))
    nh = nf.getbbox(pct_str)[3]
    ph = pf.getbbox("%")[3]
    d.text((RX + nw + 8, y + nh - ph - 2), "%", font=pf, fill=rcol)
    y += nh + 14

    bw = CW
    fw = int(bw * min(1.0, max(0.0, pct / 100.0)))
    d.rectangle([RX, y, RX + bw, y + 4], fill=DIVIDER)
    d.rectangle([RX, y, RX + fw, y + 4], fill=rcol)
    if fw > 5:
        d.ellipse([RX + fw - 5, y - 3, RX + fw + 5, y + 7], fill=rcol)
    y += 20

    archetype = (data.get("archetype") or "").upper()
    if archetype:
        lbl("ARCHETYPE", y, c=TEXT2)
        af = _font("bold", 13)
        alw = int(d.textlength("ARCHETYPE", font=_font("medium", 12)))
        ax = RX + alw + 16
        astr = f"  {archetype}  "
        aw = int(d.textlength(astr, af))
        d.rounded_rectangle([ax, y - 1, ax + aw, y + 22], radius=3,
                            fill=_rgb("#FEE2E5"), outline=(*DANGER, 90), width=1)
        d.text((ax, y + 3), astr, font=af, fill=DANGER)
        y += 34
    div(y)

    color_map = {"danger": DANGER, "amber": AMBER, "accent": ACCENT, "muted": TEXT3}
    signals = data.get("signals") or []
    signals = signals[:4]
    if signals:
        sig_top = y + 8
        sig_bot = STRIP_Y - 20
        row_h = (sig_bot - sig_top) / len(signals)
        vf = _font("semibold", 20)
        dfont = _font("semibold", 14)
        for i, sig in enumerate(signals):
            label, value, delta, cname = (list(sig) + ["", "", "", "muted"])[:4]
            dcol = color_map.get(cname, TEXT3)
            sy = sig_top + int(i * row_h)
            ch = 14 + 6 + 22
            pad = int((row_h - ch) / 2)
            ry = sy + pad
            d.rectangle([RX - 14, ry, RX - 11, ry + ch], fill=club_rgb)
            lbl(label, ry, c=TEXT2)
            d.text((RX, ry + 18), value, font=vf, fill=TEXT1)
            dw = int(d.textlength(delta, font=dfont))
            d.text((W - RM - dw, ry + 20), delta, font=dfont, fill=dcol)
            if i < len(signals) - 1:
                dr = sy + int(row_h)
                d.rectangle([RX, dr, W - RM, dr + 1], fill=DIVIDER)

    div(STRIP_Y)
    d.text((RX, STRIP_Y + 15), data.get("footer_url", "yaraspeaks.com"),
           font=_font("medium", 13), fill=ACCENT)
    strip = data.get("strip_text", "")
    if strip:
        sf = _font("medium", 13)
        sw = int(d.textlength(strip, font=sf))
        if sw > CW - 80:
            sf = _font("medium", 12)
            sw = int(d.textlength(strip, font=sf))
        d.text((W - RM - sw, STRIP_Y + 15), strip, font=sf, fill=TEXT3)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path, quality=97)
    return output_path
