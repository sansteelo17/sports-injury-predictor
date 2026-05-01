from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np

FONTS = "/home/user/fonts"
def F(weight, size):
    names = {
        "black":     "Inter-Black.ttf",
        "extrabold": "Inter-ExtraBold.ttf",
        "bold":      "Inter-Bold.ttf",
        "semibold":  "Inter-SemiBold.ttf",
        "medium":    "Inter-Medium.ttf",
        "regular":   "Inter-Regular.ttf",
        "light":     "Inter-Light.ttf",
    }
    return ImageFont.truetype(f"{FONTS}/{names[weight]}", size)

def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

BG        = rgb("#FFFFFF")
ACCENT    = rgb("#00C060")
DANGER    = rgb("#E01535")
AMBER     = rgb("#D97706")
TEXT1     = rgb("#0D0D0D")
TEXT2     = rgb("#5A5A5A")
TEXT3     = rgb("#ABABAB")
DIVIDER   = rgb("#E6E6E6")
LPOOL_RED = rgb("#C8102E")

W, H    = 1600, 900
SPLIT   = int(W * 0.455)

# Background gradient (barely-there Liverpool tint)
bg_arr = np.full((H, W, 3), 255, dtype=np.float32)
lr = np.array([200.0, 16.0, 46.0])
for x in range(SPLIT + 80):
    t   = max(0.0, 1.0 - x / (SPLIT + 80))
    col = np.clip(255 + (lr - 255) * t * 0.035, 0, 255)
    bg_arr[:, x, :] = col
canvas = Image.fromarray(bg_arr.astype(np.uint8))

# Player
raw    = Image.open("/home/user/salah_sharp.png").convert("RGBA")
scale  = H / raw.height
p_w    = int(raw.width * scale)
player = raw.resize((p_w, H), Image.LANCZOS)
rr, gg, bb, aa = player.split()
rc = Image.merge("RGB", (rr, gg, bb))
rc = ImageEnhance.Sharpness(rc).enhance(2.0)
rc = ImageEnhance.Contrast(rc).enhance(1.05)
rr, gg, bb = rc.split()
player = Image.merge("RGBA", (rr, gg, bb, aa))
canvas.paste(player, (-60, 0), player)

# Gradient fade
gm = np.ones((H, W), dtype=np.float32)
FS, FE = int(SPLIT * 0.70), SPLIT + 45
for x in range(FE, W): gm[:, x] = 0.0
for x in range(FS, FE):
    t = (x - FS) / (FE - FS)
    gm[:, x] = 1.0 - (3*t**2 - 2*t**3)
for y in range(int(H * 0.82), H):
    t = (y - H*0.82) / (H*0.18)
    gm[y, :] *= max(0.0, 1.0 - t*0.55)

mask   = Image.fromarray((gm * 255).astype(np.uint8))
canvas = Image.composite(canvas, Image.new("RGB", (W, H), BG), mask)
d      = ImageDraw.Draw(canvas)

d.rectangle([0, 0, W, 5],  fill=ACCENT)
d.rectangle([0, 5, 12, H], fill=LPOOL_RED)
d.rectangle([SPLIT + 44, 56, SPLIT + 45, H - 48], fill=DIVIDER)

RX = SPLIT + 70; RM = 72; STRIP_Y = H - 50; CW = W - RX - RM

def div(y): d.rectangle([RX, y, W-RM, y+1], fill=DIVIDER)
def lbl(t, y, c=None): d.text((RX, y), t, font=F("medium", 12), fill=c or TEXT3)

y = 52
d.text((RX, y), "INJURY RISK CARD", font=F("medium", 13), fill=ACCENT)
gw = "GW36  ·  PREMIER LEAGUE"
d.text((W-RM-int(d.textlength(gw, font=F("medium", 13))), y), gw, font=F("medium", 13), fill=TEXT3)
y += 26; div(y); y += 28

d.text((RX, y), "MOHAMED", font=F("black", 72), fill=TEXT1); y += 76
d.text((RX, y), "SALAH",   font=F("black", 72), fill=TEXT1); y += 86

tf = F("semibold", 17)
d.text((RX, y), "Liverpool", font=tf, fill=TEXT2)
sx = RX + int(d.textlength("Liverpool", font=tf)) + 14
d.ellipse([sx, y+7, sx+4, y+11], fill=TEXT3)
d.text((sx+14, y), "MID", font=tf, fill=ACCENT)
y += 34; div(y); y += 22

lbl("INJURY RISK", y, c=TEXT2); y += 18

nf = F("black", 110); pf = F("extrabold", 54)
d.text((RX, y), "78", font=nf, fill=DANGER)
nw = int(d.textlength("78", font=nf))
nh = nf.getbbox("78")[3]; ph = pf.getbbox("%")[3]
d.text((RX+nw+8, y+nh-ph-2), "%", font=pf, fill=DANGER)
y += nh + 14

bw = CW; fw = int(bw * 0.78)
d.rectangle([RX, y, RX+bw, y+4], fill=DIVIDER)
d.rectangle([RX, y, RX+fw, y+4], fill=DANGER)
d.ellipse([RX+fw-5, y-3, RX+fw+5, y+7], fill=DANGER)
y += 20

lbl("ARCHETYPE", y, c=TEXT2)
af = F("bold", 13); alw = int(d.textlength("ARCHETYPE", font=F("medium", 12)))
ax = RX+alw+16; astr = "  FRAGILE  "; aw = int(d.textlength(astr, af))
d.rounded_rectangle([ax, y-1, ax+aw, y+22], radius=3,
                    fill=rgb("#FEE2E5"), outline=(*DANGER, 90), width=1)
d.text((ax, y+3), astr, font=af, fill=DANGER)
y += 34; div(y); y += 0

signals = [
    ("MINUTES TREND", "3 of last 4 starts subbed off before 75'", "↓ −8 min avg", DANGER),
    ("LOAD INDEX",    "18% above seasonal average",               "↑ +18%",        AMBER),
    ("NEXT MATCH",    "Man City (H)  ·  GW36",                    "Top-6 clash",   TEXT3),
    ("CONGESTION",    "3 fixtures in next 7 days",                "↑ Flagged",     AMBER),
]

sig_top = y; sig_bot = STRIP_Y - 20
row_h   = (sig_bot - sig_top) / len(signals)
vf = F("semibold", 20); df = F("semibold", 14)

for i, (label, value, delta, dcol) in enumerate(signals):
    sy = sig_top + int(i * row_h); ch = 14+6+22; pad = int((row_h-ch)/2); ry = sy+pad
    d.rectangle([RX-14, ry, RX-11, ry+ch], fill=(*LPOOL_RED, 90))
    lbl(label, ry, c=TEXT2)
    d.text((RX, ry+18), value, font=vf, fill=TEXT1)
    dw = int(d.textlength(delta, font=df))
    d.text((W-RM-dw, ry+20), delta, font=df, fill=dcol)
    if i < len(signals)-1:
        dr = sy+int(row_h)
        d.rectangle([RX, dr, W-RM, dr+1], fill=DIVIDER)

div(STRIP_Y)
d.text((RX, STRIP_Y+15), "yaraspeaks.com", font=F("medium", 13), fill=ACCENT)

# ← xMINS restored, other two stay plain English
strip = "47.3% of Fantasy managers own  ·  142k transferred in this week  ·  xMINS 68"
sf = F("medium", 13)
sw = int(d.textlength(strip, font=sf))
if sw > CW - 80: sf = F("medium", 12); sw = int(d.textlength(strip, font=sf))
d.text((W-RM-sw, STRIP_Y+15), strip, font=sf, fill=TEXT3)

canvas.save("/home/user/yara_risk_card.png", quality=97)
print("Single card done.")
