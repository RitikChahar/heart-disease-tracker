"""
generate_dashboard_screenshot.py
---------------------------------
Draws a pixel-accurate mockup of the Streamlit dashboard and saves it as
models/dashboard_screenshot.png  (1280 x 900 px, 96 dpi).

Run:  python generate_dashboard_screenshot.py
"""

from PIL import Image, ImageDraw, ImageFont
import os, textwrap

# ── Canvas ────────────────────────────────────────────────────────────────────
W, H = 1280, 960
img = Image.new("RGB", (W, H), "#ffffff")
draw = ImageDraw.Draw(img)

# ── Colour palette ────────────────────────────────────────────────────────────
BG         = "#f0f2f6"   # Streamlit default page bg
SIDEBAR_BG = "#262730"   # dark sidebar
WHITE      = "#ffffff"
BORDER     = "#d1d5db"
TEXT       = "#1f2328"
MUTED      = "#57606a"
ACCENT     = "#ff4b4b"   # Streamlit red
SUCCESS_BG = "#d4edda"
SUCCESS_FG = "#155724"
METRIC_BG  = "#e8f0fe"
LABEL      = "#374151"
INPUT_BG   = "#ffffff"
BTN_BG     = "#ff4b4b"
BTN_FG     = "#ffffff"

# ── Fonts  (fallback to default if no system font found) ──────────────────────
def font(size, bold=False):
    candidates = (
        ["C:/Windows/Fonts/segoeui.ttf",  "C:/Windows/Fonts/arial.ttf"]   if not bold else
        ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"]
    )
    for p in candidates:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

F_TITLE   = font(26, bold=True)
F_H1      = font(18, bold=True)
F_H2      = font(14, bold=True)
F_BODY    = font(13)
F_SMALL   = font(11)
F_LABEL   = font(12, bold=True)
F_SIDEBAR = font(12)
F_CAPTION = font(11)
F_BTN     = font(14, bold=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def rect(x1, y1, x2, y2, fill, radius=6, outline=None):
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill,
                            outline=outline or fill)

def text(x, y, s, f, color=TEXT, anchor="lt"):
    draw.text((x, y), s, font=f, fill=color, anchor=anchor)

def hline(y, x1=0, x2=W, color=BORDER):
    draw.line([(x1, y), (x2, y)], fill=color, width=1)

def input_box(x, y, w, label, value, h=34):
    text(x, y, label, F_LABEL, LABEL)
    rect(x, y+18, x+w, y+18+h, INPUT_BG, radius=4, outline=BORDER)
    text(x+8, y+18+h//2, value, F_BODY, MUTED, anchor="lm")

def select_box(x, y, w, label, value, h=34):
    text(x, y, label, F_LABEL, LABEL)
    rect(x, y+18, x+w, y+18+h, INPUT_BG, radius=4, outline=BORDER)
    text(x+8, y+18+h//2, value, F_BODY, TEXT, anchor="lm")
    # dropdown arrow
    ax = x+w-14
    ay = y+18+h//2
    draw.polygon([(ax-5, ay-3), (ax+5, ay-3), (ax, ay+4)], fill=MUTED)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  (left 230 px)
# ═══════════════════════════════════════════════════════════════════════════════
SIDE_W = 230
rect(0, 0, SIDE_W, H, SIDEBAR_BG, radius=0)

# Streamlit ">" collapse button
rect(SIDE_W-12, H//2-18, SIDE_W+2, H//2+18, "#3d3f4e", radius=4)
text(SIDE_W-9, H//2, "›", font(14), "#aab", anchor="lm")

sy = 24
text(14, sy, "Feature Reference", F_H2, "#e5e7eb"); sy += 28

rows = [
    ("age",      "Age in years"),
    ("sex",      "0 = Female, 1 = Male"),
    ("cp",       "Chest pain type (0–3)"),
    ("trestbps", "Resting BP (mm Hg)"),
    ("chol",     "Cholesterol (mg/dl)"),
    ("fbs",      "Fasting blood sugar >120"),
    ("restecg",  "Resting ECG (0–2)"),
    ("thalach",  "Max heart rate"),
    ("exang",    "Exercise angina (0/1)"),
    ("oldpeak",  "ST depression"),
    ("slope",    "ST slope (0–2)"),
    ("ca",       "Major vessels (0–3)"),
    ("thal",     "Thalassemia (1–3)"),
]
for feat, desc in rows:
    text(14, sy, feat, F_CAPTION, "#93c5fd")
    text(90, sy, desc, F_CAPTION, "#9ca3af")
    sy += 17

hline(sy+6, x1=14, x2=SIDE_W-14, color="#3d3f4e"); sy += 18
text(14, sy, "Dataset: UCI Heart Disease", F_CAPTION, "#6b7280"); sy += 15
text(14, sy, "Model: sklearn Pipeline",    F_CAPTION, "#6b7280"); sy += 15
text(14, sy, "Academic use only",          F_CAPTION, "#6b7280")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
MX = SIDE_W + 32   # main area left margin
MW = W - MX - 32   # main area width

# Page background
rect(SIDE_W, 0, W, H, BG, radius=0)

# Top bar (thin white strip)
rect(SIDE_W, 0, W, 54, WHITE, radius=0)
hline(54, x1=SIDE_W, color=BORDER)

# App icon + title
text(MX, 14, "🫀", font(22), anchor="lt")
text(MX+36, 12, "Heart Disease Prediction System", F_TITLE, TEXT)
text(MX, 44, "Enter the patient's clinical measurements below and click  Predict  to see the model's assessment.", F_SMALL, MUTED)

hline(60, x1=SIDE_W, color=BORDER)

# ── Section heading ───────────────────────────────────────────────────────────
y0 = 72
text(MX, y0, "Patient Information", F_H1, TEXT); y0 += 28

# ── Three-column input layout ─────────────────────────────────────────────────
COL_W  = (MW - 24) // 3
C1, C2, C3 = MX, MX + COL_W + 12, MX + (COL_W + 12)*2

ROW_H = 62   # height per input row

# Column 1
input_box (C1, y0,        COL_W, "Age (years)",                    "50")
input_box (C1, y0+ROW_H,  COL_W, "Resting Blood Pressure (mm Hg)", "120")
input_box (C1, y0+ROW_H*2, COL_W, "Cholesterol (mg/dl)",           "200")
input_box (C1, y0+ROW_H*3, COL_W, "Max Heart Rate Achieved",       "150")
input_box (C1, y0+ROW_H*4, COL_W, "ST Depression (oldpeak)",       "1.0")

# Column 2
select_box(C2, y0,         COL_W, "Sex",                            "Male (1)")
select_box(C2, y0+ROW_H,   COL_W, "Fasting Blood Sugar > 120 mg/dl","No (0)")
select_box(C2, y0+ROW_H*2, COL_W, "Exercise-Induced Angina",        "No (0)")
select_box(C2, y0+ROW_H*3, COL_W, "Resting ECG Results",            "Normal (0)")

# Column 3
select_box(C3, y0,         COL_W, "Chest Pain Type",                "Typical angina (0)")
select_box(C3, y0+ROW_H,   COL_W, "Slope of Peak ST Segment",       "Flat (1)")
select_box(C3, y0+ROW_H*2, COL_W, "Major Vessels (Fluoroscopy)",    "0 vessels")
select_box(C3, y0+ROW_H*3, COL_W, "Thalassemia",                    "Normal (1)")

# ── Divider ───────────────────────────────────────────────────────────────────
div_y = y0 + ROW_H*5 + 14
hline(div_y, x1=MX, x2=W-32)

# ── Predict button ────────────────────────────────────────────────────────────
btn_y  = div_y + 14
btn_h  = 42
rect(MX, btn_y, MX + MW, btn_y + btn_h, BTN_BG, radius=6)
text(MX + MW//2, btn_y + btn_h//2, "🔍  Predict", F_BTN, BTN_FG, anchor="mm")

# ── Divider ───────────────────────────────────────────────────────────────────
hline(btn_y + btn_h + 16, x1=MX, x2=W-32)

# ── Prediction Result panel ───────────────────────────────────────────────────
res_y = btn_y + btn_h + 26
text(MX, res_y, "Prediction Result", F_H1, TEXT); res_y += 28

# Success box
sbox_h = 52
rect(MX, res_y, MX + MW, res_y + sbox_h, SUCCESS_BG, radius=6, outline="#c3e6cb")
text(MX + 16, res_y + sbox_h//2 - 10, "✅  No Heart Disease Detected", F_H2, SUCCESS_FG, anchor="lt")
text(MX + 16, res_y + sbox_h//2 + 8,  "The model predicts that this patient likely does not have heart disease.", F_SMALL, SUCCESS_FG)
res_y += sbox_h + 14

# Metric + progress row
metric_w = 190
rect(MX, res_y, MX + metric_w, res_y + 56, METRIC_BG, radius=6)
text(MX + 12, res_y + 10, "Probability of Heart Disease", F_SMALL, MUTED)
text(MX + 12, res_y + 28, "21.0%", font(20, bold=True), TEXT)

prog_x = MX + metric_w + 18
prog_y = res_y + 22
prog_w = MW - metric_w - 18
prog_h = 12
rect(prog_x, prog_y, prog_x + prog_w, prog_y + prog_h, "#e5e7eb", radius=6)
filled = int(prog_w * 0.21)
rect(prog_x, prog_y, prog_x + filled, prog_y + prog_h, "#3b82d4", radius=6)
res_y += 72

text(MX, res_y, "Model used:  Random Forest", F_CAPTION, MUTED); res_y += 22

hline(res_y, x1=MX, x2=W-32); res_y += 12

# ── Disclaimer info box ───────────────────────────────────────────────────────
info_h = 76
rect(MX, res_y, MX + MW, res_y + info_h, "#dbeafe", radius=6, outline="#bfdbfe")
text(MX + 14, res_y + 10, "ℹ  Educational Disclaimer", F_H2, "#1e40af")
disc = ("This prediction is generated by a machine learning model for academic and educational purposes only. "
        "It is not a medical diagnosis and must not be used as a substitute for professional medical advice, "
        "examination, or treatment. Always consult a qualified healthcare professional for any health concerns.")
lines = textwrap.wrap(disc, width=115)
for i, ln in enumerate(lines[:3]):
    text(MX + 14, res_y + 30 + i*14, ln, F_CAPTION, "#1e3a8a")

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join("models", "dashboard_screenshot.png")
img.save(out, dpi=(96, 96))
print(f"Saved: {out}  ({W}x{H} px)")
