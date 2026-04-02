"""
frontend/app.py — Space-themed Streamlit dashboard for JWST Spectra Pipeline.

Usage:
    cd Spectra_Analisys
    streamlit run frontend/app.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import io_fits as _io_fits
    _HAS_IO_FITS = True
except Exception:
    _HAS_IO_FITS = False

try:
    from config import MOLECULE_BANDS, SNR_VALID_BAND
except Exception:
    MOLECULE_BANDS = {}
    SNR_VALID_BAND = 2.0

st.set_page_config(
    page_title="Spectra · JWST",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — SPACE THEME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Animations ── */
@keyframes fadeInUp   { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
@keyframes shimmer    { 0%{background-position:-300% center} 100%{background-position:300% center} }
@keyframes pulse-glow { 0%,100%{opacity:1} 50%{opacity:.45} }
@keyframes scan       { 0%{top:-2px;opacity:0} 20%{opacity:1} 80%{opacity:1} 100%{top:100%;opacity:0} }
@keyframes orbit      { from{transform:rotate(0deg) translateX(24px) rotate(0deg)}
                          to{transform:rotate(360deg) translateX(24px) rotate(-360deg)} }
@keyframes borderFlow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

/* ── Base ── */
html,body,[class*="css"] { font-family:'Space Grotesk','Segoe UI',sans-serif; }

.main, section.main {
  background:
    radial-gradient(ellipse 90% 50% at 5%  10%, rgba(139,92,246,.09) 0%, transparent 65%),
    radial-gradient(ellipse 60% 40% at 95% 90%, rgba(0,212,255,.06)  0%, transparent 60%),
    radial-gradient(ellipse 40% 30% at 50% 50%, rgba(16,185,129,.03) 0%, transparent 55%),
    #04080f;
  min-height:100vh;
}
.block-container { padding:1rem 2rem 3rem; max-width:100%; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#080f20 0%,#04080f 100%);
  border-right: 1px solid rgba(0,212,255,.10);
  position: relative;
}
[data-testid="stSidebar"]::before {
  content:'';
  position:absolute; top:0;left:0;right:0;height:280px;
  background:radial-gradient(ellipse at 50% -10%, rgba(139,92,246,.18) 0%, transparent 70%);
  pointer-events:none;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label { color:#3d4f6e !important; font-size:11px !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
  background: rgba(14,22,42,.85);
  border: 1px solid rgba(255,255,255,.06);
  border-radius: 12px;
  padding: 16px 18px;
  backdrop-filter: blur(12px);
  position: relative;
  overflow: hidden;
  animation: fadeInUp .4s ease both;
}
[data-testid="metric-container"]::after {
  content:'';
  position:absolute; top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(0,212,255,.35),transparent);
}
[data-testid="metric-container"] label {
  color:#3d5070 !important;
  font-size:10px !important;
  text-transform:uppercase;
  letter-spacing:1.2px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color:#f0f6ff !important;
  font-size:24px !important;
  font-weight:700;
  font-family:'Space Grotesk',sans-serif;
}

/* ── Buttons ── */
.stButton>button {
  background: linear-gradient(135deg,rgba(0,212,255,.12),rgba(139,92,246,.12));
  border: 1px solid rgba(0,212,255,.28);
  border-radius: 8px;
  color: #00d4ff;
  font-family: 'Space Grotesk',sans-serif;
  font-weight: 600;
  font-size: 13px;
  letter-spacing: .3px;
  transition: all .2s;
}
.stButton>button:hover {
  background: linear-gradient(135deg,rgba(0,212,255,.22),rgba(139,92,246,.22));
  border-color: rgba(0,212,255,.55);
  box-shadow: 0 0 22px rgba(0,212,255,.18);
  transform: translateY(-1px);
}
.stButton>button[kind="primary"] {
  background: linear-gradient(135deg,#0ea5e9,#8b5cf6);
  border: none; color:#fff;
}
.stButton>button[kind="primary"]:hover {
  box-shadow: 0 0 32px rgba(14,165,233,.40), 0 4px 16px rgba(0,0,0,.3);
  transform: translateY(-2px);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap:4px; }
.stTabs [data-baseweb="tab"] {
  background: rgba(14,22,42,.8);
  border: 1px solid rgba(255,255,255,.05);
  border-radius: 8px;
  padding: 6px 18px;
  color: #8b9dc3;
  font-size: 13px;
  transition: all .2s;
}
.stTabs [data-baseweb="tab"]:hover { border-color:rgba(0,212,255,.3); color:#00d4ff; }
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg,rgba(0,212,255,.14),rgba(139,92,246,.14)) !important;
  border-color: rgba(0,212,255,.40) !important;
  color: #00d4ff !important;
  box-shadow: 0 0 14px rgba(0,212,255,.12) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(255,255,255,.06) !important;
  border-radius: 12px !important;
  overflow: hidden;
  backdrop-filter: blur(10px);
}
[data-testid="stDataFrame"] th {
  background: rgba(0,212,255,.05) !important;
  color: #3d5070 !important;
  font-size: 10px !important;
  text-transform: uppercase;
  letter-spacing: 1px;
  border-color: rgba(255,255,255,.04) !important;
  font-family: 'Space Grotesk',sans-serif !important;
}
[data-testid="stDataFrame"] td {
  color: #c8d8f0 !important;
  border-color: rgba(255,255,255,.03) !important;
  font-family: 'JetBrains Mono',monospace !important;
  font-size: 12px !important;
}
[data-testid="stDataFrame"] tr:hover td { background:rgba(0,212,255,.04) !important; }

/* ── Expanders ── */
details {
  background: rgba(14,22,42,.75) !important;
  border: 1px solid rgba(255,255,255,.06) !important;
  border-radius: 10px !important;
  backdrop-filter: blur(8px);
}
details>summary { color:#8b9dc3 !important; font-size:13px !important; padding:10px 14px !important; }
details>summary:hover { color:#00d4ff !important; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {
  background: rgba(14,22,42,.9) !important;
  border: 1px solid rgba(255,255,255,.08) !important;
  border-radius: 8px !important;
  color: #d1daf0 !important;
  font-family: 'Space Grotesk',sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: rgba(0,212,255,.45) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,.10) !important;
}
.stSelectbox>div>div, .stMultiSelect>div>div {
  background: rgba(14,22,42,.9) !important;
  border: 1px solid rgba(255,255,255,.08) !important;
  border-radius: 8px !important;
  color: #d1daf0 !important;
}
.stRadio label { color:#8b9dc3 !important; font-size:13px !important; }
.stCheckbox label { color:#8b9dc3 !important; font-size:13px !important; }
.stSlider label  { color:#3d5070 !important; font-size:11px !important; }

/* ── Code ── */
pre,code {
  font-family:'JetBrains Mono',monospace !important;
  background:rgba(8,15,32,.95) !important;
  border:1px solid rgba(255,255,255,.05) !important;
  border-radius:8px !important;
  color:#a5d6ff !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
  background: rgba(14,22,42,.8) !important;
  border-radius: 10px !important;
  border-left: 3px solid #00d4ff !important;
  backdrop-filter: blur(8px);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#04080f; }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg,#0ea5e9,#8b5cf6);
  border-radius:3px;
}

/* ── Caption ── */
.stCaption, [data-testid="stCaptionContainer"] { color:#3d5070 !important; font-size:11px !important; }

/* ── Divider ── */
hr { border-color:rgba(255,255,255,.04); margin:16px 0; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color:#00d4ff !important; }
[data-testid="stSpinner"] p { color:#8b9dc3 !important; }

/* ── Custom component classes ── */
.page-title {
  font-size:27px; font-weight:800; letter-spacing:-0.5px;
  background:linear-gradient(115deg,#00d4ff 0%,#8b5cf6 45%,#e0f2fe 100%);
  background-size:300% auto;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  animation:shimmer 6s linear infinite;
  margin-bottom:4px;
}
.subtitle {
  color:#1e3347; font-size:12px; margin-bottom:22px; letter-spacing:.3px;
  font-family:'JetBrains Mono',monospace;
}
.section-label {
  font-size:10px; color:#1e3347; letter-spacing:2px;
  text-transform:uppercase; margin-bottom:10px; font-weight:600;
}
.glass {
  background:rgba(14,22,42,.75);
  border:1px solid rgba(255,255,255,.06);
  border-radius:14px; padding:18px;
  backdrop-filter:blur(14px);
  position:relative; overflow:hidden;
  animation:fadeInUp .45s ease both;
}
.glass::after {
  content:'';
  position:absolute; top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);
}
.glow-n {
  font-family:'Space Grotesk',sans-serif;
  font-size:34px; font-weight:800; line-height:1; letter-spacing:-1px;
}
.badge {
  display:inline-block; padding:2px 11px; border-radius:20px;
  font-size:9px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase;
}
.stat-row {
  display:flex; justify-content:space-between; align-items:center;
  padding:6px 0; border-bottom:1px solid rgba(255,255,255,.04); font-size:12px;
}
.sl  { color:#1e3347; }
.sv  { color:#c8d8f0; font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:500; }
.warn-box {
  background:rgba(251,191,36,.06); border:1px solid rgba(251,191,36,.22);
  border-radius:8px; padding:9px 14px; color:#fbbf24; font-size:11px;
  font-family:'JetBrains Mono',monospace; margin-top:8px;
}
.scan {
  position:absolute; left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,rgba(0,212,255,.7),transparent);
  animation:scan 3.5s ease-in-out infinite; pointer-events:none;
}
.dot-live {
  display:inline-block; width:8px; height:8px; border-radius:50%;
  background:#10b981; box-shadow:0 0 8px #10b981;
  animation:pulse-glow 2s ease infinite;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
OUTPUTS_DIR = ROOT / "outputs"
DATA_INPUTS  = ROOT / "data" / "inputs"
CONFIG_PATH  = ROOT / "config.py"

MOL_COLOR = {
    "H2O": "#00d4ff",  "CH4": "#fbbf24",
    "CO":  "#f87171",  "CO2": "#a78bfa",
    "O2":  "#34d399",  "O3":  "#2dd4bf",
}
MOL_FILL = {k: f"rgba({int(v[1:3],16)},{int(v[3:5],16)},{int(v[5:7],16)},0.08)"
            for k, v in MOL_COLOR.items()}

STATUS_COLOR = {
    "STRONG":               "#34d399",
    "LIKELY":               "#86efac",
    "MARGINAL":             "#fbbf24",
    "WEAK":                 "#fb923c",
    "NOT DETECTED":         "#374151",
    "NO SPECTRAL COVERAGE": "#1d4ed8",
    "NO COVERAGE":          "#1d4ed8",
}
OBJ_ICONS = {"Y":"❄️","T":"🌑","L":"🟤","M":"🔴","FGK":"⭐","Hot":"🌟","?":"❓"}

PAGES = {
    "overview":  ("◈", "Обзор"),
    "spectrum":  ("◉", "Спектр"),
    "molecules": ("◎", "Молекулы"),
    "report":    ("◻", "Отчёт"),
    "run":       ("▶", "Запуск"),
    "history":   ("≡", "История"),
    "compare":   ("⊞", "Сравнение"),
    "settings":  ("◧", "Настройки"),
}

_DARK = dict(
    paper_bgcolor="#04080f", plot_bgcolor="#070d1a",
    font=dict(color="#8b9dc3", family="Space Grotesk"),
    margin=dict(l=55, r=20, t=48, b=42),
)
_GRID = dict(gridcolor="#0d1b2e", zerolinecolor="#0d1b2e",
             tickfont=dict(color="#2d4060", size=10))

if "page" not in st.session_state:
    st.session_state.page = "overview"


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=30)
def list_targets():
    if not OUTPUTS_DIR.exists(): return []
    return sorted(d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir())

@st.cache_data(ttl=30)
def list_runs(t):
    d = OUTPUTS_DIR / t
    if not d.exists(): return []
    return sorted((r.name for r in d.iterdir() if r.is_dir()), reverse=True)

@st.cache_data(ttl=60)
def load_results(t, r):
    p = OUTPUTS_DIR / t / r / "reports" / "results.json"
    return json.load(open(p)) if p.exists() else None

@st.cache_data(ttl=60)
def load_summary(t, r):
    p = OUTPUTS_DIR / t / r / "reports" / "summary.txt"
    return p.read_text() if p.exists() else ""

@st.cache_data(ttl=60)
def load_spectrum(fits_path):
    if not _HAS_IO_FITS: return None
    try:    return _io_fits.read_spectrum(fits_path)
    except: return None

@st.cache_data(ttl=120)
def all_runs_df():
    rows = []
    for t in list_targets():
        for r in list_runs(t):
            d = load_results(t, r) or {}
            det = d.get("detections", {})
            mols = [m for m, i in det.items() if i.get("detected")]
            q = (d.get("ml_analysis") or {}).get("quality") or {}
            rows.append({
                "Target":   t, "Run ID": r,
                "T (K)":    round(d.get("continuum",{}).get("T_K") or 0),
                "Detected": ", ".join(mols) if mols else "—",
                "N":        len(mols),
                "Quality":  round(q.get("quality_score", 0), 2),
                "Fit OK":   d.get("continuum",{}).get("fit_ok", False),
            })
    return pd.DataFrame(rows)


# ── Config ────────────────────────────────────────────────────────────────────
def read_config():
    out = {}
    if not CONFIG_PATH.exists(): return out
    for line in CONFIG_PATH.read_text().splitlines():
        m = re.match(r'^([A-Z][A-Z0-9_]+)\s*=\s*([0-9]+\.?[0-9]*)\s*(?:#.*)?$', line.strip())
        if m: out[m.group(1)] = float(m.group(2))
    return out

def write_config(key, val):
    c = CONFIG_PATH.read_text()
    CONFIG_PATH.write_text(re.sub(
        rf'^({re.escape(key)}\s*=\s*)[0-9]+\.?[0-9]*', rf'\g<1>{val}',
        c, flags=re.MULTILINE))

def classify(T_K):
    if T_K <= 0:   return "?",   "Неизвестно",  "T не определена"
    if T_K < 500:  return "Y",   "Y-карлик",    f"Очень холодный субзвёздный ({T_K:.0f} K)"
    if T_K < 1400: return "T",   "T-карлик",    f"Холодный субзвёздный ({T_K:.0f} K)"
    if T_K < 2300: return "L",   "L-карлик",    f"Тёплый субзвёздный ({T_K:.0f} K)"
    if T_K < 3900: return "M",   "M-карлик",    f"Красная звезда ({T_K:.0f} K)"
    if T_K < 7500: return "FGK", "FGK-звезда",  f"Солнцеподобная ({T_K:.0f} K)"
    return "Hot",  "Горячая звезда", f"T > 7500 K ({T_K:.0f} K)"


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def spectrum_fig(w, flux, cont, res, err, det, title,
                 show_mols=None, show_cont=True, show_err=True,
                 show_labels=True, log_y=False, w_range=None):

    if show_mols is None: show_mols = list(MOLECULE_BANDS)

    if w_range:
        m = (w >= w_range[0]) & (w <= w_range[1])
        w, flux = w[m], flux[m]
        cont = cont[m] if cont is not None and len(cont)==len(w) else None
        res  = res[m]  if res  is not None and len(res) ==len(w) else None
        err  = err[m]  if err  is not None and len(err) ==len(w) else None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.02)

    seen = set()
    for mol in show_mols:
        fill = MOL_FILL.get(mol, "rgba(100,100,100,0.07)")
        for bd in MOLECULE_BANDS.get(mol, []):
            bs, be = bd["band"]
            lbl = mol if mol not in seen and show_labels else ""
            for row in (1, 2):
                fig.add_vrect(x0=bs, x1=be, fillcolor=fill, opacity=1,
                              layer="below", line_width=0,
                              annotation_text=lbl if row==1 else "",
                              annotation_position="top left",
                              annotation_font_size=9,
                              annotation_font_color=MOL_COLOR.get(mol,"#aaa"),
                              row=row, col=1)
            seen.add(mol)

    if show_err and err is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([w, w[::-1]]),
            y=np.concatenate([flux+err, (flux-err)[::-1]]),
            fill="toself", fillcolor="rgba(139,92,246,0.07)",
            line=dict(width=0), name="±1σ", hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=w, y=flux, mode="lines", name="Flux",
        line=dict(color="#c8d8f0", width=1.0),
        hovertemplate="λ=%{x:.4f} μm<br>F=%{y:.4e}<extra></extra>",
    ), row=1, col=1)

    if show_cont and cont is not None:
        fig.add_trace(go.Scatter(
            x=w, y=cont, mode="lines", name="Континуум",
            line=dict(color="#fbbf24", width=1.8, dash="dash"),
            hovertemplate="λ=%{x:.4f}<br>Cont=%{y:.4e}<extra></extra>",
        ), row=1, col=1)

    if res is not None:
        fig.add_trace(go.Scatter(
            x=w, y=np.asarray(res), mode="lines", name="Residual",
            line=dict(color="#2d4060", width=1.0),
            hovertemplate="λ=%{x:.4f}<br>F/F_c=%{y:.4f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=1.0, line_dash="dot", line_color="#0d1b2e", line_width=1, row=2, col=1)

    for mol, info in det.items():
        if not info.get("detected") or mol not in show_mols: continue
        lc = MOL_COLOR.get(mol, "#fff")
        for b in info.get("bands", []):
            snr = b.get("snr", 0)
            if not b.get("covered") or snr < SNR_VALID_BAND: continue
            rng = b.get("band_range", (0,0))
            cx = (rng[0]+rng[1])/2
            if w_range and not (w_range[0] <= cx <= w_range[1]): continue
            fig.add_vline(x=cx, line_color=lc, line_width=1.2,
                          annotation_text=f"{mol} {snr:.1f}σ",
                          annotation_position="top",
                          annotation_font_size=8, annotation_font_color=lc,
                          row=1, col=1)

    if log_y: fig.update_yaxes(type="log", row=1, col=1)

    fig.update_layout(
        **_DARK,
        title=dict(text=f"<span style='color:#8b9dc3;font-size:13px;'>{title}</span>",
                   font=dict(size=13), x=0.01),
        hovermode="x unified", height=620,
        legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color="#8b9dc3")),
    )
    fig.update_xaxes(title_text="Длина волны (μm)", row=2, col=1, **_GRID)
    fig.update_xaxes(**_GRID, row=1, col=1)
    fig.update_yaxes(title_text="Flux",      row=1, col=1, **_GRID)
    fig.update_yaxes(title_text="F / F_cont", row=2, col=1, **_GRID)
    return fig


def snr_bar_fig(det):
    labels, vals, colors = [], [], []
    for mol, info in det.items():
        for b in info.get("bands", []):
            if not b.get("covered"): continue
            rng = b.get("band_range",(0,0))
            labels.append(f"{mol}  {rng[0]:.2f}–{rng[1]:.2f}")
            vals.append(b.get("snr", 0))
            colors.append(MOL_COLOR.get(mol,"#6b7280"))
    if not labels: return go.Figure()
    order = np.argsort(vals)[::-1]
    fig = go.Figure(go.Bar(
        x=[vals[i] for i in order], y=[labels[i] for i in order],
        orientation="h", marker_color=[colors[i] for i in order],
        marker=dict(line=dict(width=0)),
        hovertemplate="%{y}<br>SNR=%{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=SNR_VALID_BAND, line_dash="dot", line_color="#fbbf24", line_width=1,
                  annotation_text=f"≥{SNR_VALID_BAND:.0f}", annotation_font_size=8,
                  annotation_font_color="#fbbf24")
    fig.update_layout(**_DARK, height=max(220, len(labels)*30+60),
                      xaxis=dict(title="SNR", **_GRID),
                      yaxis=dict(**_GRID, autorange="reversed"),
                      title=dict(text="<span style='color:#8b9dc3'>SNR по полосам</span>",
                                 font=dict(size=12)),
                      margin=dict(l=175, r=20, t=44, b=38))
    return fig


def fap_bar_fig(det):
    labels, vals, colors = [], [], []
    for mol, info in det.items():
        for b in info.get("bands", []):
            if not b.get("covered"): continue
            rng = b.get("band_range",(0,0))
            fap = b.get("fap", 1.0)
            labels.append(f"{mol}  {rng[0]:.2f}–{rng[1]:.2f}")
            vals.append(-np.log10(max(fap, 1e-20)))
            colors.append(MOL_COLOR.get(mol,"#6b7280"))
    if not labels: return go.Figure()
    order = np.argsort(vals)[::-1]
    fig = go.Figure(go.Bar(
        x=[vals[i] for i in order], y=[labels[i] for i in order],
        orientation="h", marker_color=[colors[i] for i in order],
        marker=dict(line=dict(width=0)),
        hovertemplate="%{y}<br>-log₁₀FAP=%{x:.2f}<extra></extra>",
    ))
    for xv, lbl, col in [(1.3,"p<0.05","#fbbf24"),(2.0,"p<0.01","#fb923c"),
                          (3.0,"3σ","#f87171"),(5.0,"5σ","#ef4444")]:
        fig.add_vline(x=xv, line_dash="dot", line_color=col, line_width=1,
                      annotation_text=lbl, annotation_font_size=8, annotation_font_color=col)
    fig.update_layout(**_DARK, height=max(220, len(labels)*30+60),
                      xaxis=dict(title="-log₁₀(FAP)", **_GRID),
                      yaxis=dict(**_GRID, autorange="reversed"),
                      title=dict(text="<span style='color:#8b9dc3'>Статистическая значимость</span>",
                                 font=dict(size=12)),
                      margin=dict(l=175, r=20, t=44, b=38))
    return fig


def quality_gauge(score, usable):
    c = "#34d399" if usable else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, domain={"x":[0,1],"y":[0,1]},
        title={"text":"Data Quality","font":{"size":11,"color":"#3d5070"}},
        gauge={
            "axis":{"range":[0,1],"tickcolor":"#0d1b2e","tickfont":{"color":"#2d4060","size":9}},
            "bar":{"color":c,"thickness":0.65},
            "bgcolor":"#070d1a","borderwidth":0,
            "steps":[{"range":[0,.4],"color":"#070d1a"},
                     {"range":[.4,.7],"color":"#080f20"},
                     {"range":[.7,1.],"color":"#091225"}],
            "threshold":{"line":{"color":"#fbbf24","width":2},"thickness":.75,"value":.4},
        },
        number={"valueformat":".2f","font":{"color":"#f0f6ff","size":28}},
    ))
    fig.update_layout(paper_bgcolor="#04080f", font=dict(color="#8b9dc3"),
                      height=200, margin=dict(l=12,r=12,t=52,b=4))
    return fig


def heatmap_fig(det):
    mols = list(det.keys())
    bands_set = {}
    for mol, info in det.items():
        for b in info.get("bands",[]):
            rng = b.get("band_range",(0,0))
            k = f"{rng[0]:.2f}–{rng[1]:.2f}"
            bands_set[k] = (rng[0]+rng[1])/2
    band_labels = sorted(bands_set, key=lambda k: bands_set[k])

    z, text = [], []
    for mol in mols:
        row_z, row_t = [], []
        for bl in band_labels:
            match = next((b for b in det[mol].get("bands",[])
                          if f"{b.get('band_range',(0,0))[0]:.2f}–{b.get('band_range',(0,0))[1]:.2f}"==bl), None)
            if match is None:          row_z.append(-1);   row_t.append("—")
            elif not match.get("covered"): row_z.append(0);  row_t.append("n/a")
            else:
                snr = match.get("snr",0)
                row_z.append(min(snr/10,1.)); row_t.append(f"{snr:.1f}σ")
        z.append(row_z); text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=band_labels, y=mols,
        text=text, texttemplate="%{text}",
        textfont=dict(size=10, family="JetBrains Mono"),
        colorscale=[[0,"#04080f"],[0.001,"#0d1b2e"],[.3,"#1d4ed8"],[.65,"#059669"],[1.,"#34d399"]],
        zmin=-1, zmax=1, showscale=False,
        hovertemplate="<b>%{y}</b> %{x}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(**_DARK, height=max(170, len(mols)*36+80),
                      xaxis=dict(title="", tickangle=-35, tickfont=dict(size=10,color="#2d4060",
                                 family="JetBrains Mono"), gridcolor="#0d1b2e"),
                      yaxis=dict(tickfont=dict(size=11,color="#8b9dc3"), gridcolor="#0d1b2e"),
                      title=dict(text="<span style='color:#8b9dc3'>Покрытие и SNR полос</span>",
                                 font=dict(size=12)),
                      margin=dict(l=65,r=16,t=44,b=60))
    return fig


def radar_fig(rdata, mols):
    colors = ["#00d4ff","#34d399","#fbbf24","#a78bfa"]
    fig = go.Figure()
    theta = mols + [mols[0]]
    for i, (label, d) in enumerate(rdata.items()):
        vals = [d["detections"].get(m,{}).get("confidence",0.) for m in mols]
        vals += [vals[0]]
        c = colors[i % 4]
        fig.add_trace(go.Scatterpolar(r=vals, theta=theta, fill="toself",
                                      name=label[:30],
                                      line=dict(color=c, width=1.5),
                                      fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.08)"))
    fig.update_layout(
        paper_bgcolor="#04080f",
        polar=dict(bgcolor="#070d1a",
                   radialaxis=dict(visible=True, range=[0,1], color="#0d1b2e",
                                   tickfont=dict(color="#2d4060",size=9), gridcolor="#0d1b2e"),
                   angularaxis=dict(color="#2d4060", gridcolor="#0d1b2e",
                                    tickfont=dict(color="#8b9dc3", size=11))),
        font=dict(color="#8b9dc3",family="Space Grotesk"),
        title=dict(text="<span style='color:#8b9dc3'>Confidence Radar</span>", font=dict(size=12)),
        legend=dict(orientation="h", y=-0.12, font=dict(size=10,color="#8b9dc3")),
        height=370, margin=dict(l=25,r=25,t=52,b=50),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
<div style="padding:20px 4px 16px;border-bottom:1px solid rgba(255,255,255,.04);position:relative;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;
       background:linear-gradient(115deg,#00d4ff,#8b5cf6);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
    🔭 SPECTRA
  </div>
  <div style="font-size:9px;color:#1e3347;letter-spacing:3px;text-transform:uppercase;margin-top:3px;">
    JWST · MOLECULAR PIPELINE
  </div>
  <div style="position:absolute;top:22px;right:8px;" title="Online">
    <span class="dot-live"></span>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        targets = list_targets()
        if targets:
            sel_t = st.selectbox("ОБЪЕКТ", targets, key="sb_target")
            runs  = list_runs(sel_t)
            sel_r = st.selectbox("ЗАПУСК", runs, key="sb_run") if runs else None

            if sel_r:
                d   = load_results(sel_t, sel_r)
                det = (d or {}).get("detections", {})
                T_K = (d or {}).get("continuum", {}).get("T_K") or 0
                n_d = sum(1 for i in det.values() if i.get("detected"))
                obj_key, obj_label, _ = classify(T_K)
                icon = OBJ_ICONS.get(obj_key,"❓")

                mol_dots = "".join(
                    f"<span title='{mol}' style='display:inline-block;width:9px;height:9px;"
                    f"border-radius:50%;background:{MOL_COLOR.get(mol,'#374151') if info.get('detected') else '#0d1b2e'};"
                    f"box-shadow:{'0 0 6px '+MOL_COLOR.get(mol,'#374151') if info.get('detected') else 'none'};'>"
                    f"</span>" for mol, info in det.items()
                )
                st.markdown(
                    f"<div class='glass' style='margin:10px 0;padding:14px;'>"
                    f"<div class='section-label'>ОБЪЕКТ</div>"
                    f"<div style='font-size:14px;color:#c8d8f0;margin-bottom:6px;'>{icon} {obj_label}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#1e3347;margin-bottom:10px;'>"
                    f"T = <span style='color:#00d4ff;'>{T_K:.0f}</span> K &nbsp;·&nbsp;"
                    f"<span style='color:#34d399;font-weight:700;'>{n_d}</span>"
                    f"<span style='color:#1e3347;'>/{len(det)}</span> det</div>"
                    f"<div style='display:flex;gap:5px;align-items:center;'>{mol_dots}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div style='color:#1e3347;font-size:12px;padding:8px 0;'>Нет данных — запустите пайплайн</div>",
                        unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>НАВИГАЦИЯ</div>", unsafe_allow_html=True)

        for key, (icon, label) in PAGES.items():
            active = st.session_state.page == key
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if st.button("⟳  Сбросить кэш", use_container_width=True):
            st.cache_data.clear(); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════
def page_overview(data, sel_t, sel_r):
    det  = data.get("detections", {})
    cont = data.get("continuum", {})
    ml   = data.get("ml_analysis") or {}

    T_K     = cont.get("T_K") or 0
    chi2    = cont.get("chi2_reduced")
    fit_ok  = cont.get("fit_ok", False)
    n_det   = sum(1 for i in det.values() if i.get("detected"))
    q       = (ml.get("quality") or {})
    q_score = q.get("quality_score")
    usable  = q.get("usable", True)
    obj_key, obj_label, obj_desc = classify(T_K)
    icon    = OBJ_ICONS.get(obj_key,"❓")

    # Header
    col_h, col_b = st.columns([4, 1])
    with col_h:
        st.markdown(f'<div class="page-title">{icon} {sel_t}</div>'
                    f'<div class="subtitle">{sel_r} · {obj_desc}</div>', unsafe_allow_html=True)
    with col_b:
        c = MOL_COLOR.get("H2O","#00d4ff")
        st.markdown(f"""
<div style="text-align:right;margin-top:8px;">
  <span style="background:linear-gradient(135deg,rgba(0,212,255,.15),rgba(139,92,246,.15));
    border:1px solid rgba(0,212,255,.3);border-radius:20px;
    padding:5px 16px;font-size:12px;font-weight:700;color:#00d4ff;
    font-family:'Space Grotesk',sans-serif;letter-spacing:.5px;">
    {obj_label}
  </span>
</div>""", unsafe_allow_html=True)

    mc = st.columns(5)
    mc[0].metric("🌡️ Температура",   f"{T_K:.0f} K" if T_K else "N/A")
    mc[1].metric("🧬 Детекций",       f"{n_det} / {len(det)}")
    mc[2].metric("📈 Chi²",           f"{chi2:.2f}" if chi2 else "N/A")
    mc[3].metric("🔧 Континуум",      "✓ OK" if fit_ok else "⚠ Fallback")
    mc[4].metric("📋 Качество",       f"{q_score:.2f}" if q_score is not None else "N/A")

    st.markdown("---")
    st.markdown('<div class="section-label">СТАТУС ДЕТЕКЦИИ</div>', unsafe_allow_html=True)

    cols = st.columns(len(det))
    for i, (col, (mol, info)) in enumerate(zip(cols, det.items())):
        status  = info.get("status","NOT DETECTED")
        conf    = info.get("confidence", 0.)
        snr     = info.get("max_snr", 0.)
        fap     = info.get("fap", 1.)
        detected = info.get("detected", False)
        color   = STATUS_COLOR.get(status, "#374151")
        mc_hex  = MOL_COLOR.get(mol, "#6b7280")
        bw      = min(int(conf*100), 100)
        cov     = sum(1 for b in info.get("bands",[]) if b.get("covered"))
        tot     = len(info.get("bands",[]))
        fap_str = f"{fap:.1e}" if fap < 0.01 else f"{fap:.3f}"
        delay   = i * 0.06

        scan = '<div class="scan"></div>' if detected else ""

        with col:
            st.markdown(f"""
<div style="
  position:relative;overflow:hidden;
  background:linear-gradient(145deg,rgba(14,22,42,.90),rgba(7,13,26,.95));
  border:1px solid {mc_hex}22;
  border-radius:14px; padding:18px 12px; text-align:center;
  backdrop-filter:blur(14px);
  box-shadow:0 4px 24px rgba(0,0,0,.4),{'0 0 30px '+mc_hex+'12' if detected else 'none'};
  animation:fadeInUp .4s ease {delay:.2f}s both;
  transition:border-color .25s,box-shadow .25s;
  height:100%;
">
  {scan}
  <div style="position:absolute;top:-28px;right:-28px;width:80px;height:80px;
    background:radial-gradient(circle,{mc_hex}18 0%,transparent 70%);
    border-radius:50%;pointer-events:none;"></div>

  <div style="font-family:'JetBrains Mono',monospace;font-size:18px;
    font-weight:700;color:{mc_hex};letter-spacing:2px;">{mol}</div>
  <div style="margin:5px 0;">
    <span class="badge" style="background:{color}18;color:{color};border:1px solid {color}35;">
      {status}
    </span>
  </div>
  <div class="glow-n" style="color:{mc_hex if detected else '#1e3347'};margin:10px 0;
    {'text-shadow:0 0 24px '+mc_hex+'55;' if detected else ''}">
    {snr:.1f}<span style="font-size:13px;color:{mc_hex}60;font-weight:400;">σ</span>
  </div>
  <div style="background:rgba(255,255,255,.04);border-radius:6px;height:5px;
    margin:6px 0;overflow:hidden;">
    <div style="background:linear-gradient(90deg,{mc_hex}70,{mc_hex});
      width:{bw}%;height:100%;border-radius:6px;
      {'box-shadow:0 0 10px '+mc_hex+'60;' if detected else ''}
    "></div>
  </div>
  <div style="font-size:9px;color:#1e3347;margin-top:7px;
    font-family:'JetBrains Mono',monospace;line-height:1.7;">
    conf <span style="color:#8b9dc3;">{conf:.2f}</span>
    &nbsp;·&nbsp; FAP <span style="color:#8b9dc3;">{fap_str}</span><br>
    <span style="color:#0d1b2e;">{cov}/{tot} полос</span>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_hm, col_q = st.columns([3, 1])
    with col_hm:
        st.plotly_chart(heatmap_fig(det), use_container_width=True,
                        config={"displayModeBar": False})
    with col_q:
        if q_score is not None:
            st.plotly_chart(quality_gauge(q_score, usable), use_container_width=True,
                            config={"displayModeBar": False})
            for n in q.get("notes", []):
                st.caption(f"• {n}")

    conf_map = ml.get("confidence", {})
    if conf_map:
        with st.expander("💬 ML-интерпретация"):
            for mol, ci in conf_map.items():
                expl = ci.get("explanation","")
                lbl  = ci.get("label","")
                c    = STATUS_COLOR.get(lbl,"#374151")
                if expl:
                    st.markdown(
                        f"<span style='color:{MOL_COLOR.get(mol,c)};font-weight:700;"
                        f"font-family:JetBrains Mono,monospace;'>{mol}</span>"
                        f" <span style='color:#1e3347;'>—</span>"
                        f" <span style='color:#8b9dc3;font-size:13px;'>{expl}</span>",
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
def page_spectrum(data, sel_t, sel_r):
    det  = data.get("detections", {})
    cont = data.get("continuum", {})

    st.markdown('<div class="page-title">◉ Спектр</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_mols = st.multiselect("Молекулы",
            list(MOLECULE_BANDS), default=list(MOLECULE_BANDS))
    with c2:
        preset = st.selectbox("Диапазон", [
            "Полный","NIRSpec (0.6–5.3)","H-band (1.5–1.8)",
            "K-band (2.0–2.5)","L-band (3.0–4.2)",
            "CO₂ (4.0–4.7)","MIRI (5–12)",
        ])
    with c3:
        show_cont = st.checkbox("Континуум", True)
        show_err  = st.checkbox("±1σ", True)
    with c4:
        show_lbl = st.checkbox("Подписи", True)
        log_y    = st.checkbox("Log шкала")

    ranges = {
        "Полный":None,"NIRSpec (0.6–5.3)":(0.6,5.3),
        "H-band (1.5–1.8)":(1.5,1.8),"K-band (2.0–2.5)":(2.0,2.5),
        "L-band (3.0–4.2)":(3.0,4.2),"CO₂ (4.0–4.7)":(4.0,4.7),"MIRI (5–12)":(5.,12.),
    }

    fits_p = str(OUTPUTS_DIR/sel_t/sel_r/"input"/"spectrum_used.fits")
    spec   = load_spectrum(fits_p)

    if spec:
        w   = np.asarray(spec["wavelength_um"])
        flx = np.asarray(spec["flux"])
        err = np.asarray(spec["err"]) if spec.get("err") is not None else None
        c_arr = cont.get("continuum"); r_arr = cont.get("residual")
        c_np = np.asarray(c_arr) if c_arr else None
        r_np = np.asarray(r_arr) if r_arr else None
        fig  = spectrum_fig(w, flx, c_np, r_np, err, det,
                            f"{sel_t} · {sel_r}", show_mols, show_cont,
                            show_err, show_lbl, log_y, ranges.get(preset))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":"hover"})

        wr = ranges.get(preset)
        n_pts = int(((w>=wr[0])&(w<=wr[1])).sum()) if wr else len(w)
        ci = st.columns(4)
        ci[0].caption(f"Точек: **{n_pts}**")
        ci[1].caption(f"λ: **{w.min():.3f}–{w.max():.3f} μm**")
        ci[2].caption(f"T_cont: **{cont.get('T_K',0):.0f} K**")
        ci[3].caption(f"`{sel_r}`")
    else:
        png = OUTPUTS_DIR/sel_t/sel_r/"plots"/"spectrum.png"
        if png.exists():
            st.warning("FITS недоступен — показан PNG")
            st.image(str(png), use_container_width=True)
        else:
            st.error("Спектр недоступен")

    st.markdown("---")
    st.markdown('<div class="section-label">ЛЕГЕНДА</div>', unsafe_allow_html=True)
    lc = st.columns(len(MOLECULE_BANDS))
    for col, (mol, bds) in zip(lc, MOLECULE_BANDS.items()):
        c = MOL_COLOR.get(mol,"#6b7280")
        ranges_str = "<br>".join(f"{b['band'][0]:.2f}–{b['band'][1]:.2f}" for b in bds[:4])
        with col:
            st.markdown(
                f"<div style='border-left:3px solid {c};padding-left:8px;'>"
                f"<span style='color:{c};font-weight:700;font-family:JetBrains Mono,monospace;font-size:13px;'>{mol}</span>"
                f"<div style='font-size:10px;color:#1e3347;margin-top:4px;font-family:JetBrains Mono,monospace;"
                f"line-height:1.7;'>{ranges_str}</div></div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
def page_molecules(data):
    det = data.get("detections", {})
    st.markdown('<div class="page-title">◎ Молекулы</div>', unsafe_allow_html=True)

    col_s, col_f = st.columns(2)
    with col_s:
        fig_s = snr_bar_fig(det)
        if fig_s.data: st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})
    with col_f:
        fig_f = fap_bar_fig(det)
        if fig_f.data: st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")

    for mol, info in det.items():
        status   = info.get("status","NOT DETECTED")
        conf     = info.get("confidence", 0.)
        snr      = info.get("max_snr", 0.)
        fap      = info.get("fap", 1.)
        detected = info.get("detected", False)
        color    = STATUS_COLOR.get(status,"#374151")
        mc       = MOL_COLOR.get(mol,"#6b7280")
        bands    = info.get("bands", [])
        contams  = sorted(set(c for b in bands for c in b.get("contamination",[])))

        with st.expander(
            f"{mol}  ·  {status}  ·  {snr:.1f}σ  ·  conf {conf:.2f}",
            expanded=detected
        ):
            cl, cr = st.columns([5, 2])
            with cl:
                expl = info.get("explanation","")
                if expl:
                    st.markdown(
                        f"<div style='color:#8b9dc3;font-size:13px;margin-bottom:14px;"
                        f"padding:10px 14px;background:rgba({int(mc[1:3],16)},"
                        f"{int(mc[3:5],16)},{int(mc[5:7],16)},0.05);"
                        f"border-left:3px solid {mc}40;border-radius:0 8px 8px 0;'>"
                        f"💬 {expl}</div>", unsafe_allow_html=True)

                if bands:
                    rows = []
                    for b in bands:
                        rng = b.get("band_range",(0,0))
                        cov = b.get("covered",False)
                        rows.append({
                            "Полоса μm": f"{rng[0]:.2f}–{rng[1]:.2f}",
                            "Cov": "✓" if cov else "—",
                            "Depth":   round(b.get("depth",0),5) if cov else "—",
                            "±err":    round(b.get("depth_err",0),5) if cov else "—",
                            "EQW":     round(b.get("eqw",0),5) if cov else "—",
                            "SNR":     round(b.get("snr",0),2) if cov else "—",
                            "FAP":     f"{b.get('fap',1):.2e}" if cov else "—",
                            "R_eff":   int(b["R_eff"]) if b.get("R_eff") else "—",
                            "Dil.":    round(b.get("dilution_factor",1),3) if b.get("dilution_factor") else "—",
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            with cr:
                cov_bands = [b for b in bands if b.get("covered")]
                st.markdown(f"""
<div class="glass" style="border-color:{mc}20;">
  <div class="section-label">{mol} — ДЕТАЛИ</div>
  <div class="stat-row"><span class="sl">Статус</span>
    <span class="badge" style="background:{color}18;color:{color};border:1px solid {color}35;">{status}</span>
  </div>
  <div class="stat-row"><span class="sl">Confidence</span>
    <span class="sv">{conf:.3f}</span></div>
  <div class="stat-row"><span class="sl">Max SNR</span>
    <span style="color:{mc};font-family:'JetBrains Mono';font-size:11px;font-weight:600;">{snr:.2f}σ</span>
  </div>
  <div class="stat-row"><span class="sl">FAP</span>
    <span class="sv">{fap:.2e}</span></div>
  <div class="stat-row"><span class="sl">Полос покрыто</span>
    <span class="sv">{len(cov_bands)}/{len(bands)}</span></div>
</div>""", unsafe_allow_html=True)

                if contams:
                    st.markdown(
                        f"<div class='warn-box'>⚠ Загрязнение:<br>"
                        + "".join(f"· {c}<br>" for c in contams)
                        + "</div>", unsafe_allow_html=True)

                if cov_bands:
                    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
                    for b in cov_bands:
                        rng  = b.get("band_range",(0,0))
                        bsnr = b.get("snr",0)
                        bw   = min(bsnr/10*100, 100)
                        bc   = mc if bsnr >= SNR_VALID_BAND else "#1e293b"
                        st.markdown(
                            f"<div style='font-size:9px;color:#1e3347;font-family:JetBrains Mono,monospace;"
                            f"margin-bottom:2px;'>{rng[0]:.2f}–{rng[1]:.2f} μm</div>"
                            f"<div style='background:rgba(255,255,255,.04);border-radius:4px;height:6px;"
                            f"margin-bottom:6px;overflow:hidden;'>"
                            f"<div style='background:{bc};width:{bw:.0f}%;height:100%;border-radius:4px;"
                            f"{'box-shadow:0 0 8px '+mc+'60;' if bsnr>=SNR_VALID_BAND else ''}'></div>"
                            f"</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
def page_report(data, sel_t, sel_r):
    st.markdown('<div class="page-title">◻ Отчёт</div>', unsafe_allow_html=True)
    ml = data.get("ml_analysis") or {}

    ml_rep = ml.get("generated_report")
    if ml_rep:
        st.markdown('<div class="section-label">ML-ОТЧЁТ</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:rgba(8,15,32,.95);border:1px solid rgba(255,255,255,.05);"
            f"border-radius:12px;padding:22px 24px;font-family:JetBrains Mono,monospace;"
            f"font-size:12px;color:#8b9dc3;white-space:pre-wrap;line-height:1.75;"
            f"border-left:3px solid rgba(0,212,255,.3);'>{ml_rep}</div>",
            unsafe_allow_html=True)
        st.markdown("---")

    summary = load_summary(sel_t, sel_r)
    if summary:
        st.markdown('<div class="section-label">ПОЛНЫЙ ОТЧЁТ</div>', unsafe_allow_html=True)
        st.code(summary, language="text")

    st.markdown("---")
    st.markdown('<div class="section-label">СКАЧАТЬ</div>', unsafe_allow_html=True)
    dc = st.columns(3)
    with dc[0]:
        st.download_button("⬇ results.json",
            data=json.dumps(data, indent=2, ensure_ascii=False),
            file_name=f"{sel_t}_{sel_r}_results.json", mime="application/json")
    with dc[1]:
        if summary:
            st.download_button("⬇ summary.txt", data=summary,
                file_name=f"{sel_t}_{sel_r}_summary.txt", mime="text/plain")
    with dc[2]:
        png = OUTPUTS_DIR/sel_t/sel_r/"plots"/"spectrum.png"
        if png.exists():
            st.download_button("⬇ spectrum.png", data=png.read_bytes(),
                file_name=f"{sel_t}_{sel_r}_spectrum.png", mime="image/png")


# ─────────────────────────────────────────────────────────────────────────────
def page_run():
    st.markdown('<div class="page-title">▶ Запуск пайплайна</div>', unsafe_allow_html=True)
    col_src, col_opt = st.columns([3, 2], gap="large")

    with col_src:
        mode = st.radio("Источник", ["📂 data/inputs/","⬆ Загрузить FITS"], horizontal=True)
        uploaded, sel_path = [], None

        if mode == "⬆ Загрузить FITS":
            uploaded = st.file_uploader("FITS", type=["fits","fit","fz"], accept_multiple_files=True)
        else:
            if DATA_INPUTS.exists():
                tdirs = sorted(d.name for d in DATA_INPUTS.iterdir() if d.is_dir())
                if tdirs:
                    st_t = st.selectbox("Объект", tdirs)
                    obj_dir = DATA_INPUTS / st_t
                    sdirs = [d.name for d in obj_dir.iterdir() if d.is_dir()]
                    sel_path = str(obj_dir / st.selectbox("Инструмент", sdirs)) if sdirs else str(obj_dir)
                    st.caption(f"`{sel_path}`")
                    fits_l = list(Path(sel_path).glob("*.fits"))
                    if fits_l:
                        with st.expander(f"📄 Файлы ({len(fits_l)})"):
                            for f in fits_l[:20]: st.caption(f.name)
                else:
                    st.info("data/inputs/ пуст")
            else:
                st.warning("data/inputs/ не найдена")

    with col_opt:
        st.markdown('<div class="section-label">ПАРАМЕТРЫ</div>', unsafe_allow_html=True)
        tname = st.text_input("Имя объекта", placeholder="авто из заголовка")
        c1, c2 = st.columns(2)
        with c1: no_plot = st.checkbox("Без PNG"); force_s = st.checkbox("Один файл")
        with c2: no_json = st.checkbox("Без JSON"); verbose = st.checkbox("Verbose")

    st.markdown("---")
    if st.button("▶  Запустить анализ", type="primary", use_container_width=True):
        tmp_dir = None
        if mode == "⬆ Загрузить FITS" and uploaded:
            tmp_dir = tempfile.mkdtemp()
            for uf in uploaded:
                open(os.path.join(tmp_dir, uf.name), "wb").write(uf.read())
            run_input = tmp_dir if len(uploaded)>1 else os.path.join(tmp_dir, uploaded[0].name)
        elif sel_path:
            run_input = sel_path
        else:
            st.error("Нет источника данных"); return

        cmd = [sys.executable, str(ROOT/"run_pipeline.py"), run_input]
        if tname:   cmd += ["--target-name", tname]
        if no_plot: cmd.append("--no-plot")
        if no_json: cmd.append("--no-json")
        if verbose: cmd.append("--verbose")
        if force_s: cmd.append("--force-single")

        st.code(" ".join(cmd), language="bash")
        box = st.empty()
        with st.spinner("Анализируем..."):
            proc = subprocess.Popen(cmd, cwd=str(ROOT),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            lines = []
            for line in proc.stdout:
                lines.append(line.rstrip()); box.code("\n".join(lines[-60:]), language="bash")
            proc.wait()
        if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)
        if proc.returncode == 0:
            st.success("✅ Готово! Перейдите в **◈ Обзор**")
            st.cache_data.clear()
        else:
            st.error(f"❌ Ошибка (код {proc.returncode})")


# ─────────────────────────────────────────────────────────────────────────────
def page_history():
    st.markdown('<div class="page-title">≡ История запусков</div>', unsafe_allow_html=True)
    if st.button("⟳ Обновить"): st.cache_data.clear(); st.rerun()

    df = all_runs_df()
    if df.empty: st.info("Нет запусков"); return

    fc1, fc2, fc3 = st.columns(3)
    with fc1: tf = st.multiselect("Объект", df["Target"].unique())
    with fc2: mf = st.text_input("Содержит молекулу", placeholder="H2O…")
    with fc3: mn = st.slider("Мин. детекций", 0, 6, 0)

    d = df.copy()
    if tf: d = d[d["Target"].isin(tf)]
    if mf: d = d[d["Detected"].str.contains(mf.strip(), case=False, na=False)]
    d = d[d["N"] >= mn]

    st.dataframe(d.drop(columns=["N"]), use_container_width=True, hide_index=True,
                 column_config={
                     "Quality": st.column_config.ProgressColumn("Quality",min_value=0,max_value=1,format="%.2f"),
                     "T (K)":   st.column_config.NumberColumn("T (K)", format="%.0f"),
                     "Fit OK":  st.column_config.CheckboxColumn("Fit OK"),
                 })
    st.caption(f"{len(d)} из {len(df)} запусков")
    st.markdown("---")

    col_m, col_q = st.columns(2)
    with col_m:
        mc: dict = {}
        for row in d["Detected"]:
            for m in (row or "").split(", "):
                m = m.strip()
                if m and m != "—": mc[m] = mc.get(m,0)+1
        if mc:
            fig = go.Figure(go.Bar(x=list(mc), y=list(mc.values()),
                                   marker_color=[MOL_COLOR.get(m,"#374151") for m in mc],
                                   marker=dict(line=dict(width=0)),
                                   hovertemplate="%{x}: %{y}<extra></extra>"))
            fig.update_layout(**_DARK, height=260,
                              title=dict(text="<span style='color:#8b9dc3'>Частота детекции</span>",font=dict(size=12)),
                              xaxis=_GRID, yaxis={**_GRID,"title":"Запусков"},
                              margin=dict(l=40,r=16,t=44,b=38))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with col_q:
        qv = d["Quality"].dropna()
        if not qv.empty:
            fig = go.Figure(go.Histogram(x=qv, nbinsx=10,
                                         marker_color="#8b5cf6", opacity=0.8))
            fig.add_vline(x=0.4, line_dash="dot", line_color="#fbbf24",
                          annotation_text="Порог", annotation_font_size=8,
                          annotation_font_color="#fbbf24")
            fig.update_layout(**_DARK, height=260,
                              title=dict(text="<span style='color:#8b9dc3'>Распределение качества</span>",font=dict(size=12)),
                              xaxis={**_GRID,"title":"Quality"},
                              yaxis={**_GRID,"title":"Запусков"},
                              margin=dict(l=40,r=16,t=44,b=38))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    tv = d["T (K)"].dropna(); tv = tv[tv > 0]
    if not tv.empty:
        fig = go.Figure(go.Histogram(x=tv, nbinsx=15,
                                      marker_color="#00d4ff", opacity=0.7))
        for xv, lbl in [(500,"Y/T"),(1400,"T/L"),(2300,"L/M"),(3900,"M/FGK"),(7500,"FGK/Hot")]:
            fig.add_vline(x=xv, line_dash="dot", line_color="#0d1b2e",
                          annotation_text=lbl, annotation_font_size=8,
                          annotation_font_color="#2d4060")
        fig.update_layout(**_DARK, height=240,
                          title=dict(text="<span style='color:#8b9dc3'>Температура объектов</span>",font=dict(size=12)),
                          xaxis={**_GRID,"title":"T (K)"},
                          yaxis={**_GRID,"title":"Запусков"},
                          margin=dict(l=40,r=16,t=44,b=38))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})


# ─────────────────────────────────────────────────────────────────────────────
def page_compare():
    st.markdown('<div class="page-title">⊞ Сравнение</div>', unsafe_allow_html=True)
    opts = [f"{t} / {r}" for t in list_targets() for r in list_runs(t)]
    if not opts: st.info("Нет данных"); return

    chosen = st.multiselect("Запуски (2–4)", opts, max_selections=4)
    if len(chosen) < 2: st.info("Выберите ≥ 2 запуска"); return

    rdata = {}
    for opt in chosen:
        t, r = opt.split(" / ", 1)
        d = load_results(t, r)
        if d: rdata[opt] = d
    if not rdata: st.error("Нет данных"); return

    mols   = list(next(iter(rdata.values()))["detections"].keys())
    shorts = [f"{o.split('/')[0].strip()} · {o.split('/')[1].strip()[:14]}" for o in rdata]
    pal    = ["#00d4ff","#34d399","#fbbf24","#a78bfa"]

    ca, cb = st.columns(2)
    with ca:
        fig = go.Figure()
        for i, (lbl, d) in enumerate(zip(shorts, rdata.values())):
            fig.add_trace(go.Bar(name=lbl, x=mols,
                                  y=[d["detections"].get(m,{}).get("confidence",0.) for m in mols],
                                  marker_color=pal[i%4], marker=dict(line=dict(width=0))))
        fig.update_layout(**_DARK, barmode="group", height=300,
                          title=dict(text="<span style='color:#8b9dc3'>Confidence</span>",font=dict(size=12)),
                          yaxis={**_GRID,"range":[0,1],"title":"Confidence"}, xaxis=_GRID,
                          legend=dict(orientation="h",y=1.12,font=dict(size=10,color="#8b9dc3")),
                          margin=dict(l=40,r=16,t=60,b=38))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with cb:
        fig = go.Figure()
        for i, (lbl, d) in enumerate(zip(shorts, rdata.values())):
            fig.add_trace(go.Bar(name=lbl, x=mols,
                                  y=[d["detections"].get(m,{}).get("max_snr",0.) for m in mols],
                                  marker_color=pal[i%4], marker=dict(line=dict(width=0))))
        fig.update_layout(**_DARK, barmode="group", height=300,
                          title=dict(text="<span style='color:#8b9dc3'>Max SNR</span>",font=dict(size=12)),
                          yaxis={**_GRID,"title":"SNR"}, xaxis=_GRID,
                          legend=dict(orientation="h",y=1.12,font=dict(size=10,color="#8b9dc3")),
                          margin=dict(l=40,r=16,t=60,b=38))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    col_r, col_t = st.columns([1, 1])
    with col_r:
        if len(mols) >= 3:
            fig_r = radar_fig(rdata, mols)
            for i, t in enumerate(fig_r.data): t.name = shorts[i]
            st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar":False})
    with col_t:
        st.markdown('<div class="section-label">СТАТУС ДЕТЕКЦИИ</div>', unsafe_allow_html=True)
        tbl = {"Молекула": mols}
        for lbl, d in zip(shorts, rdata.values()):
            tbl[lbl[:20]] = [d["detections"].get(m,{}).get("status","—") for m in mols]
        st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">НАЛОЖЕНИЕ СПЕКТРОВ (нормализовано)</div>', unsafe_allow_html=True)
    fig_ov = go.Figure()
    has = False
    for i, (opt, d) in enumerate(rdata.items()):
        t, r = opt.split(" / ", 1)
        spec = load_spectrum(str(OUTPUTS_DIR/t/r/"input"/"spectrum_used.fits"))
        if spec:
            w = np.asarray(spec["wavelength_um"]); flx = np.asarray(spec["flux"])
            flx_n = flx / np.nanmedian(flx[flx>0]) if np.any(flx>0) else flx
            fig_ov.add_trace(go.Scatter(x=w, y=flx_n, mode="lines",
                                         name=shorts[i], line=dict(color=pal[i%4], width=0.9),
                                         hovertemplate="λ=%{x:.4f}μm  %{y:.3f}<extra></extra>"))
            has = True
    if has:
        fig_ov.update_layout(**_DARK, height=360, hovermode="x unified",
                              title=dict(text="<span style='color:#8b9dc3'>Нормализованные спектры</span>",font=dict(size=12)),
                              xaxis={**_GRID,"title":"λ (μm)"},
                              yaxis={**_GRID,"title":"F / median"},
                              legend=dict(orientation="h",y=1.08,font=dict(size=10,color="#8b9dc3")),
                              margin=dict(l=55,r=16,t=55,b=42))
        st.plotly_chart(fig_ov, use_container_width=True)
    else:
        st.info("FITS файлы недоступны для наложения")


# ─────────────────────────────────────────────────────────────────────────────
def page_settings():
    st.markdown('<div class="page-title">◧ Настройки</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Изменения записываются в config.py и применяются к следующим запускам</div>',
                unsafe_allow_html=True)

    if not CONFIG_PATH.exists():
        st.error(f"config.py не найден: {CONFIG_PATH}"); return

    cfg = read_config()
    groups = {
        "🎯 Пороги детекции": {
            "SNR_VALID_BAND": ("Мин. SNR полосы",     0.5, 10., 0.1),
            "SNR_ANNOTATE":   ("SNR для аннотации",   0.5, 10., 0.1),
        },
        "📊 Confidence статусы": {
            "CONF_STRONG":   ("STRONG порог",   0.5, 1.0, 0.05),
            "CONF_LIKELY":   ("LIKELY порог",   0.3, 0.9, 0.05),
            "CONF_MARGINAL": ("MARGINAL порог", 0.1, 0.7, 0.05),
            "CONF_WEAK":     ("WEAK порог",     0.05,0.5, 0.05),
        },
        "🔬 Глубина полосы (бонусы)": {
            "DEPTH_BONUS_WEAK":   ("Слабый ≥",  0.01, 0.2, 0.01),
            "DEPTH_BONUS_MED":    ("Средний ≥", 0.01, 0.3, 0.01),
            "DEPTH_BONUS_STRONG": ("Сильный ≥", 0.05, 0.5, 0.01),
        },
        "📋 Качество данных": {
            "QUALITY_USABLE":       ("Порог пригодности",    0.1, 0.8, 0.05),
            "QUALITY_CAP_POOR":     ("Ceiling — плохое",     0.05,0.5, 0.05),
            "QUALITY_CAP_MODERATE": ("Ceiling — среднее",    0.2, 0.8, 0.05),
        },
    }

    changed = {}
    for grp_name, params in groups.items():
        st.markdown(f"**{grp_name}**")
        cols = st.columns(len(params))
        for col, (key, (lbl, vmin, vmax, step)) in zip(cols, params.items()):
            cur = cfg.get(key, 0.)
            nv  = col.number_input(lbl, min_value=vmin, max_value=vmax, value=cur,
                                   step=step, key=f"cfg_{key}")
            if abs(nv - cur) > 1e-9: changed[key] = nv
        st.markdown("")

    if changed:
        st.warning(f"Несохранённые изменения: {', '.join(changed)}")

    col_s, _ = st.columns([1, 4])
    with col_s:
        if st.button("💾 Сохранить", type="primary"):
            for k, v in changed.items(): write_config(k, v)
            st.success(f"✓ {', '.join(changed)}")
            st.cache_data.clear()

    st.markdown("---")
    with st.expander("📄 config.py"):
        st.code(CONFIG_PATH.read_text(), language="python")

    with st.expander("🧬 Молекулярные полосы"):
        for mol, bds in MOLECULE_BANDS.items():
            c = MOL_COLOR.get(mol,"#6b7280")
            st.markdown(f"<span style='color:{c};font-weight:700;font-family:JetBrains Mono,monospace;'>{mol}</span>",
                        unsafe_allow_html=True)
            rows = [{"Полоса": f"{b['band'][0]:.2f}–{b['band'][1]:.2f}",
                     "Левый sb": f"{b['side'][0]:.2f}–{b['side'][1]:.2f}",
                     "Правый sb": f"{b['side'][2]:.2f}–{b['side'][3]:.2f}"} for b in bds]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
render_sidebar()

cur_t    = st.session_state.get("sb_target")
cur_r    = st.session_state.get("sb_run")
cur_data = load_results(cur_t, cur_r) if cur_t and cur_r else None
page     = st.session_state.page

if page == "run":       page_run()
elif page == "history": page_history()
elif page == "compare": page_compare()
elif page == "settings":page_settings()
else:
    if not cur_data:
        if not list_targets():
            st.markdown('<div class="page-title">🔭 Добро пожаловать</div>', unsafe_allow_html=True)
            st.markdown('<div class="subtitle">Запустите пайплайн через страницу ▶ Запуск</div>',
                        unsafe_allow_html=True)
            st.info("Нет результатов. Перейдите в **▶ Запуск** и выберите FITS-файлы.")
        else:
            st.info("Выберите объект и запуск в боковой панели.")
    elif page == "spectrum":  page_spectrum(cur_data, cur_t, cur_r)
    elif page == "molecules": page_molecules(cur_data)
    elif page == "report":    page_report(cur_data, cur_t, cur_r)
    else:                     page_overview(cur_data, cur_t, cur_r)
