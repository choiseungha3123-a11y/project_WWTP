"""Shared design tokens and UI helpers for the WWTP demo dashboard."""
from __future__ import annotations

import streamlit as st

# ── Design Tokens ─────────────────────────────────────────────────────────────
# Calm analytical palette — muted chrome, semantic data colors

# Surfaces
BG_PAGE  = "#F7F9FC"   # cool off-white (page background)
CARD_BG  = "#FFFFFF"   # pure white cards

# Borders
BORDER   = "#E4E7EB"   # subtle neutral border

# Typography
TEXT_PRIMARY = "#1C2536"   # deep charcoal (readable, not harsh black)
TEXT_MUTED   = "#6B7280"   # medium gray
TEXT_LIGHT   = "#9CA3AF"   # light gray

# Primary accent — calm steel blue (less electric than before)
PRIMARY       = "#2563EB"   # blue-600 (calmer, desaturated)
PRIMARY_DARK  = "#1E3A5F"   # deep navy for strong emphasis
PRIMARY_LIGHT = "#5A8DEE"   # lighter blue for secondary accents
PRIMARY_PALE  = "#BFCFE8"   # pale blue (very subtle)
PRIMARY_DEEP  = PRIMARY
PRIMARY_GHOST = BG_PAGE

# Semantic data colors (used by grade / performance indicators)
SUCCESS = "#16A34A"   # green-600  — 우수/good
WARNING = "#D97706"   # amber-600  — 양호/ok
DANGER  = "#DC2626"   # red-600    — 보통/poor
PURPLE  = "#7C3AED"   # violet for misc series

# Secondary decorative — calm teal (replaces loud orange)
SECONDARY = "#0891B2"   # cyan-600

# Background variants
BG_DARK = "#EDF0F4"
BG_MID  = "#EBF0FA"

# Compatibility aliases
MUTED = TEXT_MUTED
TEAL  = SECONDARY

# Chart / analysis colors
ACTUAL_COLOR  = PRIMARY
PRED_COLOR    = "#E07B39"   # warm orange — keeps actual vs predicted distinct
HIGH_ERR_FILL = "rgba(220,38,38,0.07)"

GRADE_COLORS = {"우수": SUCCESS, "양호": WARNING, "보통": DANGER}

CHART_PALETTE = [
    PRIMARY, SUCCESS, WARNING, PURPLE, DANGER, SECONDARY, "#4D7C0F"
]

PLOTLY_BASE = dict(
    plot_bgcolor=CARD_BG,
    paper_bgcolor=CARD_BG,
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=TEXT_PRIMARY),
    hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER, font_size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#EAECF0", linecolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor="#EAECF0", linecolor=BORDER, zerolinecolor=BORDER),
)

_GLOBAL_CSS = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-color: {BG_PAGE};
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* ── Calm sidebar ───────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {CARD_BG};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}
[data-testid="stSidebarNav"] a[aria-selected="true"] {{
    background: {BG_MID} !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebarNav"] a[aria-selected="true"] * {{
    color: {PRIMARY} !important;
    font-weight: 700 !important;
}}

/* ── Metric cards ────────────────────────────────────────── */
[data-testid="metric-container"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 18px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
[data-testid="stMetricValue"] {{
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    color: {TEXT_PRIMARY} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricDelta"] {{ font-weight: 600 !important; }}

/* ── Buttons ─────────────────────────────────────────────── */
[data-testid="baseButton-primary"] {{
    background: {PRIMARY} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
}}
[data-testid="baseButton-secondary"] {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    border-color: {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
}}

hr {{ border-color: {BORDER} !important; margin: 1.5rem 0 !important; }}

[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    background: {CARD_BG} !important;
}}

[data-testid="stSelectbox"] > div > div {{
    border-radius: 8px !important;
    border-color: {BORDER} !important;
}}

[data-testid="stAlert"] {{
    border-radius: 8px !important;
    font-size: 0.88rem !important;
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
</style>
"""


def apply_global_style() -> None:
    """Inject shared CSS. Call once at the top of each page."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def kpi_card(label: str, value: str, accent: str = PRIMARY, delta: str = "") -> str:
    """KPI card: dark charcoal value for readability, accent color on top border."""
    delta_html = ""
    if delta:
        color = SUCCESS if not delta.startswith("-") else DANGER
        arrow = "▲" if not delta.startswith("-") else "▼"
        delta_html = (
            f'<div style="margin-top:6px;font-size:0.78rem;font-weight:600;color:{color}">'
            f'{arrow} {delta}</div>'
        )
    val_len = len(value)
    val_font = "1.7rem" if val_len <= 8 else ("1.25rem" if val_len <= 12 else "0.95rem")
    return (
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-top:3px solid {accent};'
        f'border-radius:10px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
        f'<div style="font-size:0.67rem;font-weight:700;color:{TEXT_MUTED};text-transform:uppercase;'
        f'letter-spacing:0.07em;margin-bottom:8px">{label}</div>'
        f'<div style="font-size:{val_font};font-weight:800;color:{TEXT_PRIMARY};line-height:1.1;'
        f'word-break:break-all;overflow-wrap:anywhere">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


def section_header(title: str, subtitle: str = "") -> str:
    sub = (
        f'<p style="margin:5px 0 0;font-size:0.83rem;color:{TEXT_MUTED};font-weight:400">{subtitle}</p>'
        if subtitle
        else ""
    )
    return (
        f'<div style="margin:2rem 0 1rem">'
        f'<div style="display:inline-block;font-size:1rem;font-weight:700;color:{TEXT_PRIMARY};'
        f'padding-bottom:6px;border-bottom:2px solid {PRIMARY}">{title}</div>'
        f'{sub}'
        f'</div>'
    )


def badge(text: str, color: str = PRIMARY) -> str:
    return (
        f'<span style="display:inline-block;background:{BG_MID};color:{color};'
        f'border:1px solid {BORDER};font-size:0.7rem;font-weight:700;padding:3px 10px;border-radius:20px;'
        f'letter-spacing:0.04em">{text}</span>'
    )


def hero_banner(title: str, subtitle: str) -> str:
    return (
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-left:4px solid {PRIMARY};'
        f'border-radius:12px;padding:32px 40px;margin-bottom:1.5rem;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
        f'<h1 style="color:{TEXT_PRIMARY} !important;font-size:1.85rem;font-weight:800;margin:0 0 8px">'
        f'{title}</h1>'
        f'<p style="color:{TEXT_MUTED};font-size:0.95rem;margin:0;line-height:1.5">{subtitle}</p>'
        f'</div>'
    )


def info_banner(text: str, icon: str = "ℹ") -> str:
    return (
        f'<div style="background:{BG_MID};border:1px solid {BORDER};border-left:3px solid {PRIMARY};'
        f'border-radius:8px;padding:11px 16px;font-size:0.88rem;color:{TEXT_PRIMARY};margin:0.5rem 0">'
        f'{icon} {text}'
        f'</div>'
    )
