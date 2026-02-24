"""Shared design tokens and UI helpers for the WWTP demo dashboard."""
from __future__ import annotations

import streamlit as st

# Design Tokens (analysis-friendly, high-contrast palette)
PRIMARY = "#1D4ED8"      # blue
SECONDARY = "#D97706"    # orange
MUTED = "#64748B"        # slate
BORDER = "#D1D5DB"       # light gray
BG_PAGE = "#F8FAFC"      # near-white

# Aliases kept for compatibility across pages
PRIMARY_DEEP = PRIMARY
PRIMARY_DARK = "#0F172A"
PRIMARY_LIGHT = "#0EA5E9"
PRIMARY_PALE = "#93C5FD"
PRIMARY_GHOST = BG_PAGE
BG_DARK = "#E2E8F0"
BG_MID = "#EEF2FF"
TEXT_PRIMARY = "#0F172A"
TEXT_MUTED = "#475569"
CARD_BG = "#FFFFFF"

TEAL = PRIMARY
SUCCESS = "#16A34A"
WARNING = SECONDARY
DANGER = "#DC2626"
PURPLE = "#7C3AED"

ACTUAL_COLOR = PRIMARY
PRED_COLOR = SECONDARY
HIGH_ERR_FILL = "rgba(220,38,38,0.10)"

GRADE_COLORS = {"우수": SUCCESS, "양호": WARNING, "보통": MUTED}

CHART_PALETTE = [PRIMARY, SECONDARY, SUCCESS, PURPLE, DANGER, "#0891B2", "#4D7C0F"]

PLOTLY_BASE = dict(
    plot_bgcolor=BG_PAGE,
    paper_bgcolor=BG_PAGE,
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=TEXT_PRIMARY),
    hoverlabel=dict(bgcolor=BG_PAGE, bordercolor=BORDER, font_size=13),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
)

_GLOBAL_CSS = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-color: {BG_PAGE};
}}
[data-testid="stHeader"] {{ background: transparent; }}

[data-testid="stSidebar"] {{
    background-color: {BG_PAGE};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {PRIMARY} !important; }}
[data-testid="stSidebarNav"] a[aria-selected="true"] * {{
    color: {PRIMARY} !important;
    font-weight: 700 !important;
}}

[data-testid="metric-container"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 18px 14px !important;
    box-shadow: none;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: {PRIMARY} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {PRIMARY} !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricDelta"] {{ font-weight: 600 !important; }}

[data-testid="baseButton-primary"] {{
    background: {PRIMARY} !important;
    border: 1px solid {PRIMARY} !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em;
    box-shadow: none !important;
}}
[data-testid="baseButton-secondary"] {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    border-color: {BORDER} !important;
    color: {PRIMARY} !important;
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
    border-radius: 10px !important;
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
    """Return HTML for a styled KPI card with a colored top border."""
    delta_html = ""
    if delta:
        color = SUCCESS if not delta.startswith("-") else DANGER
        arrow = "▲" if not delta.startswith("-") else "▼"
        delta_html = (
            f'<div style="margin-top:8px;font-size:0.78rem;font-weight:600;color:{color}">'
            f'{arrow} {delta}</div>'
        )
    return (
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-top:3px solid {accent};'
        f'border-radius:10px;padding:18px 20px;box-shadow:none;">'
        f'<div style="font-size:0.68rem;font-weight:700;color:{TEXT_MUTED};text-transform:uppercase;'
        f'letter-spacing:0.07em;margin-bottom:8px">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{TEXT_PRIMARY};line-height:1.1">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


def section_header(title: str, subtitle: str = "") -> str:
    sub = (
        f'<p style="margin:5px 0 0;font-size:0.85rem;color:{TEXT_MUTED};font-weight:400">{subtitle}</p>'
        if subtitle
        else ""
    )
    return (
        f'<div style="margin:2rem 0 1.1rem">'
        f'<div style="font-size:1.05rem;font-weight:700;color:{TEXT_PRIMARY};'
        f'padding-bottom:6px;border-bottom:2.5px solid {PRIMARY};display:inline-block">{title}</div>'
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
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};'
        f'border-radius:14px;padding:36px 40px;margin-bottom:1.5rem;box-shadow:none;">'
        f'<h1 style="color:{PRIMARY} !important;font-size:2rem;font-weight:800;margin:0 0 8px">{title}</h1>'
        f'<p style="color:{SECONDARY};font-size:1rem;margin:0">{subtitle}</p>'
        f'</div>'
    )


def info_banner(text: str, icon: str = "ℹ") -> str:
    return (
        f'<div style="background:{BG_PAGE};border:1px solid {BORDER};border-left:4px solid {PRIMARY};'
        f'border-radius:8px;padding:12px 16px;font-size:0.88rem;color:{PRIMARY};margin:0.5rem 0">'
        f'{icon} {text}'
        f'</div>'
    )
