from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import TARGET_LABELS, TARGET_ORDER
from demo.utils.data_loader import load_feature_names
from demo.utils.styles import (
    BG_PAGE,
    BORDER,
    CARD_BG,
    PLOTLY_BASE,
    PRIMARY,
    PRIMARY_DARK,
    PRIMARY_LIGHT,
    PRIMARY_PALE,
    PRIMARY_GHOST,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_global_style,
    badge,
    section_header,
)
from src.config import FLOW_CONFIG, TMS_TARGETS

st.set_page_config(page_title="모델 정보", page_icon="🧠", layout="wide")
apply_global_style()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="margin-bottom:4px">🧠 모델 아키텍처 및 피처 정보</h2>'
    f'<p style="color:{TEXT_MUTED};margin:0 0 1.2rem">LSTM 구조 설계 · 추천 피처 목록 · 엔지니어링 파이프라인</p>',
    unsafe_allow_html=True,
)

# ── 모델 구조 요약 ─────────────────────────────────────────────────────────────
st.markdown(section_header("모델 구조 요약"), unsafe_allow_html=True)

rows = []
for target in TARGET_ORDER:
    cfg      = FLOW_CONFIG if target == "flow" else TMS_TARGETS[target]
    features = load_feature_names(target)
    head     = "4-layer" if cfg.get("deep_head", False) else "3-layer"
    rows.append({
        "타겟":     TARGET_LABELS[target],
        "hidden":   cfg["hidden_size"],
        "layers":   cfg["num_layers"],
        "attention": "✓" if cfg.get("use_attention", False) else "✗",
        "head":     head,
        "피처 수":  len(features),
    })

summary_df = pd.DataFrame(rows)
st.dataframe(
    summary_df.style.apply(
        lambda col: [
            f"color:{PRIMARY};font-weight:700" if v == "✓" else
            f"color:{PRIMARY_DARK}" if v == "✗" else ""
            for v in col
        ] if col.name == "attention" else [""] * len(col),
        axis=0,
    ),
    use_container_width=True,
    hide_index=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── 타겟별 추천 피처 + 파이프라인 (2열) ──────────────────────────────────────
col_feat, col_gap, col_pipe = st.columns([1, 0.08, 2.4])

with col_feat:
    st.markdown(section_header("추천 피처 목록"), unsafe_allow_html=True)
    selected_target = st.selectbox(
        "타겟 선택",
        options=TARGET_ORDER,
        format_func=lambda t: TARGET_LABELS[t],
    )
    feature_names = load_feature_names(selected_target)
    st.markdown(
        f'<div style="font-size:0.82rem;color:{TEXT_MUTED};margin-bottom:8px">'
        f'선택된 피처 수: <b style="color:{PRIMARY}">{len(feature_names)}개</b></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame({"피처명": feature_names}),
        use_container_width=True,
        hide_index=True,
        height=340,
    )

with col_gap:
    pass

with col_pipe:
    st.markdown(
        section_header(
            "피처 엔지니어링 파이프라인",
            "notebook/feature/feature_engineering.py · src/preprocess.py 공통 적용",
        ),
        unsafe_allow_html=True,
    )

    pipeline_steps = [
        ("01", "add_target_lag_features", PRIMARY,
         "타겟 컬럼의 lag / rolling / diff / EWMA 피처 생성 (min_lag=2)"),
        ("02", "타겟 원본 제거", PRIMARY_DARK,
         "미래 정보 누수 방지를 위해 raw 타겟값 삭제"),
        ("03", "add_rain_features", PRIMARY_DARK,
         "강수량 관련 파생 피처 (누적, 최대 등)"),
        ("04", "add_station_agg_rain_features", PRIMARY_DARK,
         "3개 AWS 관측소 강수량 집계 피처"),
        ("05", "add_weather_features", PRIMARY,
         "기온 · 습도 · 이슬점 등 기상 파생 피처"),
        ("06", "add_process_features", PRIMARY,
         "공정 변수 (탱크 수위 · 유량 · 약품주입량 등) 파생 피처"),
        ("07", "add_temporal_features", PRIMARY_LIGHT,
         "process_cols + weather 대상 통계 피처 (rolling std, IQR 등)"),
        ("08", "add_time_features", PRIMARY_LIGHT,
         "시간 주기성 피처 (hour, weekday, doy sin/cos, iso_week 등)"),
        ("09", "ffill → fillna(0)", PRIMARY_PALE,
         "결측치 전방 보간 후 나머지 0으로 대체"),
    ]

    for num, name, accent, desc in pipeline_steps:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:12px;'
            f'background:{CARD_BG};border:1px solid {BORDER};border-left:3px solid {accent};'
            f'border-radius:8px;padding:10px 14px;margin-bottom:6px">'
            f'<span style="background:{accent}18;color:{accent};font-size:0.7rem;font-weight:800;'
            f'padding:3px 7px;border-radius:6px;white-space:nowrap;flex-shrink:0">{num}</span>'
            f'<div>'
            f'<code style="font-size:0.82rem;color:{accent};font-weight:700">{name}</code>'
            f'<div style="font-size:0.8rem;color:{TEXT_MUTED};margin-top:3px">{desc}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── LSTM 아키텍처 다이어그램 ─────────────────────────────────────────────────
st.markdown(section_header("LSTM 아키텍처 다이어그램", "타겟 선택에 따라 구조가 동적으로 변경됩니다"), unsafe_allow_html=True)

sel_cfg   = FLOW_CONFIG if selected_target == "flow" else TMS_TARGETS[selected_target]
hidden    = sel_cfg["hidden_size"]
n_layers  = sel_cfg["num_layers"]
dropout   = sel_cfg.get("dropout", 0.2)
use_attn  = sel_cfg.get("use_attention", False)
deep_head = sel_cfg.get("deep_head", False)
n_feat    = len(feature_names)

if deep_head:
    head_dims = f"{hidden} → {hidden//2} → {hidden//4} → {hidden//8} → 1"
    head_sub  = "4-layer FC Head"
else:
    head_dims = f"{hidden} → {hidden//2} → {hidden//4} → 1"
    head_sub  = "3-layer FC Head"

# node: (title, subtitle, y_center, bg, border_hex, active)
nodes = [
    ("Input",          f"batch × 48 × {n_feat}",                      0.88, BG_PAGE,  PRIMARY,      True),
    ("LSTM",           f"hidden={hidden}  layers={n_layers}  dropout={dropout}", 0.67, BG_PAGE,  PRIMARY_DARK, True),
    ("Self-Attention", "사용" if use_attn else "미사용",               0.46,
     BG_PAGE if use_attn else PRIMARY_GHOST,
     PRIMARY if use_attn else PRIMARY_PALE,
     use_attn),
    (head_sub,         head_dims,                                      0.25, BG_PAGE,  PRIMARY_LIGHT, True),
    ("Output",         "scalar (1)",                                   0.04, BG_PAGE,  PRIMARY_DARK, True),
]

CX = 0.50
BW = 0.34
BH = 0.082

fig = go.Figure()

for title1, title2, cy, bg, border, active in nodes:
    alpha      = 1.0 if active else 0.30
    text_color = TEXT_PRIMARY if active else PRIMARY_PALE
    sub_color  = TEXT_MUTED   if active else PRIMARY_PALE

    fig.add_shape(
        type="rect",
        x0=CX - BW, x1=CX + BW,
        y0=cy - BH, y1=cy + BH,
        fillcolor=bg,
        line=dict(color=border, width=2),
        opacity=alpha,
        layer="below",
    )
    fig.add_annotation(
        x=CX, y=cy + 0.022,
        text=f"<b>{title1}</b>",
        showarrow=False,
        font=dict(size=14, color=text_color),
        align="center", xanchor="center", yanchor="middle",
    )
    fig.add_annotation(
        x=CX, y=cy - 0.028,
        text=title2,
        showarrow=False,
        font=dict(size=11, color=sub_color),
        align="center", xanchor="center", yanchor="middle",
    )

# 연결 화살표
arrow_ys = [
    (nodes[0][2] - BH, nodes[1][2] + BH),
    (nodes[1][2] - BH, nodes[2][2] + BH),
    (nodes[2][2] - BH, nodes[3][2] + BH),
    (nodes[3][2] - BH, nodes[4][2] + BH),
]
for y_tail, y_head in arrow_ys:
    fig.add_annotation(
        x=CX, y=y_head + 0.004,
        ax=CX, ay=y_tail - 0.004,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.2,
        arrowcolor=PRIMARY_PALE,
    )

# Autoregressive 루프
fig.add_annotation(
    x=CX + BW + 0.02, y=nodes[4][2],
    ax=CX + BW + 0.02, ay=nodes[0][2],
    xref="x", yref="y", axref="x", ayref="y",
    showarrow=True, arrowhead=3, arrowsize=1.0, arrowwidth=1.8,
    arrowcolor=PRIMARY_DARK,
)
fig.add_annotation(
    x=CX + BW + 0.10, y=(nodes[0][2] + nodes[4][2]) / 2,
    text="<b>Autoregressive</b><br>lag 피처 업데이트",
    showarrow=False,
    font=dict(size=10, color=PRIMARY_DARK),
    align="left", xanchor="left",
)

fig.update_layout(
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(visible=False, range=[-0.06, 1.0]),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=10, r=120, t=10, b=10),
    height=520,
)
st.plotly_chart(fig, use_container_width=True)
