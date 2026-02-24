from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import TARGET_LABELS, TARGET_ORDER
from demo.utils.data_loader import load_predictions
from demo.utils.metrics import compute_metrics
from demo.utils.styles import (
    ACTUAL_COLOR,
    BORDER,
    HIGH_ERR_FILL,
    PLOTLY_BASE,
    PRED_COLOR,
    PRIMARY,
    PRIMARY_DARK,
    PRIMARY_LIGHT,
    SUCCESS,
    TEXT_MUTED,
    apply_global_style,
    kpi_card,
    section_header,
)

st.set_page_config(page_title="예측 분석", page_icon="🔎", layout="wide")
apply_global_style()

st.markdown(
    f"<h2 style='margin-bottom:4px'>🔎 인터랙티브 예측 분석</h2>"
    f"<p style='color:{TEXT_MUTED};margin:0 0 1.2rem'>타겟별 실측값과 예측값을 비교합니다.</p>",
    unsafe_allow_html=True,
)

selected_target = st.selectbox("타겟 선택", options=TARGET_ORDER, format_func=lambda t: TARGET_LABELS[t])

df = load_predictions(selected_target).copy()
df["sample_idx"] = np.arange(len(df))
df["error"] = df["actual"] - df["predicted"]
df["abs_error"] = df["error"].abs()
metrics = compute_metrics(df["actual"], df["predicted"])

st.markdown(section_header("성능 지표"), unsafe_allow_html=True)
r2_accent = SUCCESS if metrics["r2"] >= 0.80 else (PRIMARY_LIGHT if metrics["r2"] >= 0.65 else PRIMARY_DARK)
mc = st.columns(6)
kpis = [
    ("R²", f"{metrics['r2']:.4f}", r2_accent),
    ("RMSE", f"{metrics['rmse']:.4f}", PRIMARY),
    ("MAE", f"{metrics['mae']:.4f}", PRIMARY),
    ("MAPE (%)", f"{metrics['mape']:.2f}", PRIMARY_LIGHT),
    ("5% 이내 비율", f"{metrics['within_5pct']:.1f}%", SUCCESS),
    ("샘플 수", f"{len(df):,}", PRIMARY_DARK),
]
for col, (label, value, accent) in zip(mc, kpis):
    col.markdown(kpi_card(label, value, accent), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(section_header("실측 vs 예측 시계열", "붉은 음영: 절대오차 상위 5% 구간"), unsafe_allow_html=True)

threshold = df["abs_error"].quantile(0.95)
high_err = df[df["abs_error"] >= threshold]

fig_ts = go.Figure()
for idx in high_err["sample_idx"]:
    fig_ts.add_vrect(x0=idx - 0.5, x1=idx + 0.5, fillcolor=HIGH_ERR_FILL, line_width=0, layer="below")

fig_ts.add_trace(
    go.Scatter(x=df["sample_idx"], y=df["actual"], mode="lines", name="실측", line=dict(color=ACTUAL_COLOR, width=2))
)
fig_ts.add_trace(
    go.Scatter(
        x=df["sample_idx"],
        y=df["predicted"],
        mode="lines",
        name="예측",
        line=dict(color=PRED_COLOR, width=2, dash="dash"),
    )
)

ts_layout = dict(**PLOTLY_BASE)
ts_layout.update(
    height=400,
    xaxis_title="샘플 인덱스",
    yaxis_title="값",
    xaxis=dict(
        rangeslider=dict(visible=True, bgcolor=BORDER, thickness=0.06),
        gridcolor=BORDER,
        linecolor=BORDER,
        zerolinecolor=BORDER,
    ),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_ts.update_layout(**ts_layout)
st.plotly_chart(fig_ts, use_container_width=True)
st.caption(f"절대오차 임계값(상위 5%) = {threshold:.4f}")

c1, c2 = st.columns(2)
with c1:
    st.markdown(section_header("산점도 (실측 vs 예측)", "오차 크기에 따라 색상 구분"), unsafe_allow_html=True)
    fig_scatter = px.scatter(
        df,
        x="actual",
        y="predicted",
        color="error",
        color_continuous_scale=[[0, "#1D4ED8"], [0.5, "#D1D5DB"], [1, "#D97706"]],
        opacity=0.65,
        labels={"actual": "실측값", "predicted": "예측값", "error": "오차"},
    )
    min_v = float(min(df["actual"].min(), df["predicted"].min()))
    max_v = float(max(df["actual"].max(), df["predicted"].max()))
    fig_scatter.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            name="y = x",
            line=dict(color=PRIMARY_DARK, dash="dot", width=1.5),
        )
    )
    sc_layout = dict(**PLOTLY_BASE)
    sc_layout.update(height=400, coloraxis_colorbar=dict(thickness=12, len=0.7))
    fig_scatter.update_layout(**sc_layout)
    st.plotly_chart(fig_scatter, use_container_width=True)

with c2:
    st.markdown(
        section_header("오차 분포", f"평균 {df['error'].mean():.4f} | 표준편차 {df['error'].std():.4f}"),
        unsafe_allow_html=True,
    )
    fig_hist = px.histogram(
        df,
        x="error",
        nbins=50,
        labels={"error": "오차 (actual - predicted)", "count": "빈도"},
        color_discrete_sequence=[ACTUAL_COLOR],
        opacity=0.8,
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color=PRIMARY_DARK, line_width=1.5)
    hist_layout = dict(**PLOTLY_BASE)
    hist_layout.update(height=400, xaxis_title="오차 (actual - predicted)", yaxis_title="빈도")
    fig_hist.update_layout(**hist_layout)
    st.plotly_chart(fig_hist, use_container_width=True)
