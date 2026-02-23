from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import TARGET_LABELS, TARGET_ORDER
from demo.utils.data_loader import load_predictions
from demo.utils.metrics import compute_metrics

st.set_page_config(page_title="예측 분석", page_icon="🔎", layout="wide")
st.title("🔎 인터랙티브 예측 분석")

selected_target = st.selectbox(
    "타겟 선택",
    options=TARGET_ORDER,
    format_func=lambda t: TARGET_LABELS[t],
)

df = load_predictions(selected_target)
df = df.copy()
df["sample_idx"] = np.arange(len(df))
df["error"] = df["actual"] - df["predicted"]

metrics = compute_metrics(df["actual"], df["predicted"])

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("R²", f"{metrics['r2']:.4f}")
mc2.metric("RMSE", f"{metrics['rmse']:.4f}")
mc3.metric("MAE", f"{metrics['mae']:.4f}")
mc4.metric("샘플 수", f"{len(df):,}")

st.markdown("### 실측 vs 예측 시계열")
fig_ts = go.Figure()
fig_ts.add_trace(
    go.Scatter(
        x=df["sample_idx"],
        y=df["actual"],
        mode="lines",
        name="실측",
        line=dict(color="#1f77b4", width=2),
    )
)
fig_ts.add_trace(
    go.Scatter(
        x=df["sample_idx"],
        y=df["predicted"],
        mode="lines",
        name="예측",
        line=dict(color="#d62728", width=2, dash="dash"),
    )
)
fig_ts.update_layout(
    xaxis_title="샘플 인덱스",
    yaxis_title="값",
    height=440,
    xaxis=dict(rangeslider=dict(visible=True)),
    hovermode="x unified",
)
st.plotly_chart(fig_ts, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("### 산점도 (실측 vs 예측)")
    fig_scatter = px.scatter(
        df,
        x="actual",
        y="predicted",
        color="error",
        color_continuous_scale="RdBu",
        opacity=0.7,
    )
    min_v = float(min(df["actual"].min(), df["predicted"].min()))
    max_v = float(max(df["actual"].max(), df["predicted"].max()))
    fig_scatter.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            name="y=x",
            line=dict(color="black", dash="dot"),
            showlegend=True,
        )
    )
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

with c2:
    st.markdown("### 오차 분포")
    fig_hist = px.histogram(df, x="error", nbins=50)
    fig_hist.update_layout(height=420, xaxis_title="오차 (actual - predicted)")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        f"오차 평균: {df['error'].mean():.4f} | 오차 표준편차: {df['error'].std():.4f}"
    )
