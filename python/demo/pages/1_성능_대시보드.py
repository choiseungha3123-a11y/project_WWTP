from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import FINAL_R2, STAGE_R2, TARGET_LABELS, TARGET_ORDER

from demo.utils.styles import (
    BORDER,
    CHART_PALETTE,
    DANGER,
    GRADE_COLORS,
    PLOTLY_BASE,
    PRIMARY,
    PRIMARY_DARK,
    PRIMARY_LIGHT,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WARNING,
    apply_global_style,
    kpi_card,
    section_header,
)

st.set_page_config(page_title="성능 대시보드", page_icon="📈", layout="wide")
apply_global_style()

st.markdown(
    f"<h2 style='color:{TEXT_PRIMARY};margin-bottom:4px'>📈 모델 성능 대시보드</h2>"
    f"<p style='color:{TEXT_MUTED};margin:0 0 1.2rem'>ML baseline, ML V2, DL(LSTM) 성능을 비교합니다.</p>",
    unsafe_allow_html=True,
)

best_r2 = max(FINAL_R2.values())
worst_r2 = min(FINAL_R2.values())
avg_r2 = sum(FINAL_R2.values()) / len(FINAL_R2)
above_08 = sum(1 for v in FINAL_R2.values() if v >= 0.80)

kpi_cols = st.columns(4)
kpis = [
    ("LSTM 최고 R²", f"{best_r2:.4f}", SUCCESS),
    ("LSTM 평균 R²", f"{avg_r2:.4f}", PRIMARY),
    ("LSTM 최저 R²", f"{worst_r2:.4f}", DANGER),
    ("R² ≥ 0.80 타겟", f"{above_08} / {len(FINAL_R2)}", WARNING),
]
for col, (label, value, accent) in zip(kpi_cols, kpis):
    col.markdown(kpi_card(label, value, accent), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(section_header("1) LSTM 최종 R² 비교"), unsafe_allow_html=True)

final_df = pd.DataFrame(
    {
        "target": TARGET_ORDER,
        "지표": [TARGET_LABELS[t] for t in TARGET_ORDER],
        "R²": [FINAL_R2[t] for t in TARGET_ORDER],
    }
).sort_values("R²", ascending=True)

final_df["등급"] = pd.cut(
    final_df["R²"],
    bins=[-999, 0.65, 0.80, 999],
    labels=["보통", "양호", "우수"],
)

fig_final = px.bar(
    final_df,
    x="R²",
    y="지표",
    color="등급",
    orientation="h",
    color_discrete_map=GRADE_COLORS,
    text="R²",
)
fig_final.update_traces(
    texttemplate="<b>%{text:.4f}</b>",
    textposition="outside",
    marker_line_width=0,
)
layout_final = dict(**PLOTLY_BASE)
layout_final.update(
    height=340,
    xaxis_title="R² (결정계수)",
    yaxis_title="",
    xaxis_range=[0, 1.12],
    legend_title_text="등급",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_final.update_layout(**layout_final)
st.plotly_chart(fig_final, use_container_width=True)

st.markdown(section_header("2) 개발 단계별 성능 변화"), unsafe_allow_html=True)

stage_records = []
for stage, target_scores in STAGE_R2.items():
    for target in TARGET_ORDER:
        stage_records.append({"단계": stage, "타겟": TARGET_LABELS[target], "R²": target_scores[target]})

stage_df = pd.DataFrame(stage_records)
fig_stage = px.line(
    stage_df,
    x="단계",
    y="R²",
    color="타겟",
    markers=True,
    color_discrete_sequence=CHART_PALETTE,
)
fig_stage.add_hline(y=0, line_dash="dash", line_color=BORDER, line_width=1)
fig_stage.add_hline(
    y=0.8,
    line_dash="dot",
    line_color=SUCCESS,
    line_width=1.5,
    annotation_text="목표 R²=0.80",
    annotation_font_color=SUCCESS,
)
stage_layout = dict(**PLOTLY_BASE)
stage_layout.update(
    height=460,
    xaxis_title="개발 단계",
    yaxis_title="R²",
    legend=dict(orientation="v", x=1.01, y=1),
)
fig_stage.update_layout(**stage_layout)
st.plotly_chart(fig_stage, use_container_width=True)
