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
from demo.utils.data_loader import get_png_path

st.set_page_config(page_title="성능 대시보드", page_icon="📈", layout="wide")
st.title("📈 모델 성능 대시보드")

st.subheader("1) 최종 R² 비교")
final_df = pd.DataFrame(
    {
        "target": TARGET_ORDER,
        "지표": [TARGET_LABELS[t] for t in TARGET_ORDER],
        "R²": [FINAL_R2[t] for t in TARGET_ORDER],
    }
)
fig_final = px.bar(
    final_df,
    x="지표",
    y="R²",
    color="R²",
    color_continuous_scale="Viridis",
)
fig_final.update_layout(height=420)
st.plotly_chart(fig_final, use_container_width=True)

st.subheader("2) 개발 단계별 성능 변화")
stage_records = []
for stage, target_scores in STAGE_R2.items():
    for target in TARGET_ORDER:
        stage_records.append(
            {
                "단계": stage,
                "타겟": TARGET_LABELS[target],
                "R²": target_scores[target],
            }
        )

stage_df = pd.DataFrame(stage_records)
fig_stage = px.line(
    stage_df,
    x="단계",
    y="R²",
    color="타겟",
    markers=True,
)
fig_stage.update_layout(height=480)
st.plotly_chart(fig_stage, use_container_width=True)

st.subheader("3) 타겟별 상세 (학습 곡선 / 예측 분석)")
tabs = st.tabs([TARGET_LABELS[t] for t in TARGET_ORDER])
for target, tab in zip(TARGET_ORDER, tabs):
    with tab:
        learning_path = get_png_path("learning_curve", target)
        pred_path = get_png_path("prediction", target)

        st.markdown("**학습 곡선**")
        if learning_path.exists():
            st.image(str(learning_path), use_container_width=True)
        else:
            st.warning(f"파일 없음: {learning_path}")

        st.markdown("**예측 분석**")
        if pred_path.exists():
            st.image(str(pred_path), use_container_width=True)
        else:
            st.warning(f"파일 없음: {pred_path}")
