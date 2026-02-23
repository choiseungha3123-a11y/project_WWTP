from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import FINAL_R2, TARGET_LABELS, TARGET_ORDER

st.set_page_config(
    page_title="WWTP 수질 예측 데모",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 WWTP 수질 예측 AI")
st.caption("하수처리장 7개 수질 지표 LSTM 예측 시스템 인터랙티브 데모")

metric_cols = st.columns(4)
metric_cols[0].metric("모델 수", "7개")
metric_cols[1].metric("최고 R²", f"{max(FINAL_R2.values()):.2f}")
metric_cols[2].metric("데이터 해상도", "30분")
metric_cols[3].metric("예측 폭", "12시간")

st.markdown("### 시스템 구성")
st.write("데이터 수집 → 전처리 → LSTM 예측 → API 서비스")

st.markdown("### R² 성능 요약")
plot_df = pd.DataFrame(
    {
        "target": TARGET_ORDER,
        "label": [TARGET_LABELS[t] for t in TARGET_ORDER],
        "r2": [FINAL_R2[t] for t in TARGET_ORDER],
    }
).sort_values("r2", ascending=True)

plot_df["grade"] = pd.cut(
    plot_df["r2"],
    bins=[-999, 0.65, 0.80, 999],
    labels=["보통", "양호", "우수"],
)

fig = px.bar(
    plot_df,
    x="r2",
    y="label",
    color="grade",
    orientation="h",
    color_discrete_map={"우수": "#2E8B57", "양호": "#E6A700", "보통": "#D95F02"},
    text="r2",
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig.update_layout(height=420, xaxis_title="R²", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

st.info("좌측 사이드바에서 페이지를 이동해 상세 분석을 확인할 수 있습니다.")
