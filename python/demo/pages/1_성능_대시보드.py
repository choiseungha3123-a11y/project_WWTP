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

st.divider()

# ── ML baseline/V2 데이터 ──────────────────────────────────────────────
ML_FLOW_MODELS = ["HistGBR", "Lasso", "Ridge", "XGBoost", "RandomForest"]

ML_FLOW = {
    "모델": ML_FLOW_MODELS,
    "baseline R²":   [0.5701, 0.5487, 0.5324, 0.3738, 0.2932],
    "V2 R²":   [0.6897, 0.4908, 0.4665, 0.7162, 0.7201],
    "baseline RMSE": [52.84,  54.14,  55.11,  63.78,  67.75],
    "V2 RMSE": [38.94,  49.88,  51.05,  37.24,  36.98],
}

ML_TMS_TARGETS = ["FLUX_VU", "PH_VU", "TOC_VU", "TN_VU", "SS_VU", "TP_VU"]
ML_TMS_LABELS  = {
    "FLUX_VU": "방류유량 (FLUX)",
    "PH_VU":   "pH",
    "TOC_VU":  "TOC",
    "TN_VU":   "TN",
    "SS_VU":   "SS",
    "TP_VU":   "TP",
}

ML_TMS = {
    "타겟":    [ML_TMS_LABELS[t] for t in ML_TMS_TARGETS],
    "baseline R²":  [0.9668,  0.3943, -0.6692, -0.7972, -1.0001, -6.7276],
    "V2 R²":  [0.0041, -0.2654, -0.0922, -0.2593, -2.1258, -5.0961],
}

ML_DATA_USAGE = {
    "데이터셋": ["FLOW", "TMS"],
    "baseline 사용률(%)": [4.2, 4.2],
    "V2 사용률(%)": [98.4, 90.4],
}

# ── 3) ML FLOW 모델 baseline/V2 비교 ───────────────────────────────────
st.subheader("3) ML FLOW 예측 — baseline vs V2 모델별 R² 비교")
flow_df = pd.DataFrame(ML_FLOW)
flow_melt = flow_df.melt(id_vars="모델", value_vars=["baseline R²", "V2 R²"],
                         var_name="버전", value_name="R²")
fig_flow = px.bar(
    flow_melt,
    x="모델",
    y="R²",
    color="버전",
    barmode="group",
    color_discrete_map={"baseline R²": "#636EFA", "V2 R²": "#00CC96"},
)
fig_flow.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
fig_flow.update_layout(height=400, yaxis_title="Test R²")
st.plotly_chart(fig_flow, use_container_width=True)

# ── 4) ML TMS 타겟별 baseline/V2 비교 ──────────────────────────────────
st.subheader("4) ML TMS 예측 — 타겟별 baseline vs V2 R²")
st.caption("baseline: RandomForest 기준 / V2: XGBoost 기준 (각 버전 최고 모델)")
tms_df = pd.DataFrame(ML_TMS)
tms_melt = tms_df.melt(id_vars="타겟", value_vars=["baseline R²", "V2 R²"],
                        var_name="버전", value_name="R²")
fig_tms = px.bar(
    tms_melt,
    x="타겟",
    y="R²",
    color="버전",
    barmode="group",
    color_discrete_map={"baseline R²": "#636EFA", "V2 R²": "#00CC96"},
)
fig_tms.add_hline(y=0, line_dash="dash", line_color="red",
                  annotation_text="R²=0 (평균 예측 수준)", annotation_position="top right")
fig_tms.update_layout(height=400, yaxis_title="Test R²")
st.plotly_chart(fig_tms, use_container_width=True)

# ── 5) 데이터 사용률 개선 ─────────────────────────────────────────
st.subheader("5) 데이터 사용률 개선 (baseline → V2)")
usage_df = pd.DataFrame(ML_DATA_USAGE)
usage_melt = usage_df.melt(id_vars="데이터셋", value_vars=["baseline 사용률(%)", "V2 사용률(%)"],
                            var_name="버전", value_name="사용률(%)")
fig_usage = px.bar(
    usage_melt,
    x="데이터셋",
    y="사용률(%)",
    color="버전",
    barmode="group",
    color_discrete_map={"baseline 사용률(%)": "#EF553B", "V2 사용률(%)": "#00CC96"},
    text_auto=".1f",
)
fig_usage.update_layout(height=360, yaxis_range=[0, 105])
st.plotly_chart(fig_usage, use_container_width=True)

# ── baseline/V2 주요 인사이트 ───────────────────────────────────────────
with st.expander("baseline/V2 분석 요약 보기", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### baseline 주요 결과")
        st.markdown("""
**FLOW (Q_in)**
- HistGBR 최고: R² **0.57**, RMSE 52.84
- 선형 모델(Lasso·Ridge)도 준수 (R² 0.53~0.55)
- 트리 기반 모델 심각한 과적합 (Train 0.97 → Test 0.29)

**TMS 수질 변수**
- FLUX_VU만 예측 가능 (R² **0.9668**)
- 나머지 변수 전부 R² < 0 (평균보다 못함)
- 원인: 576개 샘플만 사용(전체 4.2%), 피처-타겟 관계 부족

**데이터 손실**
- 13,848 샘플 → dropna 후 576개 (95.8% 손실)
        """)

    with col2:
        st.markdown("#### V2 주요 결과")
        st.markdown("""
**FLOW (Q_in)**
- RandomForest: R² 0.29 → **0.72** (+145%)
- XGBoost: R² 0.37 → **0.72** (+92%)
- 데이터 사용률 4.2% → **98.4%** (23배 증가)
- 신규 도메인 피처(탱크 수위 차분, 야간 여부) 효과적

**TMS 수질 변수**
- FLUX_VU 급격히 악화: R² 0.97 → **0.00** (lag 24h 제거 영향)
- 나머지 변수 여전히 R² < 0
- 데이터 사용률은 90.4%로 대폭 개선
        """)

st.divider()
st.subheader("6) DL 타겟별 상세 (학습 곡선 / 예측 분석)")
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
