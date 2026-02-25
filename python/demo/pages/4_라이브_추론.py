from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import TARGET_LABELS, TARGET_ORDER
from demo.utils.live_infer import (
    build_sequence_from_single_row,
    build_trajectory_df,
    load_runtime_artifacts,
    run_inference,
    validate_and_align_input,
)

st.set_page_config(page_title="라이브 추론", page_icon="⚡", layout="wide")
st.title("⚡ 라이브 추론")
st.caption("저장된 실제 모델/스케일러를 로드해 즉시 추론합니다. (입력 시퀀스: 48 step)")

selected_target = st.selectbox(
    "타겟 선택",
    options=TARGET_ORDER,
    format_func=lambda t: TARGET_LABELS[t],
)

with st.spinner("모델/스케일러 로딩 중..."):
    artifacts = load_runtime_artifacts(selected_target)

feature_names = artifacts["feature_names"]
st.info(f"필수 입력 피처 수: {len(feature_names)}개 | window size: 48")

input_mode = st.radio(
    "입력 방식",
    options=["CSV 업로드 (권장)", "수동 입력 (48스텝 동일값)"],
    horizontal=True,
)

input_df = None

if input_mode == "CSV 업로드 (권장)":
    template_df = pd.DataFrame([{c: 0.0 for c in feature_names} for _ in range(48)])
    csv_bytes = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "템플릿 CSV 다운로드 (48 x n_features)",
        data=csv_bytes,
        file_name=f"{selected_target}_live_infer_template.csv",
        mime="text/csv",
    )

    source_mode = st.radio(
        "CSV 소스",
        options=["생성된 템플릿에서 선택", "직접 업로드"],
        horizontal=True,
    )

    template_root = ROOT_DIR / "demo" / "live_infer_templates"
    template_candidates = []
    if template_root.exists():
        for p in template_root.glob(f"{selected_target}_live_infer_template*.csv"):
            template_candidates.append(p)
        real_dir = template_root / "real_segment"
        if real_dir.exists():
            for p in real_dir.glob(f"{selected_target}_live_infer_template*.csv"):
                template_candidates.append(p)

    # 파일명 기준 정렬 (real_segment 날짜 템플릿이 상단 오도록 역순)
    template_candidates = sorted(set(template_candidates), key=lambda x: x.name, reverse=True)

    if source_mode == "생성된 템플릿에서 선택":
        if not template_candidates:
            st.warning("선택 가능한 템플릿 파일이 없습니다. 직접 업로드를 사용해 주세요.")
        else:
            chosen_template = st.selectbox(
                "템플릿 파일 선택",
                options=template_candidates,
                format_func=lambda p: str(p.relative_to(ROOT_DIR)),
            )
            raw_df = pd.read_csv(chosen_template)
            st.write(f"선택한 파일: `{chosen_template.relative_to(ROOT_DIR)}`")
            st.write("템플릿 미리보기")
            st.dataframe(raw_df.head(5), use_container_width=True)
            input_df = raw_df

    else:
        uploaded = st.file_uploader("CSV 업로드", type=["csv"])
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
            st.write("업로드 미리보기")
            st.dataframe(raw_df.head(5), use_container_width=True)
            input_df = raw_df

    if input_df is not None:
        st.caption(f"현재 입력 데이터: {len(input_df)} rows x {len(input_df.columns)} columns")
else:
    st.write("피처별 값을 입력하면 48스텝 전체에 동일 값으로 적용됩니다.")
    values = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            values[feat] = st.number_input(feat, value=0.0, step=0.1, format="%.6f")
    input_df = build_sequence_from_single_row(feature_names, values, n_steps=48)

if st.button("실행", type="primary", use_container_width=True):
    try:
        if input_df is None:
            st.warning("먼저 입력 데이터를 준비해 주세요.")
            st.stop()

        aligned_df = validate_and_align_input(input_df, feature_names)
        one_step, multi_preds = run_inference(aligned_df, artifacts, horizon_steps=24)

        c1, c2 = st.columns(2)
        c1.metric("0.5h 예측값", f"{one_step:.6f}")
        c2.metric("12.0h 예측값", f"{multi_preds[-1]:.6f}")

        traj_df = build_trajectory_df(multi_preds)

        fig = px.line(
            traj_df,
            x="hour",
            y="prediction",
            markers=True,
            title="12시간(30분 간격) Autoregressive 예측 궤적",
        )
        fig.update_layout(xaxis_title="예측 시점 (hour)", yaxis_title="예측값", height=420)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("예측 결과 표 보기"):
            st.dataframe(traj_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"추론 실패: {e}")

st.markdown("---")
st.caption(
    "참고: CSV는 48행이어야 하며, 추천 피처 컬럼 외 컬럼은 무시됩니다. "
    "누락 컬럼은 0으로 자동 보정됩니다."
)
