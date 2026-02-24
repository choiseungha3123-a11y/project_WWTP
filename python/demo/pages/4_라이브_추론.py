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
from demo.utils.styles import (
    ACTUAL_COLOR,
    BORDER,
    CARD_BG,
    PLOTLY_BASE,
    PRIMARY,
    PRIMARY_DARK,
    PRIMARY_LIGHT,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_global_style,
    info_banner,
    kpi_card,
    section_header,
)

st.set_page_config(page_title="라이브 추론", page_icon="⚡", layout="wide")
apply_global_style()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="margin-bottom:4px">⚡ 라이브 추론</h2>'
    f'<p style="color:{TEXT_MUTED};margin:0 0 1.2rem">'
    '저장된 실제 모델 · 스케일러를 로드해 즉시 추론합니다. (입력 시퀀스: 48 step = 24시간)</p>',
    unsafe_allow_html=True,
)

selected_target = st.selectbox(
    "타겟 선택",
    options=TARGET_ORDER,
    format_func=lambda t: TARGET_LABELS[t],
)

with st.spinner("모델 / 스케일러 로딩 중..."):
    artifacts = load_runtime_artifacts(selected_target)

feature_names = artifacts["feature_names"]

st.markdown(
    info_banner(f"필수 입력 피처: <b>{len(feature_names)}개</b> &nbsp;|&nbsp; window size: <b>48 스텝 (24h)</b>"),
    unsafe_allow_html=True,
)

# ── 템플릿 후보 수집 ──────────────────────────────────────────────────────────
template_root = ROOT_DIR / "demo" / "live_infer_templates"
template_candidates: list[Path] = []
if template_root.exists():
    for p in template_root.glob(f"{selected_target}_live_infer_template*.csv"):
        template_candidates.append(p)
    real_dir = template_root / "real_segment"
    if real_dir.exists():
        for p in real_dir.glob(f"{selected_target}_live_infer_template*.csv"):
            template_candidates.append(p)
template_candidates = sorted(set(template_candidates), key=lambda x: x.name, reverse=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(section_header("입력 설정"), unsafe_allow_html=True)

# ── 입력 방식 + CSV 소스 ──────────────────────────────────────────────────────
col_mode, col_gap, col_source = st.columns([2, 0.12, 2])
chosen_template: Path | None = None
source_mode: str | None      = None

with col_mode:
    st.markdown(
        f'<div style="font-size:0.75rem;font-weight:700;color:{TEXT_MUTED};'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">입력 방식</div>',
        unsafe_allow_html=True,
    )
    input_mode = st.radio(
        "입력 방식",
        options=["CSV 업로드 (권장)", "수동 입력 (48스텝 동일값)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if input_mode == "CSV 업로드 (권장)":
        template_df = pd.DataFrame([{c: 0.0 for c in feature_names} for _ in range(48)])
        csv_bytes = template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  템플릿 CSV 다운로드 (48 × n_features)",
            data=csv_bytes,
            file_name=f"{selected_target}_live_infer_template.csv",
            mime="text/csv",
        )

with col_gap:
    pass

with col_source:
    if input_mode == "CSV 업로드 (권장)":
        st.markdown(
            f'<div style="font-size:0.75rem;font-weight:700;color:{TEXT_MUTED};'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">CSV 소스</div>',
            unsafe_allow_html=True,
        )
        source_mode = st.radio(
            "CSV 소스",
            options=["생성된 템플릿에서 선택", "직접 업로드"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if source_mode == "생성된 템플릿에서 선택":
            if not template_candidates:
                st.warning("선택 가능한 템플릿 파일이 없습니다. 직접 업로드를 사용해 주세요.")
            else:
                chosen_template = st.selectbox(
                    "템플릿 파일",
                    options=template_candidates,
                    format_func=lambda p: str(p.relative_to(ROOT_DIR)),
                )

# ── 데이터 처리 ───────────────────────────────────────────────────────────────
input_df = None

if input_mode == "CSV 업로드 (권장)":
    if source_mode == "생성된 템플릿에서 선택":
        if chosen_template is not None:
            raw_df = pd.read_csv(chosen_template)
            st.markdown(
                f'<div style="font-size:0.82rem;color:{TEXT_MUTED};margin:12px 0 6px">'
                f'📄 <b>{chosen_template.relative_to(ROOT_DIR)}</b></div>',
                unsafe_allow_html=True,
            )
            with st.expander("데이터 미리보기 (상위 5행)"):
                st.dataframe(raw_df.head(5), use_container_width=True)
            input_df = raw_df
    else:
        uploaded = st.file_uploader("CSV 업로드", type=["csv"])
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
            with st.expander("업로드 미리보기 (상위 5행)"):
                st.dataframe(raw_df.head(5), use_container_width=True)
            input_df = raw_df

    if input_df is not None:
        st.caption(f"입력 데이터: {len(input_df):,} rows × {len(input_df.columns):,} columns")

else:
    st.markdown(
        f'<div style="font-size:0.84rem;color:{TEXT_MUTED};margin-bottom:12px">'
        '피처별 값을 입력하면 48스텝 전체에 동일값으로 적용됩니다.</div>',
        unsafe_allow_html=True,
    )
    values = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            values[feat] = st.number_input(feat, value=0.0, step=0.1, format="%.6f")
    input_df = build_sequence_from_single_row(feature_names, values, n_steps=48)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("▶  추론 실행", type="primary", use_container_width=True):
    try:
        if input_df is None:
            st.warning("먼저 입력 데이터를 준비해 주세요.")
            st.stop()

        aligned_df           = validate_and_align_input(input_df, feature_names)
        one_step, multi_preds = run_inference(aligned_df, artifacts, horizon_steps=24)

        st.markdown(section_header("추론 결과"), unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card("0.5h 예측값", f"{one_step:.4f}", PRIMARY), unsafe_allow_html=True)
        c2.markdown(kpi_card("6.0h 예측값", f"{multi_preds[11]:.4f}", PRIMARY_LIGHT), unsafe_allow_html=True)
        c3.markdown(kpi_card("12.0h 예측값", f"{multi_preds[-1]:.4f}", SUCCESS), unsafe_allow_html=True)
        c4.markdown(kpi_card("예측 스텝 수", "24 steps", PRIMARY_DARK), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        traj_df = build_trajectory_df(multi_preds)

        fig = px.line(
            traj_df, x="hour", y="prediction",
            markers=True,
            labels={"hour": "예측 시점 (hour)", "prediction": "예측값"},
            color_discrete_sequence=[ACTUAL_COLOR],
        )
        fig.update_traces(
            marker=dict(size=7, color="white", line=dict(color=ACTUAL_COLOR, width=2)),
            line=dict(width=2.5),
            hovertemplate="t+<b>%{x:.1f}h</b>  →  <b>%{y:.4f}</b><extra></extra>",
        )
        traj_layout = dict(**PLOTLY_BASE)
        traj_layout.update(
            height=380,
            title=dict(
                text=f"<b>12시간 Autoregressive 예측 궤적</b>  ({TARGET_LABELS[selected_target]})",
                font=dict(size=14),
                x=0,
            ),
            xaxis_title="예측 시점 (hour)",
            yaxis_title="예측값",
        )
        fig.update_layout(**traj_layout)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("예측 결과 전체 표 보기"):
            st.dataframe(traj_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"추론 실패: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.caption(
    "참고: CSV는 48행이어야 하며, 추천 피처 컬럼 외 컬럼은 무시됩니다. "
    "누락 컬럼은 0으로 자동 보정됩니다."
)
