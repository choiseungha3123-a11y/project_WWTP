from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import TARGET_LABELS, TARGET_ORDER
from demo.utils.data_loader import load_comparison
from demo.utils.metrics import compute_metrics
from demo.utils.styles import (
    ACTUAL_COLOR,
    BORDER,
    CARD_BG,
    DANGER,
    PLOTLY_BASE,
    PRED_COLOR,
    PRIMARY,
    PRIMARY_DARK,
    PURPLE,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WARNING,
    apply_global_style,
    info_banner,
    kpi_card,
    section_header,
)

st.set_page_config(page_title="업체 모델 비교", page_icon="⚖️", layout="wide")
apply_global_style()

VENDOR_COLOR = PURPLE   # 업체 예측 고유 색상

st.markdown(
    f"<h2 style='color:{TEXT_PRIMARY};margin-bottom:4px'>⚖️ LSTM vs 업체 모델 비교</h2>"
    f"<p style='color:{TEXT_MUTED};margin:0 0 0.3rem'>"
    f"동일 테스트 기간의 실측값 대비 두 모델 예측을 비교합니다.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    info_banner(
        "LSTM 예측(30분 해상도)과 업체 예측(1분 해상도)을 업체 예측 기준인 "
        "1시간 단위로 리샘플링하여 비교합니다. "
        "FLUX는 LSTM 증분값을 합산(sum)하고, 업체 누적값에 차분(diff)을 적용합니다. "
        "그 외 타겟은 평균(mean)으로 리샘플링합니다."
    ),
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ── 타겟 탭 ───────────────────────────────────────────────────────────────────
tabs = st.tabs([TARGET_LABELS[t] for t in TARGET_ORDER])

for target, tab in zip(TARGET_ORDER, tabs):
    with tab:
        # ── 데이터 로드 ───────────────────────────────────────────────────────
        try:
            df = load_comparison(target)
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            continue

        has_vendor = df["vendor_pred"].notna().sum() > 0
        if not has_vendor:
            st.warning("해당 기간의 업체 예측 데이터가 없습니다.")
            continue

        df_v = df.dropna(subset=["vendor_pred"])   # 업체 예측이 있는 행만

        # ── 성능 지표 ─────────────────────────────────────────────────────────
        lstm_m  = compute_metrics(df["actual"], df["lstm_pred"])
        vendor_m = compute_metrics(df_v["actual"], df_v["vendor_pred"])

        st.markdown(section_header("성능 지표 비교"), unsafe_allow_html=True)

        def _accent(r2: float) -> str:
            return SUCCESS if r2 >= 0.80 else (WARNING if r2 >= 0.65 else DANGER)

        def _fmt(v: float, decimals: int = 3) -> str:
            """절댓값이 크거나 소수점이 길면 축약 표기."""
            if abs(v) >= 100_000:
                return f"{v/1000:.1f}K"
            if abs(v) >= 10_000:
                return f"{v:.0f}"
            if abs(v) >= 1_000:
                return f"{v:.1f}"
            return f"{v:.{decimals}f}"

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(
                f"<div style='font-size:0.8rem;font-weight:700;color:{PRIMARY};"
                f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px'>"
                f"LSTM 모델</div>",
                unsafe_allow_html=True,
            )
            mc = st.columns(4)
            for col, (label, val, accent) in zip(
                mc,
                [
                    ("R²", _fmt(lstm_m["r2"]), _accent(lstm_m["r2"])),
                    ("RMSE", _fmt(lstm_m["rmse"]), PRIMARY),
                    ("MAE", _fmt(lstm_m["mae"]), PRIMARY),
                    ("MAPE (%)", _fmt(lstm_m["mape"], 2), PRIMARY),
                ],
            ):
                col.markdown(kpi_card(label, val, accent), unsafe_allow_html=True)

        with col_r:
            st.markdown(
                f"<div style='font-size:0.8rem;font-weight:700;color:{VENDOR_COLOR};"
                f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px'>"
                f"업체 모델</div>",
                unsafe_allow_html=True,
            )
            mc2 = st.columns(4)
            for col, (label, val, accent) in zip(
                mc2,
                [
                    ("R²", _fmt(vendor_m["r2"]), _accent(vendor_m["r2"])),
                    ("RMSE", _fmt(vendor_m["rmse"]), VENDOR_COLOR),
                    ("MAE", _fmt(vendor_m["mae"]), VENDOR_COLOR),
                    ("MAPE (%)", _fmt(vendor_m["mape"], 2), VENDOR_COLOR),
                ],
            ):
                col.markdown(kpi_card(label, val, accent), unsafe_allow_html=True)

        # ── R² 델타 배너 ──────────────────────────────────────────────────────
        delta_r2 = lstm_m["r2"] - vendor_m["r2"]
        winner = "LSTM" if delta_r2 > 0 else "업체"
        winner_color = PRIMARY if delta_r2 > 0 else VENDOR_COLOR
        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
            f"border-radius:8px;padding:10px 18px;margin:12px 0;font-size:0.88rem;"
            f"color:{TEXT_PRIMARY}'>"
            f"R² 차이: <b style='color:{winner_color}'>{winner} 우세</b> "
            f"(Δ = {abs(delta_r2):.4f})"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 시계열 비교 차트 ──────────────────────────────────────────────────
        st.markdown(
            section_header("실측 vs LSTM vs 업체 시계열 (1시간 단위 테스트 기간)"),
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["actual"],
            mode="lines", name="실측",
            line=dict(color=ACTUAL_COLOR, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["lstm_pred"],
            mode="lines", name="LSTM 예측",
            line=dict(color=PRED_COLOR, width=1.5, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=df_v.index, y=df_v["vendor_pred"],
            mode="lines", name="업체 예측",
            line=dict(color=VENDOR_COLOR, width=1.5, dash="dot"),
        ))

        ts_layout = dict(**PLOTLY_BASE)
        ts_layout.update(
            height=420,
            xaxis_title="시간",
            yaxis_title="값",
            xaxis=dict(
                rangeslider=dict(visible=True, bgcolor=BORDER, thickness=0.06),
                gridcolor="#EAECF0", linecolor=BORDER, zerolinecolor=BORDER,
            ),
            yaxis=dict(gridcolor="#EAECF0", linecolor=BORDER, zerolinecolor=BORDER),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_layout(**ts_layout)
        st.plotly_chart(fig, use_container_width=True)

        # ── 오차 비교 차트 ────────────────────────────────────────────────────
        st.markdown(section_header("절대 오차 비교 (|실측 - 예측|)"), unsafe_allow_html=True)

        df["lstm_abs_err"]   = (df["actual"] - df["lstm_pred"]).abs()
        df_v["vendor_abs_err"] = (df_v["actual"] - df_v["vendor_pred"]).abs()

        c1, c2 = st.columns(2)
        with c1:
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(
                x=df.index, y=df["lstm_abs_err"],
                mode="lines", name="LSTM |오차|",
                line=dict(color=PRED_COLOR, width=1.2),
                fill="tozeroy", fillcolor=f"rgba(224,123,57,0.15)",
            ))
            fig_err.add_trace(go.Scatter(
                x=df_v.index, y=df_v["vendor_abs_err"],
                mode="lines", name="업체 |오차|",
                line=dict(color=VENDOR_COLOR, width=1.2),
                fill="tozeroy", fillcolor=f"rgba(124,58,237,0.10)",
            ))
            err_layout = dict(**PLOTLY_BASE)
            err_layout.update(
                height=320, xaxis_title="시간", yaxis_title="|오차|",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_err.update_layout(**err_layout)
            st.plotly_chart(fig_err, use_container_width=True)

        with c2:
            # 오차 분포 박스플롯
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=df["lstm_abs_err"], name="LSTM",
                marker_color=PRED_COLOR, boxmean="sd",
            ))
            fig_box.add_trace(go.Box(
                y=df_v["vendor_abs_err"], name="업체",
                marker_color=VENDOR_COLOR, boxmean="sd",
            ))
            box_layout = dict(**PLOTLY_BASE)
            box_layout.update(height=320, yaxis_title="|오차| 분포", showlegend=True)
            fig_box.update_layout(**box_layout)
            st.plotly_chart(fig_box, use_container_width=True)

        # ── 데이터 커버리지 정보 ──────────────────────────────────────────────
        with st.expander("데이터 상세 정보"):
            st.markdown(f"""
| 항목 | LSTM | 업체 |
|------|------|------|
| 예측 기간 | `{df.index[0].strftime('%Y-%m-%d %H:%M')}` ~ `{df.index[-1].strftime('%Y-%m-%d %H:%M')}` | 동일 기간 |
| 리샘플 해상도 | 30분 → 1시간 | 1분 → 1시간 |
| 샘플 수 | {len(df):,} | {df_v["vendor_pred"].notna().sum():,} (1시간 슬롯 커버) |
| 업체 커버율 | - | {df_v["vendor_pred"].notna().sum() / len(df) * 100:.1f}% |
| 평균 실측값 | {df["actual"].mean():.4f} | (동일) |
| 평균 LSTM 예측 | {df["lstm_pred"].mean():.4f} | - |
| 평균 업체 예측 | {df_v["vendor_pred"].mean():.4f} | - |
""")
            if target == "flux":
                st.info(
                    "FLUX: LSTM 30분 증분값을 합산(sum)하여 시간당 총 증분으로 변환합니다. "
                    "업체 누적값(FLUX_VU)은 1시간 단위 마지막 값 기준으로 차분(diff)하여 시간당 증분을 산출합니다."
                )
