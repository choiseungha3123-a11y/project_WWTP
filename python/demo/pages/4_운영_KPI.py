from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import (
    FINAL_R2,
    OPERATIONAL_THRESHOLD_LABELS,
    OPERATIONAL_THRESHOLDS,
    OPERATIONAL_TIERS,
    TARGET_LABELS,
    TARGET_ORDER,
)
from demo.utils.data_loader import load_predictions
from demo.utils.metrics import compute_metrics, compute_operational_metrics
from demo.utils.styles import (
    BG_MID,
    BORDER,
    CARD_BG,
    CHART_PALETTE,
    DANGER,
    PLOTLY_BASE,
    PRIMARY,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WARNING,
    apply_global_style,
    kpi_card,
    section_header,
)

st.set_page_config(page_title="운영 KPI", page_icon="🎯", layout="wide")
apply_global_style()

st.markdown(
    f"<h2 style='color:{TEXT_PRIMARY};margin-bottom:4px'>🎯 운영 KPI 대시보드</h2>"
    f"<p style='color:{TEXT_MUTED};margin:0 0 1.2rem'>"
    "타겟별 운영 중요도 · 예측품질(R², MAE, RMSE, MAPE) · "
    "기준초과 탐지 성능(Recall / Precision / F1 / FPR)을 종합합니다.</p>",
    unsafe_allow_html=True,
)

# ── 1. 운영 중요도 등급 ─────────────────────────────────────────────────────────
st.markdown(section_header("타겟 운영 중요도 등급"), unsafe_allow_html=True)

TIER_COLORS = {1: SUCCESS, 2: WARNING, 3: DANGER}

tier_cols = st.columns(3)
for col, tier in zip(tier_cols, OPERATIONAL_TIERS):
    tier_id = tier["id"]
    color   = TIER_COLORS[tier_id]

    # 타겟별 행 (약어 + R²)
    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 0;border-bottom:1px solid {BORDER};">'
        f'<span style="font-size:0.85rem;color:{TEXT_PRIMARY};font-weight:500">'
        f'{TARGET_LABELS[t].split("(")[1][:-1]}'          # 약어만 추출
        f'</span>'
        f'<span style="font-size:0.85rem;font-weight:700;color:{color}">'
        f'R² {FINAL_R2[t]:.4f}'
        f'</span>'
        f'</div>'
        for t in tier["targets"]
    )

    with col:
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-top:3px solid {color};'
            f'border-radius:10px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
            f'<div style="font-size:0.68rem;font-weight:700;color:{color};'
            f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px">'
            f'Tier {tier_id}</div>'
            f'<div style="font-size:1rem;font-weight:700;color:{TEXT_PRIMARY};margin-bottom:4px">'
            f'{tier["label"]}</div>'
            f'<div style="font-size:0.78rem;color:{TEXT_MUTED};margin-bottom:12px">'
            f'{tier["desc"]}</div>'
            f'{rows_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    f'<div style="background:{BG_MID};border:1px solid {BORDER};border-radius:8px;'
    f'padding:10px 16px;font-size:0.82rem;color:{TEXT_MUTED};margin:0.75rem 0">'
    f'<b style="color:{TEXT_PRIMARY}">분류 기준</b>  —  '
    f'<span style="color:{SUCCESS}">■ Tier 1</span>: R² ≥ 0.80 · 규제 직결  &nbsp;|&nbsp; '
    f'<span style="color:{WARNING}">■ Tier 2</span>: 0.60 ≤ R² &lt; 0.80 · 주요 관리  &nbsp;|&nbsp; '
    f'<span style="color:{DANGER}">■ Tier 3</span>: R² &lt; 0.60 · 불확실성 높음'
    f'</div>',
    unsafe_allow_html=True,
)

# ── 데이터 로드 (전 타겟) ───────────────────────────────────────────────────────
all_pred_metrics: dict[str, dict] = {}
all_op_metrics:   dict[str, dict] = {}

for target in TARGET_ORDER:
    try:
        df  = load_predictions(target).copy()
        all_pred_metrics[target] = compute_metrics(df["actual"], df["predicted"])

        thr = OPERATIONAL_THRESHOLDS[target]
        if thr is None:
            op = compute_operational_metrics(df["actual"], df["predicted"], thr=None, kind="percentile")
        elif isinstance(thr, tuple):
            op = compute_operational_metrics(df["actual"], df["predicted"], thr=thr, kind="range")
        else:
            op = compute_operational_metrics(df["actual"], df["predicted"], thr=thr, kind="upper")
        all_op_metrics[target] = op

    except FileNotFoundError:
        pass  # 데이터 파일이 없는 타겟은 생략

# 타겟-Tier 매핑
target_tier = {t: tier["id"] for tier in OPERATIONAL_TIERS for t in tier["targets"]}

# ── 2. 예측품질 KPI 종합 ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    section_header("예측품질 KPI", "테스트셋 기준 · 전 타겟 비교"),
    unsafe_allow_html=True,
)

if all_pred_metrics:
    pred_rows = []
    for target in TARGET_ORDER:
        if target not in all_pred_metrics:
            continue
        m   = all_pred_metrics[target]
        tid = target_tier.get(target, "-")
        pred_rows.append({
            "Tier":         f"T{tid}",
            "타겟":          TARGET_LABELS[target],
            "R²":           m["r2"],
            "MAE":          m["mae"],
            "RMSE":         m["rmse"],
            "MAPE (%)":     m["mape"],
            "5% 이내 (%)":  m["within_5pct"],
        })

    pred_df = pd.DataFrame(pred_rows)

    def _r2_style(val: float) -> str:
        if val >= 0.80:
            return f"color:{SUCCESS};font-weight:700"
        if val >= 0.65:
            return f"color:{WARNING};font-weight:700"
        return f"color:{DANGER};font-weight:700"

    styled = (
        pred_df.style
        .applymap(_r2_style, subset=["R²"])
        .format({
            "R²":          "{:.4f}",
            "MAE":         "{:.4f}",
            "RMSE":        "{:.4f}",
            "MAPE (%)":    "{:.2f}",
            "5% 이내 (%)": "{:.1f}",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # 가로 막대 차트 (R² 비교)
    bar_df = pred_df.sort_values("R²")
    bar_colors = [TIER_COLORS[target_tier.get(
        TARGET_ORDER[pred_df[pred_df["타겟"] == row["타겟"]].index[0]], 1
    )] for _, row in bar_df.iterrows()]

    fig_r2 = go.Figure()
    for _, row in bar_df.iterrows():
        t_key  = next((t for t in TARGET_ORDER if TARGET_LABELS[t] == row["타겟"]), None)
        color  = TIER_COLORS.get(target_tier.get(t_key, 1), PRIMARY)
        fig_r2.add_trace(go.Bar(
            x=[row["R²"]],
            y=[row["타겟"]],
            orientation="h",
            marker_color=color,
            marker_line_width=0,
            text=[f"<b>{row['R²']:.4f}</b>"],
            textposition="outside",
            showlegend=False,
        ))

    # 기준선
    fig_r2.add_vline(x=0.80, line_dash="dot", line_color=SUCCESS,  line_width=1.5,
                     annotation_text="R²=0.80", annotation_font_color=SUCCESS)
    fig_r2.add_vline(x=0.65, line_dash="dot", line_color=WARNING,  line_width=1.5,
                     annotation_text="R²=0.65", annotation_font_color=WARNING)

    r2_layout = dict(**PLOTLY_BASE)
    r2_layout.update(height=300, xaxis_range=[0, 1.12], xaxis_title="R² (결정계수)", yaxis_title="")
    fig_r2.update_layout(**r2_layout)
    st.plotly_chart(fig_r2, use_container_width=True)

# ── 3. 운영품질 KPI ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    section_header(
        "운영품질 KPI",
        "기준초과 이벤트 탐지율(재현율) · 오경보율(FPR) · 정밀도 · F1",
    ),
    unsafe_allow_html=True,
)

# 임계치 안내
with st.expander("운영 임계치 기준 보기 (하수도법·물환경보전법 방류 허용 기준 참고)"):
    thr_rows = [
        {"타겟": TARGET_LABELS[t], "임계치": OPERATIONAL_THRESHOLD_LABELS[t]}
        for t in TARGET_ORDER
    ]
    st.dataframe(pd.DataFrame(thr_rows), use_container_width=True, hide_index=True)

if all_op_metrics:
    op_rows = []
    for target in TARGET_ORDER:
        if target not in all_op_metrics:
            continue
        op  = all_op_metrics[target]
        tid = target_tier.get(target, "-")

        thr_val = op["threshold_used"]
        if isinstance(thr_val, tuple):
            thr_str = f"범위 이탈"
        elif op["kind"] == "upper":
            thr_str = f">{thr_val:.2f}"
        else:
            thr_str = f"p90={thr_val:.2f}"

        op_rows.append({
            "Tier":             f"T{tid}",
            "타겟":              TARGET_LABELS[target],
            "임계치":            thr_str,
            "이벤트 수":         op["event_count"],
            "재현율 (Recall)":   op["recall"],
            "정밀도 (Precision)": op["precision"],
            "F1":                op["f1"],
            "오경보율 (FPR)":    op["fpr"],
        })

    op_df = pd.DataFrame(op_rows)

    def _recall_style(val) -> str:
        if np.isnan(val):
            return ""
        if val >= 0.80:
            return f"color:{SUCCESS};font-weight:700"
        if val >= 0.50:
            return f"color:{WARNING};font-weight:700"
        return f"color:{DANGER};font-weight:700"

    def _fpr_style(val) -> str:
        if np.isnan(val):
            return ""
        if val <= 0.10:
            return f"color:{SUCCESS};font-weight:700"
        if val <= 0.25:
            return f"color:{WARNING};font-weight:700"
        return f"color:{DANGER};font-weight:700"

    def _fmt(val) -> str:
        return f"{val:.3f}" if not np.isnan(val) else "—"

    styled_op = (
        op_df.style
        .applymap(_recall_style, subset=["재현율 (Recall)", "정밀도 (Precision)", "F1"])
        .applymap(_fpr_style,    subset=["오경보율 (FPR)"])
        .format({
            "재현율 (Recall)":    _fmt,
            "정밀도 (Precision)": _fmt,
            "F1":                 _fmt,
            "오경보율 (FPR)":     _fmt,
        })
    )
    recall_col = next(col for col in op_df.columns if "Recall" in col)
    precision_col = next(col for col in op_df.columns if "Precision" in col)
    fpr_col = next(col for col in op_df.columns if "FPR" in col)
    target_col = next(
        col
        for col in op_df.columns
        if col != "Tier" and op_df[col].dropna().isin(TARGET_LABELS.values()).all()
    )

    label_to_target = {label: key for key, label in TARGET_LABELS.items()}
    op_display_df = op_df.copy()

    for idx, row in op_df.iterrows():
        target_key = label_to_target.get(row[target_col])
        op_meta = all_op_metrics.get(target_key, {}) if target_key else {}

        actual_event_count = int(op_meta.get("event_count", 0))
        pred_event_count = int(op_meta.get("tp", 0) + op_meta.get("fp", 0))

        if pd.isna(row[recall_col]):
            op_display_df.at[idx, recall_col] = (
                "실제 이벤트 0 (계산 불능)" if actual_event_count == 0 else "계산 불능"
            )

        if pd.isna(row[precision_col]):
            op_display_df.at[idx, precision_col] = (
                "예측 이벤트 0 (계산 불능)" if pred_event_count == 0 else "계산 불능"
            )

        if pd.isna(row["F1"]):
            if actual_event_count == 0 and pred_event_count == 0:
                f1_msg = "실제/예측 이벤트 0 (계산 불능)"
            elif actual_event_count == 0:
                f1_msg = "실제 이벤트 0 (계산 불능)"
            elif pred_event_count == 0:
                f1_msg = "예측 이벤트 0 (계산 불능)"
            else:
                f1_msg = "계산 불능"
            op_display_df.at[idx, "F1"] = f1_msg

        if pd.isna(row[fpr_col]):
            op_display_df.at[idx, fpr_col] = "계산 불능"

    def _recall_style_display(val) -> str:
        if isinstance(val, str) or pd.isna(val):
            return ""
        if val >= 0.80:
            return f"color:{SUCCESS};font-weight:700"
        if val >= 0.50:
            return f"color:{WARNING};font-weight:700"
        return f"color:{DANGER};font-weight:700"

    def _fpr_style_display(val) -> str:
        if isinstance(val, str) or pd.isna(val):
            return ""
        if val <= 0.10:
            return f"color:{SUCCESS};font-weight:700"
        if val <= 0.25:
            return f"color:{WARNING};font-weight:700"
        return f"color:{DANGER};font-weight:700"

    def _fmt_display(val) -> str:
        if isinstance(val, str):
            return val
        return f"{val:.3f}" if not np.isnan(val) else "-"

    styled_op = (
        op_display_df.style
        .applymap(_recall_style_display, subset=[recall_col, precision_col, "F1"])
        .applymap(_fpr_style_display, subset=[fpr_col])
        .format({
            recall_col: _fmt_display,
            precision_col: _fmt_display,
            "F1": _fmt_display,
            fpr_col: _fmt_display,
        })
    )
    st.dataframe(styled_op, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TP/FP/FN/TN 혼동 행렬 요약 ────────────────────────────────────────────
    st.markdown(section_header("혼동 행렬 요약", "이벤트 탐지 결과 분류"), unsafe_allow_html=True)

    cm_rows = []
    for target in TARGET_ORDER:
        if target not in all_op_metrics:
            continue
        op = all_op_metrics[target]
        cm_rows.append({
            "타겟":       TARGET_LABELS[target],
            "TP (정탐)":  op["tp"],
            "FP (오경보)": op["fp"],
            "FN (미탐)":  op["fn"],
            "TN (정상)":  op["tn"],
            "전체 이벤트": op["event_count"],
        })

    cm_df = pd.DataFrame(cm_rows)

    def _tp_style(val) -> str:
        return f"color:{SUCCESS};font-weight:700" if val > 0 else ""

    def _fp_fn_style(val) -> str:
        return f"color:{DANGER};font-weight:700" if val > 0 else f"color:{SUCCESS};font-weight:700"

    styled_cm = (
        cm_df.style
        .applymap(_tp_style,     subset=["TP (정탐)"])
        .applymap(_fp_fn_style,  subset=["FP (오경보)", "FN (미탐)"])
    )
    st.dataframe(styled_cm, use_container_width=True, hide_index=True)
