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
from demo.utils.data_loader import load_experiment_results
from demo.utils.styles import (
    BG_PAGE,
    BORDER,
    CHART_PALETTE,
    PLOTLY_BASE,
    PRIMARY,
    PRIMARY_DARK,
    PRIMARY_LIGHT,
    PRIMARY_PALE,
    SUCCESS,
    TEXT_MUTED,
    apply_global_style,
    kpi_card,
    section_header,
)

st.set_page_config(page_title="하이퍼파라미터 탐색", page_icon="🧪", layout="wide")
apply_global_style()

st.markdown(
    f"<h2 style='margin-bottom:4px'>🧪 하이퍼파라미터 탐색 분석</h2>"
    f"<p style='color:{TEXT_MUTED};margin:0 0 1.2rem'>타겟별 실험 결과를 확인하고 기본값 대비 개선폭을 비교합니다.</p>",
    unsafe_allow_html=True,
)

HP_BEFORE_AFTER = {
    "flow": {
        "before": {"hidden_size": 512, "num_layers": 3, "learning_rate": "2e-4", "r2": 0.8166},
        "after": {"hidden_size": 512, "num_layers": 2, "learning_rate": "2e-3", "r2": 0.8425},
        "note": "FLOW는 학습률 상향과 레이어 축소 조합에서 개선되었습니다.",
    },
    "toc": {
        "before": {"hidden_size": 256, "num_layers": 2, "learning_rate": "1e-3", "r2": 0.4731},
        "after": {"hidden_size": 384, "num_layers": 1, "learning_rate": "1e-3", "r2": 0.5574},
        "note": "TOC는 얕은 구조(1-layer)에서 성능이 개선되었습니다.",
    },
    "ss": {
        "before": {"hidden_size": 512, "num_layers": 4, "learning_rate": "2e-4", "r2": 0.6712},
        "after": {"hidden_size": 256, "num_layers": 2, "learning_rate": "2e-3", "r2": 0.6906},
        "note": "SS는 과대 모델보다 중간 규모 모델이 더 안정적이었습니다.",
    },
    "tn": {
        "before": {"hidden_size": 512, "num_layers": 4, "learning_rate": "2e-4", "r2": 0.9011},
        "after": {"hidden_size": 512, "num_layers": 2, "learning_rate": "2e-3", "r2": 0.9019},
        "note": "TN은 이미 높은 성능으로 개선폭이 작았습니다.",
    },
    "tp": {
        "before": {"hidden_size": 512, "num_layers": 4, "learning_rate": "2e-4", "r2": 0.6252},
        "after": {"hidden_size": 384, "num_layers": 1, "learning_rate": "1e-3", "r2": 0.6378},
        "note": "TP는 단순 구조와 중간 학습률 조합이 유리했습니다.",
    },
    "flux": {
        "before": {"hidden_size": 512, "num_layers": 4, "learning_rate": "2e-4", "r2": 0.6296},
        "after": {"hidden_size": 256, "num_layers": 2, "learning_rate": "5e-4", "r2": 0.6241},
        "note": "FLUX는 탐색 결과가 기존 대비 큰 개선으로 이어지지 않았습니다.",
    },
    "ph": {
        "before": {"hidden_size": 512, "num_layers": 4, "learning_rate": "2e-4", "r2": 0.8432},
        "after": {"hidden_size": 512, "num_layers": 1, "learning_rate": "2e-3", "r2": 0.8574},
        "note": "PH는 레이어 감소와 학습률 상향 조합이 유효했습니다.",
    },
}

selected = st.selectbox("타겟 선택", options=TARGET_ORDER, format_func=lambda t: TARGET_LABELS[t])

try:
    exp_df = load_experiment_results(selected)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

exp_df = exp_df.copy()
exp_df["learning_rate"] = exp_df["learning_rate"].map(lambda x: f"{x:.0e}")

st.markdown(section_header("탐색 개요"), unsafe_allow_html=True)
ba = HP_BEFORE_AFTER[selected]
delta = ba["after"]["r2"] - ba["before"]["r2"]

mc = st.columns(5)
kpis = [
    ("실험 조합 수", f"{len(exp_df):,}", PRIMARY_LIGHT, ""),
    ("최고 R²", f"{exp_df['r2'].max():.4f}", SUCCESS, ""),
    ("최저 R²", f"{exp_df['r2'].min():.4f}", PRIMARY_DARK, ""),
    ("노트북 R²", f"{ba['before']['r2']:.4f}", PRIMARY_DARK, ""),
    ("탐색 후 R²", f"{ba['after']['r2']:.4f}", PRIMARY, f"{delta:+.4f}"),
]
for col, (label, value, accent, delta_v) in zip(mc, kpis):
    col.markdown(kpi_card(label, value, accent, delta=delta_v), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(section_header("파라미터별 R² 분포"), unsafe_allow_html=True)

param_options = {
    "hidden_size": "Hidden Size",
    "num_layers": "Num Layers",
    "learning_rate": "Learning Rate",
    "dropout": "Dropout",
    "batch_size": "Batch Size",
}

col_sel, _ = st.columns([2, 5])
with col_sel:
    sel_param = st.radio("분석 파라미터", options=list(param_options.keys()), format_func=lambda k: param_options[k])

plot_df = exp_df.copy()
plot_df[sel_param] = plot_df[sel_param].astype(str)

fig_box = px.box(
    plot_df,
    x=sel_param,
    y="r2",
    color=sel_param,
    points="all",
    labels={sel_param: param_options[sel_param], "r2": "Test R²"},
    color_discrete_sequence=CHART_PALETTE,
)
fig_box.update_traces(
    marker=dict(size=4, opacity=0.6),
    hovertemplate=f"<b>{param_options[sel_param]}</b>: %{{x}}<br>R²: <b>%{{y:.4f}}</b><extra></extra>",
)
box_layout = dict(**PLOTLY_BASE)
box_layout.update(height=400, showlegend=False, yaxis_title="Test R²")
fig_box.update_layout(**box_layout)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown(section_header("상위 10개 조합"), unsafe_allow_html=True)
display_cols = [
    "hidden_size",
    "num_layers",
    "learning_rate",
    "dropout",
    "batch_size",
    "use_attention",
    "weight_decay",
    "r2",
    "test_rmse",
    "epochs",
]
available = [c for c in display_cols if c in exp_df.columns]
top10 = exp_df.nlargest(10, "r2")[available].reset_index(drop=True)
top10.index += 1
st.dataframe(top10.style.format({"r2": "{:.4f}", "test_rmse": "{:.4f}"}), use_container_width=True)

st.markdown(section_header("노트북 vs 탐색 결과"), unsafe_allow_html=True)
before = ba["before"]
after = ba["after"]
compare_df = pd.DataFrame(
    {
        "파라미터": ["hidden_size", "num_layers", "learning_rate", "R²"],
        "노트북(기본값)": [before["hidden_size"], before["num_layers"], before["learning_rate"], f"{before['r2']:.4f}"],
        "탐색(최적값)": [after["hidden_size"], after["num_layers"], after["learning_rate"], f"{after['r2']:.4f}"],
    }
)
st.dataframe(compare_df, use_container_width=True, hide_index=True)

st.markdown(
    f"<div style='background:{BG_PAGE};border:1px solid {BORDER};border-left:4px solid {PRIMARY_DARK};"
    f"border-radius:8px;padding:12px 16px;font-size:0.88rem;color:{PRIMARY_DARK};margin:0.5rem 0'>"
    f"메모: {ba['note']}"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(section_header("전 타겟 최적 R² 비교"), unsafe_allow_html=True)

summary_records = []
for t in TARGET_ORDER:
    ba_t = HP_BEFORE_AFTER[t]
    summary_records.append({"타겟": TARGET_LABELS[t], "버전": "노트북", "R²": ba_t["before"]["r2"]})
    summary_records.append({"타겟": TARGET_LABELS[t], "버전": "탐색 후", "R²": ba_t["after"]["r2"]})

summary_df = pd.DataFrame(summary_records)
fig_sum = px.bar(
    summary_df,
    x="타겟",
    y="R²",
    color="버전",
    barmode="group",
    color_discrete_map={"노트북": PRIMARY_PALE, "탐색 후": PRIMARY},
    text="R²",
    labels={"R²": "Test R²"},
)
fig_sum.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
    marker_line_width=0,
    hovertemplate="<b>%{x}</b><br>%{fullData.name}: <b>%{y:.4f}</b><extra></extra>",
)
sum_layout = dict(**PLOTLY_BASE)
sum_layout.update(
    height=380,
    yaxis_range=[0, 1.08],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig_sum.update_layout(**sum_layout)
st.plotly_chart(fig_sum, use_container_width=True)
