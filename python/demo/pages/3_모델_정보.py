from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import TARGET_LABELS, TARGET_ORDER
from demo.utils.data_loader import load_feature_names
from src.config import FLOW_CONFIG, TMS_TARGETS

st.set_page_config(page_title="모델 정보", page_icon="🧠", layout="wide")
st.title("🧠 모델 아키텍처 및 피처 정보")

rows = []
for target in TARGET_ORDER:
    if target == "flow":
        cfg = FLOW_CONFIG
    else:
        cfg = TMS_TARGETS[target]

    features = load_feature_names(target)
    head = "4-layer" if cfg.get("deep_head", False) else "3-layer"
    rows.append(
        {
            "타겟": TARGET_LABELS[target],
            "hidden": cfg["hidden_size"],
            "layers": cfg["num_layers"],
            "attention": "✓" if cfg.get("use_attention", False) else "✗",
            "head": head,
            "피처수": len(features),
        }
    )

st.subheader("모델 구조 요약")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("타겟별 추천 피처 목록")
selected_target = st.selectbox(
    "타겟 선택",
    options=TARGET_ORDER,
    format_func=lambda t: TARGET_LABELS[t],
)
feature_names = load_feature_names(selected_target)
st.write(f"선택된 피처 수: {len(feature_names)}")
st.dataframe(
    pd.DataFrame({"feature_name": feature_names}),
    use_container_width=True,
    hide_index=True,
)

st.subheader("LSTM 아키텍처 다이어그램")
attention_text = "사용" if (FLOW_CONFIG if selected_target == "flow" else TMS_TARGETS[selected_target]).get("use_attention", False) else "미사용"
head_text = "4-layer" if (FLOW_CONFIG if selected_target == "flow" else TMS_TARGETS[selected_target]).get("deep_head", False) else "3-layer"
st.code(
    f"""Input (batch, seq=48, n_features)
    ↓
LSTM (num_layers, hidden_size)
    ↓
[Attention: {attention_text}]
    ↓
FC Head ({head_text})
    ↓
Output (1)""",
    language="text",
)
