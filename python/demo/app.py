from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import FINAL_R2
from demo.utils.styles import (
    BG_PAGE,
    BORDER,
    PRIMARY,
    PRIMARY_DARK,
    apply_global_style,
    hero_banner,
    kpi_card,
    section_header,
)

st.set_page_config(page_title="WWTP 수질 예측 데모", page_icon="📊", layout="wide")
apply_global_style()

st.markdown(
    hero_banner(
        "WWTP 수질 예측 AI",
        "하수처리장 7개 수질 지표에 대한 LSTM 기반 예측 대시보드",
    ),
    unsafe_allow_html=True,
)

kpi_cols = st.columns(4)
kpis = [
    ("모델 수", "7개", PRIMARY),
    ("최고 R²", f"{max(FINAL_R2.values()):.4f}", PRIMARY),
    ("데이터 해상도", "30분", PRIMARY_DARK),
    ("예측 길이", "12시간", PRIMARY_DARK),
]
for col, (label, value, accent) in zip(kpi_cols, kpis):
    col.markdown(kpi_card(label, value, accent), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(section_header("시스템 구성"), unsafe_allow_html=True)

steps = [
    ("1", "데이터 수집", "1분 단위 원천 데이터를 집계"),
    ("2", "전처리", "30분 리샘플링 및 결측 처리"),
    ("3", "LSTM 추론", "12시간 미래 시점 예측"),
    ("4", "API 서비스", "FastAPI 엔드포인트 제공"),
    ("5", "시각화", "Streamlit 대시보드 제공"),
]

arch_cols = st.columns(len(steps))
for i, (col, (idx, title, desc)) in enumerate(zip(arch_cols, steps)):
    connector = (
        f'<div style="position:absolute;right:-16px;top:50%;transform:translateY(-50%);'
        f'font-size:1.1rem;color:{BORDER};z-index:10">&rsaquo;</div>'
        if i < len(steps) - 1
        else ""
    )
    with col:
        st.markdown(
            (
                f'<div style="position:relative;background:{BG_PAGE};border:1px solid {BORDER};'
                f'border-top:3px solid {PRIMARY};border-radius:12px;padding:20px 14px;'
                f'text-align:center;min-height:140px;">'
                f"{connector}"
                f'<div style="font-size:1.5rem;font-weight:800;color:{PRIMARY};margin-bottom:8px">{idx}</div>'
                f'<div style="font-weight:700;font-size:0.95rem;color:{PRIMARY};margin-bottom:6px">{title}</div>'
                f'<div style="font-size:0.8rem;color:{PRIMARY_DARK};line-height:1.5">{desc}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f'<div style="background:{BG_PAGE};border:1px solid {BORDER};border-left:4px solid {PRIMARY};'
    f'border-radius:8px;padding:12px 16px;font-size:0.88rem;color:{PRIMARY_DARK};margin-top:0.5rem">'
    "좌측 사이드바에서 페이지를 이동해 상세 분석을 확인할 수 있습니다."
    "</div>",
    unsafe_allow_html=True,
)
