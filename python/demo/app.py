from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.utils.constants import FINAL_R2
from demo.utils.styles import (
    BG_MID,
    BG_PAGE,
    BORDER,
    CARD_BG,
    DANGER,
    PRIMARY,
    SECONDARY,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WARNING,
    apply_global_style,
    hero_banner,
    kpi_card,
    section_header,
)

st.set_page_config(page_title="WWTP 수질 예측 데모", page_icon="📊", layout="wide")
apply_global_style()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    hero_banner(
        "WWTP 수질 예측 AI",
        "하수처리장 유입유량 및 6개 수질 지표에 대한 LSTM 기반 예측·진단 대시보드",
    ),
    unsafe_allow_html=True,
)

# ── Summary KPI ───────────────────────────────────────────────────────────────
kpi_cols = st.columns(4)
kpis = [
    ("예측 타겟 수", "7개 지표", PRIMARY),
    ("최고 R²", f"{max(FINAL_R2.values()):.4f}", SUCCESS),
    ("데이터 해상도", "30분 단위", WARNING),
    ("예측 리드타임", "최대 12시간", PRIMARY),
]
for col, (label, value, accent) in zip(kpi_cols, kpis):
    col.markdown(kpi_card(label, value, accent), unsafe_allow_html=True)

# ── 프로젝트 개요 ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(section_header("프로젝트 개요"), unsafe_allow_html=True)

st.markdown(
    f'<p style="font-size:0.92rem;color:{TEXT_MUTED};line-height:1.75;margin:0 0 1.2rem">'
    "본 시스템은 하수처리장의 <b style='color:{tp}'>미래 유입량·수질</b>을 사전에 예측하고, "
    "운영 기준을 초과할 가능성이 있을 경우 <b style='color:{tp}'>선제 경고 및 이상 진단</b>을 제공하는 "
    "AI 기반 의사결정 지원 플랫폼입니다. "
    "1분 단위 공정 데이터와 기상 데이터를 결합해 30분 해상도로 향후 12시간을 Autoregressive 방식으로 예측합니다."
    "</p>".format(tp=TEXT_PRIMARY),
    unsafe_allow_html=True,
)

# 3-pillar feature cards
def _pillar_card(icon: str, title: str, desc: str, bullets: list[str], accent: str) -> str:
    li = "".join(
        f'<li style="margin-bottom:4px;line-height:1.55">{b}</li>' for b in bullets
    )
    return (
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-top:3px solid {accent};'
        f'border-radius:10px;padding:22px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.05)">'
        f'<div style="font-size:1.6rem;margin-bottom:10px;line-height:1">{icon}</div>'
        f'<div style="font-weight:700;font-size:0.97rem;color:{TEXT_PRIMARY};margin-bottom:4px">{title}</div>'
        f'<div style="font-size:0.8rem;color:{TEXT_MUTED};margin-bottom:12px">{desc}</div>'
        f'<ul style="font-size:0.8rem;color:{TEXT_PRIMARY};padding-left:16px;margin:0;line-height:1">'
        f'{li}'
        f'</ul>'
        f'</div>'
    )

pillar_cols = st.columns(3)
pillars = [
    (
        "🔮", "예측 (Forecasting)",
        "유입유량 및 TMS 수질 지표 12시간 선행 예측",
        PRIMARY,
        [
            "유입유량(Flow) + TMS 6개 지표, 총 7개 타겟 모델",
            "30분 단위 · Autoregressive 24 스텝 예측",
            "LSTM + Self-Attention · 피처 수 30~120개",
            "Sliding Window 48 스텝(24시간) 입력",
        ],
    ),
    (
        "📊", "분석 (Analytics)",
        "단계별 성능 비교 및 예측 오차 심층 분석",
        SECONDARY,
        [
            "ML baseline → ML v2 → DL 단계별 R² 추이 비교",
            "실측 vs 예측 시계열 · 오차 분포 시각화",
            "피처 엔지니어링 파이프라인 9단계 가시화",
            "하이퍼파라미터 탐색 실험 결과 분석",
        ],
    ),
    (
        "🚨", "진단 (Diagnosis)",
        "기준초과 이벤트 탐지 및 운영 리스크 평가",
        DANGER,
        [
            "방류 기준(TN 20mg/L · TP 2mg/L 등) 초과 탐지",
            "Recall / Precision / F1 / FPR 기반 탐지 성능 평가",
            "최대 12시간 리드타임으로 선제 대응 지원",
            "타겟별 운영 중요도 Tier 분류 (핵심 / 주요 / 보조)",
        ],
    ),
]
for col, (icon, title, desc, accent, bullets) in zip(pillar_cols, pillars):
    col.markdown(_pillar_card(icon, title, desc, bullets, accent), unsafe_allow_html=True)

# ── 개발 목표 ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(section_header("개발 목표"), unsafe_allow_html=True)

goal_left, goal_right = st.columns(2)

with goal_left:
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};'
        f'border-radius:10px;padding:20px 22px;box-shadow:0 1px 4px rgba(0,0,0,0.05);min-height:260px">'
        f'<div style="font-size:0.68rem;font-weight:700;color:{TEXT_MUTED};'
        f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:12px">예측 정확도 목표</div>'

        f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px">'
        f'<span style="font-size:2rem;font-weight:800;color:{PRIMARY}">95%</span>'
        f'<span style="font-size:0.85rem;color:{TEXT_MUTED}">유입유량 (Flow)</span>'
        f'</div>'
        f'<div style="font-size:0.78rem;color:{TEXT_MUTED};margin-bottom:14px">'
        f'MAE·RMSE·MAPE 기준 — |실측 − 예측| / 실측 ≤ 5% 비율</div>'

        f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px">'
        f'<span style="font-size:2rem;font-weight:800;color:{SUCCESS}">90%</span>'
        f'<span style="font-size:0.85rem;color:{TEXT_MUTED}">TMS 수질 지표 6개</span>'
        f'</div>'
        f'<div style="font-size:0.78rem;color:{TEXT_MUTED}">'
        f'TOC · SS · TN · TP · FLUX · pH 지표별 개별 평가</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with goal_right:
    goals = [
        ("📐", "시계열 패턴 분석",
         "시간별·일별·계절별 유입 변동 패턴 파악 및 기상 상관 분석"),
        ("🔔", "실시간 이상 탐지",
         "사용자 정의 기준 기반 기준초과 이벤트 조기 감지 및 경보"),
        ("⚡", "FastAPI 서비스화",
         "REST API 엔드포인트로 운영 시스템과 즉시 연동 가능"),
    ]
    items_html = "".join(
        f'<div style="display:flex;gap:12px;align-items:flex-start;padding:10px 0;'
        f'{"border-bottom:1px solid " + BORDER + ";" if i < len(goals) - 1 else ""}">'
        f'<span style="font-size:1.1rem;margin-top:1px">{icon}</span>'
        f'<div>'
        f'<div style="font-size:0.85rem;font-weight:600;color:{TEXT_PRIMARY};margin-bottom:2px">{t}</div>'
        f'<div style="font-size:0.78rem;color:{TEXT_MUTED};line-height:1.5">{d}</div>'
        f'</div></div>'
        for i, (icon, t, d) in enumerate(goals)
    )
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};'
        f'border-radius:10px;padding:20px 22px;box-shadow:0 1px 4px rgba(0,0,0,0.05);min-height:230px">'
        f'<div style="font-size:0.68rem;font-weight:700;color:{TEXT_MUTED};'
        f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:12px">시스템 목표</div>'
        f'{items_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── 시스템 구성 ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(section_header("시스템 구성"), unsafe_allow_html=True)

steps = [
    ("1", "데이터 수집", "1분 단위 공정·기상 원천 데이터"),
    ("2", "전처리", "30분 리샘플링 · 피처 엔지니어링"),
    ("3", "LSTM 추론", "Autoregressive 12시간 예측"),
    ("4", "API 서비스", "FastAPI /predict/flow·tms"),
    ("5", "대시보드", "Streamlit 실시간 시각화"),
]

arch_cols = st.columns(len(steps))
for i, (col, (idx, title, desc)) in enumerate(zip(arch_cols, steps)):
    connector = (
        f'<div style="position:absolute;right:-14px;top:50%;transform:translateY(-50%);'
        f'font-size:1rem;color:{BORDER};z-index:10">›</div>'
        if i < len(steps) - 1
        else ""
    )
    with col:
        st.markdown(
            (
                f'<div style="position:relative;background:{BG_PAGE};border:1px solid {BORDER};'
                f'border-top:3px solid {PRIMARY};border-radius:10px;padding:18px 12px;'
                f'text-align:center;min-height:120px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
                f"{connector}"
                f'<div style="font-size:1.25rem;font-weight:800;color:{PRIMARY};margin-bottom:6px">{idx}</div>'
                f'<div style="font-weight:700;font-size:0.9rem;color:{TEXT_PRIMARY};margin-bottom:4px">{title}</div>'
                f'<div style="font-size:0.75rem;color:{TEXT_MUTED};line-height:1.45">{desc}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

# ── 페이지 안내 ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

nav_items = [
    ("📈", "1. 성능 대시보드", "ML → DL 단계별 R² 추이 및 학습곡선"),
    ("🔎", "2. 예측 분석",    "실측 vs 예측 시계열 · 오차 분포 분석"),
    ("🧠", "3. 모델 정보",    "LSTM 아키텍처 · 피처 엔지니어링 파이프라인"),
    ("🎯", "4. 운영 KPI",     "Tier 등급 · 탐지 정확도 · 리드타임 분석"),
    ("⚡", "5. 라이브 추론",  "CSV 업로드 후 즉시 추론 결과 확인"),
]

nav_html = "".join(
    f'<div style="display:flex;gap:10px;align-items:flex-start">'
    f'<span style="font-size:1rem">{icon}</span>'
    f'<div>'
    f'<span style="font-size:0.83rem;font-weight:700;color:{TEXT_PRIMARY}">{name}</span>'
    f'<span style="font-size:0.78rem;color:{TEXT_MUTED}"> — {desc}</span>'
    f'</div></div>'
    for icon, name, desc in nav_items
)

st.markdown(
    f'<div style="background:{BG_MID};border:1px solid {BORDER};border-left:3px solid {PRIMARY};'
    f'border-radius:8px;padding:16px 20px">'
    f'<div style="font-size:0.72rem;font-weight:700;color:{TEXT_MUTED};'
    f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:10px">사이드바 페이지 안내</div>'
    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:7px">'
    f'{nav_html}'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)
