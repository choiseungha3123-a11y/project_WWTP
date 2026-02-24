from __future__ import annotations

TARGET_ORDER = ["flow", "toc", "ss", "tn", "tp", "flux", "ph"]

TARGET_LABELS = {
    "flow": "유입유량 (Flow)",
    "toc": "총유기탄소 (TOC)",
    "ss": "부유물질 (SS)",
    "tn": "총질소 (TN)",
    "tp": "총인 (TP)",
    "flux": "방류유량 (FLUX)",
    "ph": "수소이온농도 (pH)",
}

FINAL_R2 = {
    "flow": 0.8425,
    "toc": 0.5574,
    "ss": 0.6906,
    "tn": 0.9011,
    "tp": 0.6201,
    "flux": 0.6241,
    "ph": 0.8574,
}

STAGE_R2 = {
    "ML_baseline" : {
        "flow": 0.57,
        "toc": -0.66,
        "ss": -1.00,
        "tn": -0.79,
        "tp": -6.72,
        "flux": 0.96,
        "ph": 0.39,
    },
    "ML_v2" : {
        "flow": 0.72,
        "toc": -0.09,
        "ss": -2.12,
        "tn": -0.25,
        "tp": -5.09,
        "flux": 0.00,
        "ph": -0.26,
    },
    "DL": {
        "flow": 0.30,
        "toc": -1.86,
        "ss": -0.52,
        "tn": -0.16,
        "tp": -2.15,
        "flux": -0.01,
        "ph": -0.17,
    },
    "DL_Lag 피처": {
        "flow": 0.79,
        "toc": 0.29,
        "ss": 0.21,
        "tn": 0.78,
        "tp": -0.41,
        "flux": 0.23,
        "ph": 0.56,
    },
    "DL_HP 최적화": {
        "flow": 0.82,
        "toc": 0.47,
        "ss": 0.67,
        "tn": 0.90,
        "tp": 0.61,
        "flux": 0.61,
        "ph": 0.84,
    },
    "DL_최종": {
        "flow": 0.84,
        "toc": 0.55,
        "ss": 0.69,
        "tn": 0.90,
        "tp": 0.62,
        "flux": 0.62,
        "ph": 0.85,
    },
}

# ── 운영 KPI ────────────────────────────────────────────────────────────────────

# 타겟 운영 중요도 등급
# Tier 1: 규제 기준·운영 안정성 직결, 예측 신뢰도 높음
# Tier 2: 배출 허용 기준 연계, 중간 예측 성능
# Tier 3: 예측 불확실성 높음, 추세 참고 수준
OPERATIONAL_TIERS = [
    {
        "id": 1,
        "label": "핵심 지표",
        "desc": "규제 기준·운영 안정성 직결 / 예측 신뢰도 높음",
        "targets": ["tn", "ph", "flow"],
    },
    {
        "id": 2,
        "label": "주요 관리 지표",
        "desc": "배출 허용 기준 연계 / 중간 예측 성능",
        "targets": ["ss", "tp", "flux"],
    },
    {
        "id": 3,
        "label": "보조 지표",
        "desc": "예측 불확실성 높음 / 정성적 경향 참고 수준",
        "targets": ["toc"],
    },
]

# 기준초과 이벤트 탐지를 위한 운영 임계치
# None  → 실데이터 95th percentile (유량 계열)
# float → 상한 초과 기준 (단위: 각 타겟 단위)
# tuple → (하한, 상한) 범위 이탈 기준 (pH)
# 참고: 하수도법 및 물환경보전법 방류 허용 기준
OPERATIONAL_THRESHOLDS: dict[str, float | tuple | None] = {
    "flow": None,        # 유입유량 — 95th percentile (m³/h 단위)
    "toc":  15.0,        # TOC 15 mg/L (방류 허용 기준)
    "ss":   10.0,        # SS  10 mg/L (방류 허용 기준)
    "tn":   10.0,        # TN  10 mg/L (방류 허용 기준)
    "tp":   0.5,         # TP   0.5 mg/L (방류 허용 기준)
    "flux": None,        # 방류유량 — 95th percentile (m³/h 단위)
    "ph":   (5.8, 8.6),  # pH 5.8 ~ 8.6 (범위 이탈 기준)
}

OPERATIONAL_THRESHOLD_LABELS: dict[str, str] = {
    "flow": "p95 (m³/h, 데이터 기반)",
    "toc":  "15 mg/L",
    "ss":   "10 mg/L",
    "tn":   "10 mg/L",
    "tp":   "0.5 mg/L",
    "flux": "p95 (m³/h, 데이터 기반)",
    "ph":   "pH 5.8 ~ 8.6",
}
