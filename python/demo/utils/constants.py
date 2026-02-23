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
    "flow": 0.8166,
    "toc": 0.5762,
    "ss": 0.6712,
    "tn": 0.9011,
    "tp": 0.6378,
    "flux": 0.6296,
    "ph": 0.8432,
}

STAGE_R2 = {
    "베이스라인": {
        "flow": 0.30,
        "toc": -1.86,
        "ss": -0.52,
        "tn": -0.16,
        "tp": -2.15,
        "flux": -0.01,
        "ph": -0.17,
    },
    "Lag 피처": {
        "flow": 0.79,
        "toc": 0.29,
        "ss": 0.21,
        "tn": 0.78,
        "tp": -0.41,
        "flux": 0.23,
        "ph": 0.56,
    },
    "HP 최적화": {
        "flow": 0.82,
        "toc": 0.47,
        "ss": 0.67,
        "tn": 0.90,
        "tp": 0.63,
        "flux": 0.61,
        "ph": 0.84,
    },
    "최종": {
        "flow": 0.82,
        "toc": 0.58,
        "ss": 0.67,
        "tn": 0.90,
        "tp": 0.64,
        "flux": 0.63,
        "ph": 0.84,
    },
}
