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
