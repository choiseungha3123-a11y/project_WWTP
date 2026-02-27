#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, io
# Windows cp949 콘솔에서 UTF-8 출력 강제 (✓ ✗ 등 특수문자 포함)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
"""
toc_extend_experiment.py
========================
TOC LSTM 하이퍼파라미터 랜덤 탐색 실험

고정값:
  window_size=60, horizon=1, mode="toc", num_epochs=100

탐색 대상 (SEARCH_SPACE):
  - 모델 구조   : hidden_size, num_layers, dropout, deep_head, bidirectional
  - Attention   : use_attention, num_heads
  - 학습        : learning_rate, batch_size, optimizer, weight_decay,
                  loss_function, patience, gradient_clip, shuffle_train
  - 손실 파라미터: huber_delta, bias_alpha, bias_beta
  - 스케줄러    : scheduler_patience, scheduler_factor
  - 스케일러    : scaler_type
  - 특성 선택   : stability_ratio, min_features

사용법:
    python toc_extend_experiment.py                  # 기본 50회 탐색
    python toc_extend_experiment.py --n_trials 100
    python toc_extend_experiment.py --force_preprocess   # 캐시 무시
"""

import sys
import os
import argparse
import random
import pickle
import json
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import zscore

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- 프로젝트 경로 ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import notebook.feature.WF_feature_selection as wf_fs
import notebook.feature.feature_engineering as feat_eng

# --- 경로 설정 ----------------------------------------------------------------
DATA_DIR       = BASE_DIR / "data"
MODEL_DIR      = BASE_DIR / "model"
RESULTS_DIR    = BASE_DIR / "results" / "DL"
FEATURE_DIR    = DATA_DIR / "features"
EXPERIMENT_DIR = RESULTS_DIR / "toc_experiment"

# --- 고정 하이퍼파라미터 ------------------------------------------------------
MODE        = "toc"
TARGET_COL  = "TOC_VU"
WINDOW_SIZE = 60
HORIZON     = 1
NUM_EPOCHS  = 200
RANDOM_SEED = 42
TIME_COL    = "SYS_TIME"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# GPU 메모리 예산: 전체 VRAM의 70% 또는 CPU 모드 시 무제한
if DEVICE == "cuda":
    _total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    GPU_MEMORY_BUDGET_MB = _total_vram_mb * 0.70   # 전체의 70%
else:
    GPU_MEMORY_BUDGET_MB = float("inf")

SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

# --- 탐색 공간 ----------------------------------------------------------------
# window_size / horizon 은 고정. 나머지 전체를 탐색.
SEARCH_SPACE = {
    # -- 모델 구조 --------------------------------------------------------------
    "hidden_size":        [128, 256, 384, 512],
    "num_layers":         [1, 2, 3, 4],
    "dropout":            [0.1, 0.2, 0.3, 0.4],
    "deep_head":          [True, False],     # 헤드 깊이: True=4-layer, False=3-layer
    # bidirectional: 고정 False (기존 버전)

    # -- Attention --------------------------------------------------------------
    "use_attention":      [True, False],
    "num_heads":          [2, 4, 8],         # use_attention=True 일 때만 의미

    # -- 학습 설정 --------------------------------------------------------------
    "learning_rate":      [1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
    "batch_size":         [512, 1024, 2048],
    # optimizer: 고정 "adam" (기존 버전)
    "weight_decay":       [0.0, 1e-5, 1e-4, 1e-3],
    # loss_function: 고정 "mse" (기존 버전)
    "patience":           [10, 20, 30],
    "gradient_clip":      [0.5, 1.0, 2.0, None],   # None = 클리핑 없음
    "shuffle_train":      [True, False],

    # 손실 함수 파라미터: 고정값 사용 (huber_delta=1.0, bias_alpha=0.5, bias_beta=0.3)

    # -- 스케줄러 (ReduceLROnPlateau) --------------------------------------------
    "scheduler_patience": [3, 5, 10],
    "scheduler_factor":   [0.3, 0.5, 0.7],

    # scaler_type: 고정 "standard" (기존 버전)

    # -- 특성 선택 --------------------------------------------------------------
    "stability_ratio":    [0.3, 0.4, 0.5, 0.6],    # WalkForward 안정성 임계
    "min_features":       [10, 15, 20, 30],         # 최소 특성 수 하한
}


# =============================================================================
# 1. 데이터 로딩 / 전처리
# =============================================================================

@dataclass
class ImputationConfig:
    short_term_hours:  int = 3
    medium_term_hours: int = 12
    long_term_hours:   int = 48
    ewma_span:         int = 6


@dataclass
class OutlierConfig:
    method:           str   = "iqr"
    iqr_threshold:    float = 1.5
    zscore_threshold: float = 3.0
    require_both:     bool  = True


def load_data():
    dfs = {}
    dfs["flow"] = pd.read_csv(DATA_DIR / "actual/FLOW_extended.csv")
    dfs["tms"]  = pd.read_csv(DATA_DIR / "actual/TMS_extended.csv")
    for sid in ["368", "541", "569"]:
        df = pd.read_csv(DATA_DIR / f"actual/AWS_{sid}.csv")
        if "datetime" in df.columns:
            tc = df["datetime"].copy()
            df = df.drop(columns=["datetime", "YYMMDDHHMI", "STN"], errors="ignore")
            df = df.add_suffix(f"_{sid}")
            df[TIME_COL] = tc
        else:
            df = df.drop(columns=["YYMMDDHHMI", "STN"], errors="ignore")
            df = df.add_suffix(f"_{sid}")
        dfs[f"aws{sid}"] = df
    return dfs


def set_datetime_index(df, time_col=TIME_COL):
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    return out.dropna(subset=[time_col]).set_index(time_col).sort_index()


def align_data(dfs):
    return {name: set_datetime_index(df).resample("1min").ffill()
            for name, df in dfs.items()}


def merge_data(aligned):
    def dedup(df):
        df = df.sort_index()
        return df[~df.index.duplicated(keep="last")] if df.index.has_duplicates else df

    v = {k: dedup(df) for k, df in aligned.items()}
    return {
        "flow": pd.concat([v["flow"], v["aws368"], v["aws541"], v["aws569"]],
                           axis=1, join="inner"),
        "tms":  pd.concat([v["flow"], v["tms"], v["aws368"], v["aws541"], v["aws569"]],
                           axis=1, join="inner"),
    }


def impute_missing(df, freq="1min", cfg=None):
    if cfg is None:
        cfg = ImputationConfig()
    freq_h = pd.Timedelta(freq).total_seconds() / 3600
    df_out = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].copy()
        lim_s = max(1, int(cfg.short_term_hours / freq_h))
        s_ff = s.ffill(limit=lim_s)
        still = s_ff.isna()
        if still.sum() > 0:
            span_m = max(1, int(cfg.ewma_span / freq_h))
            ewma = s_ff.ewm(span=span_m, adjust=False).mean()
            lim_m = max(1, int(cfg.medium_term_hours / freq_h))
            grp = (still != still.shift()).cumsum()
            lens = still.groupby(grp).transform("sum")
            mask_m = still & (lens <= lim_m)
            s_ff[mask_m] = ewma[mask_m]
        still2 = s_ff.isna()
        if still2.sum() > 0:
            span_l = max(1, int(cfg.ewma_span * 4 / freq_h))
            ewma_l = s_ff.ewm(span=span_l, adjust=False).mean()
            s_ff[still2] = ewma_l[still2]
        df_out[col] = s_ff
    return df_out


def _domain_outliers(series, col):
    out = pd.Series(False, index=series.index)
    if not pd.api.types.is_numeric_dtype(series):
        return out
    reg = {"TOC_VU": (0, 30), "SS_VU": (0, 20), "PH_VU": (4.3, 10.0),
           "TN_VU": (0, 20), "TP_VU": (0, 1.0), "FLUX_VU": (0, 12.0)}
    phy = {"level_TankA": (0, 10), "level_TankB": (0, 10),
           "TA": (-30, 45), "HM": (0, 100), "TD": (-40, 35)}
    if col in reg:
        lo, hi = reg[col];  return (series < lo) | (series > hi)
    if col in phy:
        lo, hi = phy[col];  return (series < lo) | (series > hi)
    if "RN_" in col:
        return (series < 0) | (series > 300)
    if "flow" in col.lower() or "flux" in col.lower():
        v = series.dropna()
        if len(v): return (series < 0) | (series > v.quantile(0.99) * 3)
    else:
        v = series.dropna()
        if len(v): return (series < 0) | (series > v.quantile(0.999) * 2)
    return out


def process_outliers(df, cfg=None, ewma_span=12):
    if cfg is None:
        cfg = OutlierConfig()
    df_out = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].copy()
        dom = _domain_outliers(s, col)
        if cfg.method == "iqr":
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            sta = (s < Q1 - cfg.iqr_threshold * IQR) | (s > Q3 + cfg.iqr_threshold * IQR)
        else:
            vm = ~s.isna()
            sta = pd.Series(False, index=s.index)
            if vm.sum() > 0:
                z = np.abs(zscore(s[vm]))
                sta[vm] = z > cfg.zscore_threshold
        final = dom & sta if cfg.require_both else dom | sta
        if final.sum() > 0:
            sc = s.copy()
            sc[final] = np.nan
            s[final] = sc.ewm(span=ewma_span, adjust=False).mean()[final]
        df_out[col] = s
    return df_out


def resample_data(df, freq="30min"):
    num = df.select_dtypes(include=[np.number]).columns
    agg = {c: ("sum" if c.startswith("RN_") or c.startswith("AR_") or c == "FLUX_VU"
               else "mean")
           for c in num}
    return df[num].resample(freq).agg(agg)


def preprocess_data_full():
    """
    TOC 전처리 전체 실행 → (X_all, y_df, feature_names, wf_selector) 반환.
    실험 캐시 생성용으로 한 번만 호출됨.
    """
    print("=" * 70)
    print("전처리 시작 (최초 실행 시 WalkForward 포함 약 5~15분 소요)")
    print("=" * 70)

    dfs     = load_data()
    aligned = align_data(dfs)

    # FLUX 차분 처리 (누적 → 증분)
    if "tms" in aligned and "FLUX_VU" in aligned["tms"].columns:
        flux = aligned["tms"]["FLUX_VU"].copy()
        fd   = flux.diff()
        fd[fd < 0] = flux[fd < 0]
        fd.iloc[0] = 0
        aligned["tms"]["FLUX_VU"] = fd.clip(lower=0)
        print(f"  [OK] FLUX_VU 차분 처리 완료")

    merged = merge_data(aligned)

    # 결측치 / 이상치 / 리샘플링
    tms = merged["tms"]
    tms = impute_missing(tms, freq="1min")
    tms = process_outliers(tms)
    tms = resample_data(tms, freq="30min")

    df = tms.loc[:, ~tms.columns.duplicated()].copy()

    # 타겟 lag 특성 추가 (분리 전)
    df = feat_eng.add_target_lag_features(df, [TARGET_COL], min_lag=2)
    print(f"  [OK] 타겟 lag 특성 추가")

    # 타겟 분리
    y_df = df[[TARGET_COL]].copy()
    base = df.drop(columns=[TARGET_COL])

    # 데이터 누수 방지 (TOC의 경우 safe_vars = 모든 다른 TMS 변수 → 제거 없음)
    all_proc = ["TOC_VU", "SS_VU", "TN_VU", "TP_VU", "FLUX_VU", "PH_VU"]
    safe     = set(feat_eng.DATA_LEAKAGE_CONFIG["toc"]["safe_process_features"])
    unsafe   = [v for v in all_proc if v != TARGET_COL and v not in safe and v in base.columns]
    if unsafe:
        base = base.drop(columns=unsafe)
        print(f"  [OK] 누수 방지: {unsafe} 제외")

    # 특성 엔지니어링
    base = feat_eng.add_rain_features(base)
    base = feat_eng.add_station_agg_rain_features(base)
    base = feat_eng.add_weather_features(base)
    base = feat_eng.add_process_features(base)
    base = feat_eng.add_temporal_features(base)
    base = feat_eng.add_time_features(base)

    if "weekday" not in base.columns and "dayofweek" in base.columns:
        base["weekday"] = base["dayofweek"]
    if "iso_week" not in base.columns and isinstance(base.index, pd.DatetimeIndex):
        base["iso_week"] = base.index.isocalendar().week.astype(int).to_numpy()
    if "hour_x_weekday" not in base.columns:
        if "hour" in base.columns and "weekday" in base.columns:
            base["hour_x_weekday"] = base["hour"] * base["weekday"]

    df_fe = base.join(y_df, how="left").dropna()

    X_all          = df_fe.drop(columns=[TARGET_COL]).values
    y_df_out       = df_fe[[TARGET_COL]]
    feature_names  = df_fe.drop(columns=[TARGET_COL]).columns.tolist()

    print(f"\n전체 특성: {len(feature_names)}, 샘플: {len(X_all)}")

    # Walk-Forward Feature Selector
    print("\nWalkForwardFeatureSelector 실행 중...")
    wf_selector = wf_fs.WalkForwardFeatureSelector(
        X=X_all,
        y=y_df_out,
        feature_names=feature_names,
        n_splits=5,
        train_size=700,
        val_size=200,
        test_size=200,
        window_step=200,
    )
    wf_selector.run(model_type="rf", verbose=True)
    print("  [OK] WalkForward 완료")

    return X_all, y_df_out, feature_names, wf_selector


def get_feature_subset(wf_selector, feature_names_all, stability_ratio, min_features):
    """
    stability_ratio 기준으로 추천 특성 선택.
    min_features 미달 시 중요도 상위로 보충.
    """
    cnt = np.zeros(wf_selector.n_features)
    for idx_arr in wf_selector.best_features_per_fold:
        cnt[idx_arr] += 1

    threshold  = wf_selector.n_splits * stability_ratio
    rec        = np.where(cnt >= threshold)[0]

    if len(rec) < min_features:
        imp_mean = np.mean(wf_selector.feature_importance_folds, axis=0)
        top_k    = np.argsort(imp_mean)[::-1][:min_features]
        rec      = np.sort(np.unique(np.concatenate([rec, top_k])))

    return rec, [feature_names_all[i] for i in rec]


# =============================================================================
# 2. 스케일러
# =============================================================================

class StandardScaler:
    def fit(self, x):
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_  = x.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, x):         return (x - self.mean_) / self.std_
    def inverse_transform(self, x): return x * self.std_ + self.mean_


class RobustScaler:
    """IQR 기반 스케일러 — 분포 이동에 강건"""
    def fit(self, x):
        self.median_ = np.median(x, axis=0, keepdims=True)
        q75 = np.percentile(x, 75, axis=0, keepdims=True)
        q25 = np.percentile(x, 25, axis=0, keepdims=True)
        self.iqr_ = (q75 - q25) + 1e-8
        return self
    def transform(self, x):         return (x - self.median_) / self.iqr_
    def inverse_transform(self, x): return x * self.iqr_ + self.median_


# =============================================================================
# 3. 데이터셋
# =============================================================================

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X, y, window_size, horizon):
        self.X          = torch.as_tensor(X, dtype=torch.float32)
        self.y          = torch.as_tensor(y, dtype=torch.float32)
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)
        self.window_size = window_size
        self.horizon     = horizon
        self.max_start   = len(self.X) - window_size - horizon + 1

    def __len__(self):
        return max(0, self.max_start)

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.window_size]
        y_t   = self.y[idx + self.window_size + self.horizon - 1]
        return x_seq, y_t.reshape(-1)


# =============================================================================
# 4. 모델 — 확장 LSTMRegressor
# =============================================================================

class LSTMRegressor(nn.Module):
    """
    확장된 LSTMRegressor

    추가 파라미터:
      num_heads    : MultiheadAttention 헤드 수 (기존 4 고정 → 탐색)
      deep_head    : True=4-layer 헤드, False=3-layer 헤드 (기존 MODE 판별 → 파라미터)
      bidirectional: 양방향 LSTM (기존 미구현 → 활성화)
    """
    def __init__(
        self,
        n_features:    int,
        hidden_size:   int   = 512,
        num_layers:    int   = 1,
        dropout:       float = 0.2,
        out_size:      int   = 1,
        use_attention: bool  = True,
        num_heads:     int   = 4,
        deep_head:     bool  = True,
        bidirectional: bool  = False,
    ):
        super().__init__()
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        lstm_out = hidden_size * (2 if bidirectional else 1)

        if use_attention:
            # num_heads 가 embed_dim 을 나눠야 함 — 불가능 시 자동 강하
            valid = [h for h in [num_heads, 4, 2, 1] if lstm_out % h == 0]
            actual_heads = valid[0]
            self.layer_norm1 = nn.LayerNorm(lstm_out)
            self.attention   = nn.MultiheadAttention(
                embed_dim   = lstm_out,
                num_heads   = actual_heads,
                dropout     = dropout,
                batch_first = True,
            )
            self.layer_norm2 = nn.LayerNorm(lstm_out)

        # 예측 헤드 (deep_head + hidden_size 기준)
        if deep_head and lstm_out >= 256:
            # 4-layer: lstm_out → /2 → /4 → /8 → out
            self.head = nn.Sequential(
                nn.Linear(lstm_out,       lstm_out // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(lstm_out // 2,  lstm_out // 4), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(lstm_out // 4,  lstm_out // 8), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(lstm_out // 8,  out_size),
            )
        elif lstm_out >= 256:
            # 3-layer
            self.head = nn.Sequential(
                nn.Linear(lstm_out,       lstm_out // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(lstm_out // 2,  lstm_out // 4), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(lstm_out // 4,  out_size),
            )
        else:
            # 2-layer (small hidden_size)
            self.head = nn.Sequential(
                nn.Linear(lstm_out,       lstm_out // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(lstm_out // 2,  out_size),
            )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)       # (batch, seq, lstm_out)
        if self.use_attention:
            normed   = self.layer_norm1(lstm_out)
            attn, _  = self.attention(normed, normed, normed)
            last     = self.layer_norm2(attn + lstm_out)[:, -1, :]
        else:
            last = lstm_out[:, -1, :]
        return self.head(last)


# =============================================================================
# 5. 손실 함수
# =============================================================================

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    def forward(self, p, t):
        d = (p - t).abs()
        return torch.where(d <= self.delta,
                           0.5 * d ** 2,
                           self.delta * (d - 0.5 * self.delta)).mean()


class BiasAwareLoss(nn.Module):
    """MSE + 배치 편향 패널티 + 예측 범위 압축 패널티"""
    def __init__(self, alpha=0.5, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
    def forward(self, p, t):
        mse   = nn.functional.mse_loss(p, t)
        bias  = (p.mean() - t.mean()) ** 2
        p_std = p.std() + 1e-8;  t_std = t.std() + 1e-8
        rng   = torch.clamp(t_std / p_std - 1.0, min=0.0) ** 2
        return mse + self.alpha * bias + self.beta * rng


def build_criterion(cfg):
    fn = cfg.get("loss_function", "mse")
    if fn == "huber":
        return HuberLoss(delta=cfg.get("huber_delta", 1.0))
    if fn == "bias_aware":
        return BiasAwareLoss(alpha=cfg.get("bias_alpha", 0.5),
                             beta=cfg.get("bias_beta",  0.3))
    return nn.MSELoss()


# =============================================================================
# 6. 유틸리티
# =============================================================================

def report_and_fix(arr):
    if hasattr(arr, "values"):
        arr = arr.values
    t = torch.as_tensor(np.asarray(arr, dtype=np.float32))
    if torch.isnan(t).any() or torch.isinf(t).any():
        if t.ndim == 1:
            m = torch.nanmean(t)
            m = torch.zeros(()) if torch.isnan(m) else m
        else:
            m = torch.nanmean(t, dim=0)
            m = torch.where(torch.isnan(m), torch.zeros_like(m), m)
            m = m.unsqueeze(0).expand_as(t)
        t = torch.where(torch.isfinite(t), t, m)
    return t.numpy()


def split_timewise(X, y):
    T = len(X)
    n_tr = int(T * SPLIT_RATIOS["train"])
    n_va = int(T * SPLIT_RATIOS["val"])
    return (X[:n_tr], y[:n_tr],
            X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va],
            X[n_tr + n_va:], y[n_tr + n_va:])


# =============================================================================
# 7. 학습 루프 (단일 trial)
# =============================================================================

def train_one_trial(model, train_dl, val_dl, cfg, device):
    """
    단일 설정으로 학습 → best_val_rmse 반환.
    verbose 없이 epoch 단위 진행만 출력.
    """
    criterion = build_criterion(cfg)

    opt_name = cfg.get("optimizer", "adam")
    lr       = cfg["learning_rate"]
    wd       = cfg.get("weight_decay", 0.0)
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg.get("scheduler_factor", 0.5),
        patience=cfg.get("scheduler_patience", 5),
    )

    patience      = cfg.get("patience", 20)
    grad_clip     = cfg.get("gradient_clip", 1.0)
    best_val_rmse = float("inf")
    best_state    = None
    p_count       = 0
    prev_v_loss   = None
    streak        = 0
    MIN_E         = 50

    pbar = tqdm(range(NUM_EPOCHS), desc="Training", leave=False, ncols=90)
    for epoch in pbar:
        # -- Train --
        model.train()
        tr_loss = tr_mse = n_tr = 0.0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device).float(), yb.to(device).float()
            optimizer.zero_grad()
            p = model(Xb)
            if p.dim()  > 2: p  = p.squeeze(1)
            if yb.dim() > 2: yb = yb.squeeze(1)
            loss = criterion(p, yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            bs = yb.size(0); n_tr += bs
            tr_loss += loss.item() * bs
            with torch.no_grad():
                tr_mse += ((p - yb) ** 2).sum().item()

        avg_tr_loss = tr_loss / max(n_tr, 1)

        # -- Val --
        model.eval()
        va_loss = va_mse = n_va = 0.0
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(device).float(), yb.to(device).float()
                p = model(Xb)
                if p.dim()  > 2: p  = p.squeeze(1)
                if yb.dim() > 2: yb = yb.squeeze(1)
                loss = criterion(p, yb)
                bs = yb.size(0); n_va += bs
                va_loss += loss.item() * bs
                va_mse  += ((p - yb) ** 2).sum().item()

        v_loss = va_loss / max(n_va, 1)
        v_rmse = (va_mse  / max(n_va, 1)) ** 0.5
        scheduler.step(v_rmse)

        pbar.set_postfix({"val_rmse": f"{v_rmse:.4f}", "best": f"{best_val_rmse:.4f}"})

        # Early stopping (연속 증가)
        if epoch >= MIN_E:
            streak = 0 if (prev_v_loss is None or v_loss < prev_v_loss) else streak + 1
            if streak >= 5 or v_loss > avg_tr_loss * 2.0:
                break
        prev_v_loss = v_loss

        # Best / patience
        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            p_count       = 0
        else:
            p_count += 1
            if epoch >= MIN_E and p_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_rmse


def evaluate_trial(model, test_dl, device):
    """테스트 세트 평가 → dict(r2, rmse, mae, mse)"""
    model.eval()
    preds_list, y_list = [], []
    with torch.no_grad():
        for Xb, yb in test_dl:
            Xb, yb = Xb.to(device).float(), yb.to(device).float()
            p = model(Xb)
            if p.dim()  > 2: p  = p.squeeze(1)
            if yb.dim() > 2: yb = yb.squeeze(1)
            preds_list.append(p.cpu())
            y_list.append(yb.cpu())

    preds = torch.cat(preds_list)
    ys    = torch.cat(y_list)
    mse   = ((preds - ys) ** 2).mean().item()
    mae   = (preds - ys).abs().mean().item()
    rmse  = mse ** 0.5
    ss_res = ((ys - preds) ** 2).sum()
    ss_tot = ((ys - ys.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot.item() > 0 else float("nan")
    return {"r2": r2, "rmse": rmse, "mae": mae, "mse": mse}


# =============================================================================
# 8. 단일 Trial 실행
# =============================================================================

def run_trial(trial_id, cfg, X_all, y_all, feature_names, wf_selector):
    """단일 하이퍼파라미터 조합 실행 → 결과 dict 반환"""

    # 특성 선택
    rec_idx, _ = get_feature_subset(
        wf_selector, feature_names,
        stability_ratio=cfg["stability_ratio"],
        min_features=cfg["min_features"],
    )
    n_feat   = len(rec_idx)
    X_sel    = X_all[:, rec_idx]

    # y 배열
    y_np = y_all.values if hasattr(y_all, "values") else np.asarray(y_all)

    # 분할
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_timewise(X_sel, y_np)

    # NaN/Inf 수정
    X_tr = report_and_fix(X_tr); X_va = report_and_fix(X_va); X_te = report_and_fix(X_te)
    y_tr = report_and_fix(y_tr); y_va = report_and_fix(y_va); y_te = report_and_fix(y_te)

    # 스케일링
    SC  = RobustScaler if cfg.get("scaler_type") == "robust" else StandardScaler
    xsc = SC().fit(X_tr);  ysc = SC().fit(y_tr)
    X_tr_s = xsc.transform(X_tr); X_va_s = xsc.transform(X_va); X_te_s = xsc.transform(X_te)
    y_tr_s = ysc.transform(y_tr); y_va_s = ysc.transform(y_va); y_te_s = ysc.transform(y_te)

    def to2d(a):
        t = torch.as_tensor(np.asarray(a, dtype=np.float32))
        return t.unsqueeze(1) if t.ndim == 1 else t

    y_tr_t = to2d(y_tr_s); y_va_t = to2d(y_va_s); y_te_t = to2d(y_te_s)

    bs = cfg.get("batch_size", 2048)
    train_ds = TimeSeriesWindowDataset(X_tr_s, y_tr_t, WINDOW_SIZE, HORIZON)
    val_ds   = TimeSeriesWindowDataset(X_va_s, y_va_t, WINDOW_SIZE, HORIZON)
    test_ds  = TimeSeriesWindowDataset(X_te_s, y_te_t, WINDOW_SIZE, HORIZON)

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError(f"데이터 부족: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=bs,
                          shuffle=cfg.get("shuffle_train", True), drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, drop_last=False)

    # 모델 생성
    model = LSTMRegressor(
        n_features    = n_feat,
        hidden_size   = cfg["hidden_size"],
        num_layers    = cfg["num_layers"],
        dropout       = cfg["dropout"],
        out_size      = 1,
        use_attention = cfg["use_attention"],
        num_heads     = cfg.get("num_heads", 4),
        deep_head     = cfg["deep_head"],
        bidirectional = cfg.get("bidirectional", False),
    ).to(DEVICE)

    best_val_rmse = train_one_trial(model, train_dl, val_dl, cfg, DEVICE)
    metrics       = evaluate_trial(model, test_dl, DEVICE)

    return {
        "trial_id":            trial_id,
        "n_features_selected": n_feat,
        **cfg,
        "best_val_rmse":  best_val_rmse,
        "test_r2":        metrics["r2"],
        "test_rmse":      metrics["rmse"],
        "test_mae":        metrics["mae"],
        "test_mse":        metrics["mse"],
    }


# =============================================================================
# 9. 랜덤 탐색 엔진
# =============================================================================

def estimate_vram_mb(cfg: dict) -> float:
    """
    학습 시 대략적 VRAM 사용량 추정 (MiB).
    forward 메모리 × 4 (forward + backward + Adam 옵티마이저 상태)
    """
    hidden = cfg["hidden_size"]
    layers = cfg["num_layers"]
    batch  = cfg["batch_size"]
    seq    = WINDOW_SIZE

    # LSTM 출력 텐서 (모든 레이어, 모든 타임스텝 저장)
    lstm_bytes = layers * seq * batch * hidden * 4

    # Attention (Q, K, V + attention weight matrix)
    if cfg.get("use_attention", True):
        attn_bytes = (3 * batch * seq * hidden + batch * seq * seq) * 4
    else:
        attn_bytes = 0

    fwd_mb = (lstm_bytes + attn_bytes) / (1024 ** 2)
    return fwd_mb * 4   # forward + backward + optimizer states


def sample_config(rng: random.Random) -> dict:
    """메모리 예산 초과 설정은 최대 200회 재샘플링 후 반환."""
    for _ in range(200):
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        if estimate_vram_mb(cfg) <= GPU_MEMORY_BUDGET_MB:
            return cfg
    # fallback: 안전한 소형 설정
    return {k: v[0] for k, v in SEARCH_SPACE.items()}


def run_random_search(n_trials, X_all, y_all, feature_names, wf_selector, results_path):
    rng     = random.Random(RANDOM_SEED)
    results = []

    # 재개 지원: 기존 결과 로드
    start = 0
    if results_path.exists():
        existing = pd.read_csv(results_path)
        results  = existing.to_dict("records")
        start    = len(results)
        print(f"[OK] 기존 결과 {start}개 로드 → Trial {start}부터 재개")
    # rng 상태를 start 만큼 앞으로 돌림
    for _ in range(start):
        sample_config(rng)

    print(f"\n{'='*70}")
    print(f"랜덤 탐색: {n_trials}개 Trial  /  device={DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU 메모리 예산: {GPU_MEMORY_BUDGET_MB:.0f} MiB "
              f"(전체 {_total_vram_mb:.0f} MiB의 70%)")
    total_params = 1
    for v in SEARCH_SPACE.values():
        total_params *= len(v)
    print(f"탐색 공간 크기 (전체 조합): {total_params:,}")
    print(f"{'='*70}\n")

    for tid in range(start, n_trials):
        cfg     = sample_config(rng)
        t_start = datetime.now()
        print(f"\n[Trial {tid+1:3d}/{n_trials}]  "
              f"hs={cfg['hidden_size']}  nl={cfg['num_layers']}  "
              f"attn={cfg['use_attention']}  "
              f"lr={cfg['learning_rate']}  bs={cfg['batch_size']}  "
              f"sr={cfg['stability_ratio']}  minf={cfg['min_features']}")

        try:
            result          = run_trial(tid, cfg, X_all, y_all, feature_names, wf_selector)
            elapsed         = (datetime.now() - t_start).total_seconds()
            result["elapsed_sec"] = elapsed
            result["status"]      = "ok"
            print(f"  → R²={result['test_r2']:.4f}  RMSE={result['test_rmse']:.4f}  "
                  f"MAE={result['test_mae']:.4f}  feats={result['n_features_selected']}  "
                  f"({elapsed:.0f}s)")
        except Exception as e:
            elapsed = (datetime.now() - t_start).total_seconds()
            print(f"  [FAIL] 실패: {e}")
            result = {"trial_id": tid, "status": "error", "error_msg": str(e),
                      "elapsed_sec": elapsed, **cfg}

        results.append(result)
        pd.DataFrame(results).to_csv(results_path, index=False)

    return pd.DataFrame(results)


# =============================================================================
# 10. 메인
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TOC 하이퍼파라미터 랜덤 탐색 실험")
    parser.add_argument("--n_trials",         type=int, default=50,
                        help="랜덤 탐색 횟수 (기본: 50)")
    parser.add_argument("--force_preprocess", action="store_true",
                        help="전처리 캐시 무시 후 재실행")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    warnings.filterwarnings("ignore")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"N Trials: {args.n_trials}")
    print(f"탐색 파라미터 수: {len(SEARCH_SPACE)}")

    # -- 전처리 캐시 ----------------------------------------------------------
    cache_path = EXPERIMENT_DIR / "preprocess_cache.pkl"
    if cache_path.exists() and not args.force_preprocess:
        print(f"\n[OK] 전처리 캐시 로드: {cache_path}")
        with open(cache_path, "rb") as f:
            X_all, y_all, feature_names, wf_selector = pickle.load(f)
        print(f"  X_all: {X_all.shape},  특성 수: {len(feature_names)}")
    else:
        X_all, y_all, feature_names, wf_selector = preprocess_data_full()
        with open(cache_path, "wb") as f:
            pickle.dump((X_all, y_all, feature_names, wf_selector), f)
        print(f"\n[OK] 전처리 캐시 저장: {cache_path}")

    # -- 랜덤 탐색 ------------------------------------------------------------
    results_path = EXPERIMENT_DIR / "toc_experiment_results.csv"
    df = run_random_search(
        n_trials      = args.n_trials,
        X_all         = X_all,
        y_all         = y_all,
        feature_names = feature_names,
        wf_selector   = wf_selector,
        results_path  = results_path,
    )

    # -- 결과 요약 ------------------------------------------------------------
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("\n성공한 trial이 없습니다.")
        return

    best = ok.loc[ok["test_r2"].idxmax()]

    print(f"\n{'='*70}")
    print("탐색 완료 — 최고 성능 설정 (R² 기준)")
    print(f"{'='*70}")
    print(f"  Trial ID           : {best['trial_id']}")
    print(f"  test R²            : {best['test_r2']:.4f}")
    print(f"  test RMSE          : {best['test_rmse']:.4f}")
    print(f"  test MAE           : {best['test_mae']:.4f}")
    print(f"  n_features         : {best.get('n_features_selected','?')}")
    print(f"  best_val_rmse      : {best.get('best_val_rmse','?'):.4f}")
    print("\n  하이퍼파라미터:")
    for k in SEARCH_SPACE:
        if k in best:
            print(f"    {k:25s}: {best[k]}")

    # 최고 설정 JSON 저장
    best_cfg = {}
    for k, v in best.items():
        if isinstance(v, (np.floating, float)):  best_cfg[k] = float(v)
        elif isinstance(v, (np.integer, int)):   best_cfg[k] = int(v)
        elif isinstance(v, bool):                best_cfg[k] = bool(v)
        else:                                    best_cfg[k] = v

    best_path = EXPERIMENT_DIR / "toc_experiment_best.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, ensure_ascii=False, indent=2)

    # Top-10 표
    print(f"\n-- Top-10 (R² 기준) --")
    cols = ["trial_id", "test_r2", "test_rmse", "test_mae",
            "n_features_selected", "hidden_size", "num_layers", "dropout",
            "use_attention", "num_heads", "bidirectional", "deep_head",
            "learning_rate", "batch_size", "optimizer", "loss_function",
            "scaler_type", "stability_ratio", "min_features"]
    top10 = ok.sort_values("test_r2", ascending=False).head(10)
    print(top10[[c for c in cols if c in top10.columns]].to_string(index=False))

    print(f"\n[OK] 전체 결과 저장: {results_path}")
    print(f"[OK] 최고 설정 저장: {best_path}")


if __name__ == "__main__":
    main()
