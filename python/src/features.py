"""
피처 엔지니어링 모듈
시간 피처, 지연 피처, 롤링 통계 피처 생성
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


def add_time_features(df, add_sin_cos=True):
    """시간 관련 피처 추가 (시간, 요일, 월, 주말, 시간대, 계절)"""
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("시간 피처 생성을 위해서는 DatetimeIndex가 필요합니다.")

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek  # 월요일=0
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # 시간대 구간 (필요시 커스터마이징 가능)
    # 0: 밤(0-5), 1: 아침(6-11), 2: 오후(12-17), 3: 저녁(18-23)
    out["tod_bucket"] = pd.cut(
        out["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # 계절
    m = out["month"]
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    out["season"] = m.map(season_map).astype(int)

    if add_sin_cos:
        # 시간 주기 (24시간)
        out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)

        # 요일 주기 (7일)
        out["sin_dow"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
        out["cos_dow"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)

        # 월 주기 (12개월)
        out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)

    return out


def add_lag_features(df, base_cols, lags):
    """지연(lag) 피처 추가"""
    out = df.copy()
    for c in base_cols:
        if c not in out.columns:
            continue
        for k in lags:
            out[f"{c}_lag{k}"] = out[c].shift(k)
    return out


def add_rolling_features(df, base_cols, windows, stats=["mean"]):
    """
    롤링 통계 피처 추가
    stats: ["mean", "std", "min", "max"] - 통계량 종류
    """
    out = df.copy()
    for c in base_cols:
        if c not in out.columns:
            continue
        for w in windows:
            r = out[c].rolling(window=w, min_periods=w)
            if "mean" in stats:
                out[f"{c}_r{w}_mean"] = r.mean()
            if "std" in stats:
                out[f"{c}_r{w}_std"] = r.std()
            if "min" in stats:
                out[f"{c}_r{w}_min"] = r.min()
            if "max" in stats:
                out[f"{c}_r{w}_max"] = r.max()
    return out


def make_supervised_dataset(df, target_cols, dropna=True):
    """지도학습용 X, y 데이터셋 생성"""
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"타겟 컬럼을 찾을 수 없습니다: {missing}")

    y = df[target_cols].copy()
    X = df.drop(columns=target_cols).copy()

    X = X.select_dtypes(include=[np.number])

    keep = X.notna().all(axis=1) & y.notna().all(axis=1)
    return X.loc[keep], y.loc[keep]


@dataclass
class FeatureConfig:
    """피처 생성 설정"""
    add_time: bool = True
    add_sin_cos: bool = True
    lag_hours: list = None
    roll_hours: list = None

    def __post_init__(self):
        if self.lag_hours is None:
            self.lag_hours = [1, 2, 3, 6, 12, 24]
        if self.roll_hours is None:
            self.roll_hours = [1, 2, 24]


def build_features(df_hourly, target_cols, feature_base_cols=None, cfg=FeatureConfig()):
    """전체 피처 생성 파이프라인"""
    out = df_hourly.copy()

    if cfg.add_time:
        out = add_time_features(out, add_sin_cos=cfg.add_sin_cos)

    if feature_base_cols is None:
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        feature_base_cols = [c for c in numeric_cols if c not in target_cols]

    out = add_lag_features(out, base_cols=feature_base_cols, lags=cfg.lag_hours)
    out = add_rolling_features(out, base_cols=feature_base_cols, windows=cfg.roll_hours)
    return out
