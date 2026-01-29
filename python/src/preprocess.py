"""
데이터 전처리 모듈
결측치 처리, 리샘플링, 연속성 확인
"""

import pandas as pd


def drop_missing_rows(df, cols=None):
    """결측치가 있는 행 제거"""
    out = df.copy()
    if cols is None:
        return out.dropna()
    return out.dropna(subset=cols)


def resample_hourly(df, rule="1h", agg="mean"):
    """시계열 데이터 리샘플링"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("리샘플링을 위해서는 DatetimeIndex가 필요합니다.")
    out = df.copy()
    if isinstance(agg, str):
        return out.resample(rule).agg(agg)
    return out.resample(rule).agg(agg)


def check_continuity(df, freq="1h"):
    """시계열 데이터의 연속성 확인"""
    if len(df) == 0:
        return {"is_continuous": True, "n_missing_timestamps": 0, "max_gap": pd.Timedelta(0)}

    idx = df.index
    expected = pd.date_range(start=idx.min(), end=idx.max(), freq=freq, tz=idx.tz)
    missing = expected.difference(idx)
    # diff 기반 최대 간격 계산
    diffs = idx.to_series().diff().dropna()
    max_gap = diffs.max() if len(diffs) else pd.Timedelta(0)
    return {
        "is_continuous": len(missing) == 0,
        "n_missing_timestamps": len(missing),
        "max_gap": max_gap
    }
