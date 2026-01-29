"""
데이터 분할 모듈
시간 기반 train/valid/test 분할
"""

import math
from dataclasses import dataclass


@dataclass
class SplitConfig:
    """데이터 분할 비율 설정"""
    train_ratio: float = 0.6
    valid_ratio: float = 0.2
    test_ratio: float = 0.2


def time_split(X, y, cfg=SplitConfig()):
    """시간 순서 기반 데이터 분할"""
    n = len(X)
    if n == 0:
        raise ValueError("전처리/피처 생성 후 데이터셋이 비어있습니다.")
    if not math.isclose(cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train/valid/test 비율의 합이 1.0이어야 합니다.")

    n_train = int(n * cfg.train_ratio)
    n_valid = int(n * cfg.valid_ratio)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_valid, y_valid = X.iloc[n_train:n_train + n_valid], y.iloc[n_train:n_train + n_valid]
    X_test, y_test = X.iloc[n_train + n_valid:], y.iloc[n_train + n_valid:]

    return {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test),
    }
