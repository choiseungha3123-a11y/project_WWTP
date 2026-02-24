from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(actual, predicted) -> dict[str, float]:
    actual_np = np.asarray(actual, dtype=float)
    predicted_np = np.asarray(predicted, dtype=float)

    # MAPE: 실제값이 0인 샘플은 제외
    nonzero = actual_np != 0
    mape = float(np.mean(np.abs((actual_np[nonzero] - predicted_np[nonzero]) / actual_np[nonzero])) * 100) if nonzero.any() else float("nan")

    # README.md 정의: |실제 - 예측| / |실제| ≤ 5% 비율
    within_5pct = float(np.mean(np.abs((actual_np[nonzero] - predicted_np[nonzero]) / actual_np[nonzero]) <= 0.05) * 100) if nonzero.any() else float("nan")

    return {
        "r2": float(r2_score(actual_np, predicted_np)),
        "rmse": float(np.sqrt(mean_squared_error(actual_np, predicted_np))),
        "mae": float(mean_absolute_error(actual_np, predicted_np)),
        "mape": mape,
        "within_5pct": within_5pct,
    }


def compute_operational_metrics(
    actual,
    predicted,
    thr,
    kind: str = "upper",
) -> dict:
    """기준초과 이벤트 탐지 성능 지표를 계산합니다.

    Parameters
    ----------
    actual, predicted : array-like
        실측값 / 예측값 시퀀스
    thr : float | tuple | None
        - float  : 상한 초과 기준 (kind="upper")
        - tuple  : (하한, 상한) 범위 이탈 기준 (kind="range")
        - None   : 실데이터 95th percentile 자동 계산 (kind="percentile")
    kind : str
        "upper" | "range" | "percentile"

    Returns
    -------
    dict with keys:
        recall, precision, f1, fpr,
        event_count, threshold_used, kind,
        tp, fp, fn, tn
    """
    actual_np    = np.asarray(actual, dtype=float)
    predicted_np = np.asarray(predicted, dtype=float)

    threshold_used = thr

    if kind == "percentile":
        threshold_used = float(np.percentile(actual_np, 95))
        actual_event   = actual_np    > threshold_used
        pred_event     = predicted_np > threshold_used
        kind = "upper"

    elif kind == "upper":
        actual_event   = actual_np    > thr
        pred_event     = predicted_np > thr

    elif kind == "range":
        low, high      = thr
        actual_event   = (actual_np    < low) | (actual_np    > high)
        pred_event     = (predicted_np < low) | (predicted_np > high)

    else:
        raise ValueError(f"지원하지 않는 kind: {kind}")

    tp = int(np.sum( pred_event &  actual_event))
    fp = int(np.sum( pred_event & ~actual_event))
    fn = int(np.sum(~pred_event &  actual_event))
    tn = int(np.sum(~pred_event & ~actual_event))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (not np.isnan(precision)) and (not np.isnan(recall)) and (precision + recall) > 0
        else float("nan")
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    return {
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "fpr":            fpr,
        "event_count":    int(np.sum(actual_event)),
        "threshold_used": threshold_used,
        "kind":           kind,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
