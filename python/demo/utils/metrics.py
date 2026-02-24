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
