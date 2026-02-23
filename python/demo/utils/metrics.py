from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(actual, predicted) -> dict[str, float]:
    actual_np = np.asarray(actual)
    predicted_np = np.asarray(predicted)
    return {
        "r2": float(r2_score(actual_np, predicted_np)),
        "rmse": float(np.sqrt(mean_squared_error(actual_np, predicted_np))),
        "mae": float(mean_absolute_error(actual_np, predicted_np)),
    }
