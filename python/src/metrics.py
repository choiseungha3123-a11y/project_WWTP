"""
평가 지표 모듈
모델 성능 평가 및 시각화
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):
    """예측 결과에 대한 평가 지표 계산"""
    # 2D 형태로 변환
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    r2s, rmses, mapes = [], [], []
    for j in range(yt.shape[1]):
        r2s.append(r2_score(yt[:, j], yp[:, j]))
        rmses.append(math.sqrt(mean_squared_error(yt[:, j], yp[:, j])))
        mapes.append(mean_absolute_percentage_error(yt[:, j], yp[:, j]) * 100.0)

    return {
        "R2_mean": float(np.mean(r2s)),
        "RMSE_mean": float(np.mean(rmses)),
        "MAPE_mean(%)": float(np.mean(mapes)),
        "R2_by_target": r2s,
        "RMSE_by_target": rmses,
        "MAPE_by_target(%)": mapes,
    }


def fit_and_evaluate(model, splits):
    """모델 학습 및 평가"""
    X_train, y_train = splits["train"]

    model.fit(X_train, y_train)

    out = {}
    for name, (X_, y_) in splits.items():
        pred = model.predict(X_)
        out[name] = compute_metrics(y_.to_numpy(), pred)
    return out


def plot_predictions(y_true, y_pred, title, n_points=500):
    """예측 결과 시각화"""
    yt = y_true.copy()
    yp = np.asarray(y_pred)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    # 마지막 n_points 정렬
    yt = yt.iloc[-n_points:]
    yp = yp[-len(yt):, :]

    for j, col in enumerate(yt.columns):
        plt.figure(figsize=(12, 4))
        plt.plot(yt.index, yt[col].values, label="true")
        plt.plot(yt.index, yp[:, j], label="pred")
        plt.title(f"{title} | target={col}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_metric_table(result_by_model, split="test"):
    """모델별 평가 지표 테이블 생성"""
    rows = []
    for model_name, res in result_by_model.items():
        m = res[split]
        rows.append([model_name, m["R2_mean"], m["RMSE_mean"], m["MAPE_mean(%)"]])
    tbl = pd.DataFrame(rows, columns=["model", "R2_mean", "RMSE_mean", "MAPE_mean(%)"])
    return tbl.sort_values(by="RMSE_mean", ascending=True)
