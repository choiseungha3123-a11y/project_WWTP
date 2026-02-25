from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

# 업체 예측 컬럼 매핑 (LSTM target → 업체 CSV 컬럼명)
_VENDOR_TMS_COL = {
    "toc": "TOC_VU",
    "ph": "PH_VU",
    "ss": "SS_VU",
    "tn": "TN_VU",
    "tp": "TP_VU",
    "flux": "FLUX_VU",
}
_VENDOR_FLOW_COL = "Q_in"


def get_demo_root() -> Path:
    return ROOT_DIR


def load_predictions(target: str) -> pd.DataFrame:
    path = ROOT_DIR / "data" / "output" / "save" / f"{target}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"예측 CSV를 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)
    required = {"actual", "predicted"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 컬럼이 올바르지 않습니다: {path}")
    return df


def load_feature_names(target: str) -> list[str]:
    path = ROOT_DIR / "data" / "features" / "save" / f"{target}_recommended_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"추천 피처 CSV를 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)
    if "feature_name" not in df.columns:
        raise ValueError(f"feature_name 컬럼이 없습니다: {path}")
    return df["feature_name"].dropna().astype(str).tolist()


def get_png_path(kind: str, target: str) -> Path:
    dl_dir = ROOT_DIR / "results" / "DL"
    save_dir = dl_dir / "save"

    if kind == "learning_curve":
        return save_dir / f"{target}_learning_curve.png"

    if kind == "prediction":
        return save_dir / f"prediction_analysis_{target}.png"

    if kind == "diagnosis":
        return save_dir / f"{target}_diagnosis.png"

    raise ValueError(f"지원하지 않는 kind입니다: {kind}")


def load_experiment_results(target: str) -> pd.DataFrame:
    path = ROOT_DIR / "results" / "DL" / f"{target}_experiment_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"실험 결과 CSV를 찾을 수 없습니다: {path}")
    return pd.read_csv(path)


def load_comparison(target: str) -> pd.DataFrame:
    """LSTM 예측값과 업체 예측값을 1시간 단위로 리샘플링하여 정렬해 반환.

    Returns
    -------
    DataFrame with DatetimeIndex (1H) and columns:
        actual      – 실측값 (LSTM 테스트 세트 기준)
        lstm_pred   – LSTM 예측값
        vendor_pred – 업체 모델 예측값

    Notes
    -----
    - LSTM 예측(30분 해상도)과 업체 예측(1분 해상도)을 1시간 단위로 리샘플링해 비교.
    - FLUX: LSTM은 30분 증분값을 합산(sum), 업체는 누적값에 diff 적용.
    - 그 외: 평균(mean)으로 리샘플링.
    """
    # ── LSTM 예측 로드 (time 컬럼 직접 사용) ─────────────────────────────
    lstm_df = load_predictions(target)
    lstm_df["time"] = pd.to_datetime(lstm_df["time"])
    lstm_df = lstm_df.set_index("time").sort_index()
    lstm_df.index.name = "SYS_TIME"
    lstm_df = lstm_df.rename(columns={"predicted": "lstm_pred"})

    t_start = lstm_df.index[0]
    t_end = lstm_df.index[-1]

    # ── 업체 예측 로드 ────────────────────────────────────────────────────
    if target == "flow":
        vendor_path = ROOT_DIR / "data" / "pred" / "FLOW_Pred.csv"
        vendor_raw = pd.read_csv(vendor_path, encoding="utf-8-sig")
        vendor_col = _VENDOR_FLOW_COL
    else:
        vendor_path = ROOT_DIR / "data" / "pred" / "TMS_Pred.csv"
        vendor_raw = pd.read_csv(vendor_path, encoding="utf-8-sig")
        vendor_col = _VENDOR_TMS_COL[target]

    vendor_raw["SYS_TIME"] = pd.to_datetime(vendor_raw["SYS_TIME"], errors="coerce")
    vendor_raw = vendor_raw.set_index("SYS_TIME").sort_index()
    vendor_series = vendor_raw[vendor_col].loc[t_start:t_end]

    # ── 1시간 단위 리샘플링 ───────────────────────────────────────────────
    if target == "flux":
        # LSTM: 30분 증분값 합산 → 시간당 총 증분
        lstm_1h = lstm_df.resample("1h").agg({"actual": "sum", "lstm_pred": "sum"})
        # 업체: 누적값의 마지막 포인트 기준 diff → 시간당 증분
        vendor_1h = (
            vendor_series
            .resample("1h")
            .last()
            .diff()
            .clip(lower=0)
        )
        vendor_1h.iloc[0] = float("nan")
    else:
        # LSTM: 평균
        lstm_1h = lstm_df.resample("1h").mean()
        # 업체: 평균
        vendor_1h = vendor_series.resample("1h").mean()

    # ── 병합 ──────────────────────────────────────────────────────────────
    result = lstm_1h.join(vendor_1h.rename("vendor_pred"), how="left")
    # actual·lstm_pred NaN 제거 (리샘플링 빈 bin 방어)
    result = result.dropna(subset=["actual", "lstm_pred"])
    return result[["actual", "lstm_pred", "vendor_pred"]]
