from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]


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
    path = ROOT_DIR / "data" / "recommand_features" / "save" / f"{target}_recommended_features.csv"
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
