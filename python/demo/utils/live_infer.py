from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import FLOW_CONFIG, SAVE_DIR, TMS_TARGETS
from src.models import LSTMRegressor, StandardScaler
from src.predict import autoregressive_predict, find_target_lag_idx

# pickle 호환성: 학습 당시 __main__.StandardScaler 로 저장된 객체 복원용
sys.modules["__main__"].StandardScaler = StandardScaler


class _CompatUnpickler(pickle.Unpickler):
    """Map legacy pickled class paths to current runtime classes."""

    def find_class(self, module, name):
        if module == "__main__" and name == "StandardScaler":
            return StandardScaler
        return super().find_class(module, name)


def _load_pickle_compat(path: Path):
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def _target_config(target: str) -> dict:
    if target == "flow":
        return {
            "hidden_size": FLOW_CONFIG["hidden_size"],
            "num_layers": FLOW_CONFIG["num_layers"],
            "dropout": FLOW_CONFIG["dropout"],
            "use_attention": FLOW_CONFIG.get("use_attention", False),
            "deep_head": FLOW_CONFIG.get("deep_head", False),
            "target_col": None,
        }
    cfg = TMS_TARGETS[target]
    return {
        "hidden_size": cfg["hidden_size"],
        "num_layers": cfg["num_layers"],
        "dropout": cfg["dropout"],
        "use_attention": cfg.get("use_attention", False),
        "deep_head": cfg.get("deep_head", False),
        "target_col": cfg.get("target_col"),
    }


@st.cache_resource(show_spinner=False)
def load_runtime_artifacts(target: str) -> dict:
    feature_path = ROOT_DIR / "data" / "recommand_features" / "save" / f"{target}_recommended_features.csv"
    feat_df = pd.read_csv(feature_path)
    feature_names = feat_df["feature_name"].dropna().astype(str).tolist()

    cfg = _target_config(target)
    model = LSTMRegressor(
        n_features=len(feature_names),
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        out_size=1,
        use_attention=cfg["use_attention"],
        deep_head=cfg["deep_head"],
    )

    ckpt_path = SAVE_DIR / f"{target}_lstm_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    # src.predict.autoregressive_predict 내부에서 src.config.device 를 사용하므로
    # 모델도 동일 디바이스로 올려야 입력/파라미터 디바이스 불일치가 발생하지 않는다.
    from src.config import device as runtime_device
    model = model.to(runtime_device)
    model.eval()

    x_scaler = _load_pickle_compat(SAVE_DIR / f"X_scaler_{target}.pkl")
    y_scaler = _load_pickle_compat(SAVE_DIR / f"y_scaler_{target}.pkl")

    target_lag_idx = None
    if cfg["target_col"]:
        target_lag_idx = find_target_lag_idx(feature_names, cfg["target_col"])

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_names": feature_names,
        "target_lag_idx": target_lag_idx,
    }


def build_sequence_from_single_row(feature_names: list[str], values: dict[str, float], n_steps: int = 48) -> pd.DataFrame:
    row = {f: float(values.get(f, 0.0)) for f in feature_names}
    return pd.DataFrame([row] * n_steps)


def validate_and_align_input(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = df.copy()

    if len(out) != 48:
        raise ValueError(f"입력 행 수는 48이어야 합니다. 현재: {len(out)}")

    for col in feature_names:
        if col not in out.columns:
            out[col] = 0.0

    out = out[feature_names]
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


def run_inference(df_48xN: pd.DataFrame, artifacts: dict, horizon_steps: int = 24) -> tuple[float, list[float]]:
    x = artifacts["x_scaler"].transform(df_48xN.values)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        from src.config import device as runtime_device
        pred_scaled = artifacts["model"](x_tensor.to(runtime_device))
    one_step = float(artifacts["y_scaler"].inverse_transform(pred_scaled.cpu().numpy())[0, 0])

    multi = autoregressive_predict(
        input_tensor=x_tensor,
        n_steps=horizon_steps,
        pred_model=artifacts["model"],
        y_scaler_obj=artifacts["y_scaler"],
        target_feat_idx=artifacts.get("target_lag_idx"),
    )
    return one_step, multi


def build_trajectory_df(preds: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": np.arange(1, len(preds) + 1),
            "hour": np.arange(1, len(preds) + 1) * 0.5,
            "prediction": preds,
        }
    )
