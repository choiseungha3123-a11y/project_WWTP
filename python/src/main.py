from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import time
import uuid
import numpy as np
import torch
import pandas as pd

app = FastAPI(title="Model Serving API", version="0.1.0")

modelA = joblib.load(r"c:\project_WWTP\python\model\modelA_lstm_model.pth", map_location="cpu")
modelB = joblib.load(r"c:\project_WWTP\python\model\modelB_lstm_model.pth", map_location="cpu")
modelC = joblib.load(r"c:\project_WWTP\python\model\modelC_lstm_model.pth", map_location="cpu")
modelA.eval(); modelB.eval(); modelC.eval()

origins = [
    "http://www.projectwwtp.kro.kr:8081",
    "http://localhost:8081",  # 로컬 테스트용
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # 허용할 도메인 목록
    allow_credentials=True,
    allow_methods=["*"],               # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],               # 모든 HTTP 헤더 허용
)

# feature 순서: 모델 학습 시 사용한 순서로 맞춰야 함
FEATURE_COLUMNS = ["TOC_VU", "PH_VU", "SS_VU", "FLUX_VU", "TN_VU", "TP_VU"]

# 각 모델이 입력으로 받는 피쳐 (학습 때 사용한 입력 피쳐를 여기에 맞춰야 함)
MODEL_INPUTS = {
    "A": ["TOC_VU", "SS_VU"],       # modelA 입력 (예: TOC, SS 기반 예측)
    "B": ["TN_VU", "TP_VU"],       # modelB 입력 (예: TN, TP)
    "C": ["FLUX_VU", "PH_VU"],     # modelC 입력 (예: FLUX, PH)
}

# 각 모델이 예측하는 타깃 순서 (모델 출력의 각 채널이 어떤 지표인지)
MODEL_TARGETS = {
    "A": ["TOC_VU", "SS_VU"],
    "B": ["TN_VU", "TP_VU"],
    "C": ["FLUX_VU", "PH_VU"],
}

def preprocess_sequence(data_list: list[dict]) -> torch.Tensor:
    """
    원래 전체 FEATURES로 시퀀스 텐서를 만듦 (보존).
    """
    if not data_list:
        raise ValueError("dataList is empty")

    df = pd.DataFrame(data_list)
    if "SYS_TIME" in df.columns:
        df["SYS_TIME"] = pd.to_datetime(df["SYS_TIME"], errors="coerce")
        df = df.sort_values("SYS_TIME")
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    arr = df[FEATURE_COLUMNS].astype(float).fillna(method="ffill").fillna(0.0).values
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    return tensor

def build_tensor_for_features(data_list: list[dict], features: list[str]) -> torch.Tensor:
    """
    지정된 features만 추출해 (1, seq_len, n_features) 텐서 반환.
    """
    if not data_list:
        raise ValueError("dataList is empty")
    df = pd.DataFrame(data_list)
    if "SYS_TIME" in df.columns:
        df["SYS_TIME"] = pd.to_datetime(df["SYS_TIME"], errors="coerce")
        df = df.sort_values("SYS_TIME")
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    arr = df[features].astype(float).fillna(method="ffill").fillna(0.0).values
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

def extract_scalar_from_output(y: object):
    """
    모델 출력(y)이 tensor/ndarray/list 등일 때 '1시간 뒤' 단일 스칼라(또는 채널별 값)로 추출.
    - 텐서/ndarray인 경우 마지막 시점의 값 또는 마지막 채널을 취함(모델 구조에 따라 조정 필요).
    - 반환은 numpy.ndarray 또는 float 형태 (채널이 여러개면 1D array).
    """
    if isinstance(y, torch.Tensor):
        arr = y.cpu().numpy()
    elif isinstance(y, np.ndarray):
        arr = y
    else:
        # dict/string 등 비표준 반환이면 그대로 리턴
        return y

    # arr shape possibilities:
    # (1, horizon, channels) -> take last timestep arr[0, -1, :]
    # (1, channels) -> arr[0, :]
    # (channels,) -> arr
    try:
        if arr.ndim == 3 and arr.shape[0] == 1:
            out = arr[0, -1, :]
        elif arr.ndim == 2 and arr.shape[0] == 1:
            out = arr[0, :]
        elif arr.ndim == 1:
            out = arr
        else:
            out = arr.reshape(-1)
    except Exception:
        out = arr.reshape(-1)
    # if single value, return scalar
    if out.size == 1:
        return float(out.ravel()[0])
    return out.tolist()

class PredictIn(BaseModel):
    request_id: str | None = None
    input: dict

class PredictOut(BaseModel):
    request_id: str
    ok: bool
    output: dict | None = None
    latency_ms: int
    error: dict | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/ready")
def ready():
    # 모델 로딩이 있으면 여기서 상태 체크로 바꾸면 됨
    return {"ok": True, "model_loaded": True, "model_version": "0.1.0"}

@app.post("/predict", response_model=PredictOut)
def predict(x: PredictIn):
    try:
        t0 = time.perf_counter()
        rid = x.request_id or str(uuid.uuid4())

        data_list = x.input.get("dataList") if isinstance(x.input, dict) else None
        if data_list is None:
            raise ValueError("input.dataList is required")

        # 각 모델별 입력 텐서 생성 -> 예측 -> 타깃별 값 추출
        models = {"A": modelA, "B": modelB, "C": modelC}
        per_model = {}
        per_target_values = {}  # 예: {"TOC_VU": 1.23, ...}

        with torch.no_grad():
            for name, m in models.items():
                input_features = MODEL_INPUTS.get(name, FEATURE_COLUMNS)
                tensor = build_tensor_for_features(data_list, input_features)  # (1, seq_len, n_in_features)
                y = m(tensor)
                extracted = extract_scalar_from_output(y)
                per_model[name] = extracted
                # map extracted channels -> target names
                targets = MODEL_TARGETS.get(name, [])
                if isinstance(extracted, list):
                    for tgt_name, val in zip(targets, extracted):
                        per_target_values[tgt_name] = val
                else:
                    # 단일 스칼라 출력인 경우 대상이 하나라면 할당, 여러개면 첫번째에 할당
                    if len(targets) == 1:
                        per_target_values[targets[0]] = extracted
                    else:
                        # 여러 타깃인데 스칼라만 반환되면 첫 타깃에 넣고 나머지는 None
                        per_target_values[targets[0]] = extracted
                        for tname in targets[1:]:
                            per_target_values.setdefault(tname, None)

        # 최종 combined row: FEATURE_COLUMNS 순서로 모델들이 예측한 값을 한 행으로 만듦
        combined_row = [per_target_values.get(col, None) for col in FEATURE_COLUMNS]

        latency = int((time.perf_counter() - t0) * 1000)
        return PredictOut(
            request_id=rid,
            ok=True,
            output={
                "per_model": per_model,
                "per_target": per_target_values,
                "combined_row": combined_row,   # 한 열(타임스탬프 기준) 형태: [TOC, PH, SS, FLUX, TN, TP]
            },
            latency_ms=latency,
            error=None
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
