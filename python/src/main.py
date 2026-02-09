from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
import pickle
import time
import uuid
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path


# 프로젝트 루트를 sys.path에 추가
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Feature engineering 모듈 import
from notebook.feature import feature_engineering as feat_eng

app = FastAPI(title="Flow Prediction API", version="0.2.0")

# ====== 경로 설정 ======
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "recommand_features"

# ====== LSTM 모델 정의 (LSTM.ipynb와 동일해야 함) ======
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2, out_size=1):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size
        self.layer_norm1 = nn.LayerNorm(lstm_out_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(lstm_out_size)

        self.head = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size * 2),
            nn.LayerNorm(lstm_out_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size * 2, lstm_out_size),
            nn.LayerNorm(lstm_out_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, out_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out_normed = self.layer_norm1(lstm_out)
        attn_out, _ = self.attention(lstm_out_normed, lstm_out_normed, lstm_out_normed)
        attn_out = attn_out + lstm_out
        attn_out = self.layer_norm2(attn_out)
        last = attn_out[:, -1, :]
        yhat = self.head(last)
        return yhat

# ====== StandardScaler 정의 (LSTM.ipynb와 동일) ======
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x):
        return x * self.std_ + self.mean_

# ====== 모델 및 스케일러 로드 ======
print("Loading model and scalers...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 추천 특성 목록 로드
recommended_features_df = pd.read_csv(FEATURE_DIR / "flow_recommended_features.csv")
FEATURE_NAMES = recommended_features_df["feature_name"].tolist()
N_FEATURES = len(FEATURE_NAMES)

print(f"Number of features: {N_FEATURES}")

# 모델 로드
model = LSTMRegressor(
    n_features=N_FEATURES,
    hidden_size=128,
    num_layers=4,
    dropout=0.3,
    out_size=1
).to(device)

checkpoint = torch.load(MODEL_DIR / "flow_lstm_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 스케일러 로드 (pickle 호환성을 위해 현재 모듈의 StandardScaler를 등록)
sys.modules['__main__'].StandardScaler = StandardScaler

with open(MODEL_DIR / "X_scaler_flow.pkl", "rb") as f:
    x_scaler = pickle.load(f)
with open(MODEL_DIR / "y_scaler_flow.pkl", "rb") as f:
    y_scaler = pickle.load(f)

print("Model and scalers loaded successfully")

# ====== CORS 설정 ======
origins = [
    "http://www.projectwwtp.kro.kr:8081",
    "http://localhost:8081",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 데이터 전처리 함수 ======
def resample_to_30min(df: pd.DataFrame) -> pd.DataFrame:
    """1분 단위 데이터를 30분 단위로 리샘플링"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg_dict = {}
    for col in numeric_cols:
        if col.startswith("RN_") or col.startswith("AR_"):
            agg_dict[col] = "sum"  # 강수량은 누적
        else:
            agg_dict[col] = "mean"

    return df[numeric_cols].resample("30min").agg(agg_dict)

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    LSTM.ipynb와 동일한 전체 특성 엔지니어링 적용

    Parameters
    ----------
    df : pd.DataFrame
        30분 리샘플링된 데이터 (DatetimeIndex 필수)

    Returns
    -------
    pd.DataFrame
        특성 엔지니어링이 적용된 DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for feature engineering")

    # flow 모드 타겟 제거 (Q_in은 예측 대상이므로 특성에서 제외)
    target_cols = ["Q_in"]
    df_base = df.drop(columns=[c for c in target_cols if c in df.columns], errors='ignore')

    # 전체 특성 엔지니어링 적용 (LSTM.ipynb와 동일)
    # mode="flow"로 설정하여 데이터 누수 방지
    df_features = feat_eng.add_rain_features(df_base)
    df_features = feat_eng.add_station_agg_rain_features(df_features)
    df_features = feat_eng.add_weather_features(df_features)
    df_features = feat_eng.add_process_features(df_features)
    df_features = feat_eng.add_temporal_features(df_features)
    # df_features = feat_eng.add_interaction_features(df_features)  # LSTM.ipynb에서 주석 처리됨
    df_features = feat_eng.add_time_features(df_features)

    # NaN 처리: ffill -> 0
    df_features = df_features.ffill().fillna(0)

    return df_features

def preprocess_input(data_list, aws368, aws541, aws569) -> torch.Tensor:
    """
    백엔드 입력 데이터를 모델 입력 형식으로 변환

    Args:
        data_list: 1분 단위 24시간 데이터 (1440개 레코드)
        aws368: 368번 AWS 데이터
        aws541: 541번 AWS 데이터
        aws569: 569번 AWS 데이터

    Returns:
        tensor: (1, 48, n_features) - 30분 단위 24시간
    """
    if not data_list or not aws368 or not aws541 or not aws569:
        raise ValueError("dataList and awsList are empty")

    # DataFrame 생성
    df = pd.DataFrame(data_list)
    aws368 = pd.DataFrame(aws368)
    aws541 = pd.DataFrame(aws541)
    aws569 = pd.DataFrame(aws569)

    # 시간 컬럼 처리
    if "SYS_TIME" in df.columns:
        df["SYS_TIME"] = pd.to_datetime(df["SYS_TIME"], errors="coerce")
        df = df.set_index("SYS_TIME").sort_index()
    else:
        raise ValueError("SYS_TIME column is required in dataList")
    
    if "SYS_TIME" in aws368.columns:
        aws368["SYS_TIME"] = pd.to_datetime(aws368["SYS_TIME"], errors="coerce")
        aws368 = aws368.set_index("SYS_TIME").sort_index()
    else:
        raise ValueError("SYS_TIME column is required in aws368")

    if "SYS_TIME" in aws541.columns:
        aws541["SYS_TIME"] = pd.to_datetime(aws541["SYS_TIME"], errors="coerce")
        aws541 = aws541.set_index("SYS_TIME").sort_index()
    else:
        raise ValueError("SYS_TIME column is required in aws541")

    if "SYS_TIME" in aws569.columns:
        aws569["SYS_TIME"] = pd.to_datetime(aws569["SYS_TIME"], errors="coerce")
        aws569 = aws569.set_index("SYS_TIME").sort_index()
    else:
        raise ValueError("SYS_TIME column is required in aws569")
    
    # AWS 데이터 병합
    aws = aws368.add_suffix("_368").join(
        aws541.add_suffix("_541"), how="inner"
    ).join(
        aws569.add_suffix("_569"), how="inner"
    )

    # AWS 데이터와 병합 (내부 조인)
    df = df.join(aws, how="inner")

    # 30분 리샘플링
    df_resampled = resample_to_30min(df)

    # 특성 엔지니어링 (LSTM.ipynb와 동일)
    df_features = apply_feature_engineering(df_resampled)

    # 필요한 특성만 선택 (학습 시 사용한 특성 순서대로)
    missing_features = [f for f in FEATURE_NAMES if f not in df_features.columns]
    if missing_features:
        # 누락된 특성은 0으로 채움
        for feat in missing_features:
            df_features[feat] = 0.0

    # 특성 순서 맞춤
    X = df_features[FEATURE_NAMES].values

    # 정규화
    X_scaled = x_scaler.transform(X)

    # 텐서 변환: (48, n_features) -> (1, 48, n_features)
    tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

    return tensor

def autoregressive_predict(input_tensor: torch.Tensor, n_steps: int) -> list[float]:
    """
    Autoregressive 방식으로 다중 시점 예측

    Args:
        input_tensor: (1, 48, n_features) 초기 입력
        n_steps: 예측할 시점 수 (1h=2, 3h=6, 12h=24)

    Returns:
        predictions: 각 시점의 예측값 (역정규화된 실제 값)
    """
    predictions = []
    current_input = input_tensor.clone()

    with torch.no_grad():
        for step in range(n_steps):
            # 현재 윈도우로 1시간 후 예측
            pred_scaled = model(current_input.to(device))  # (1, 1)

            # 역정규화
            pred_original = y_scaler.inverse_transform(pred_scaled.cpu().numpy())
            predictions.append(float(pred_original[0, 0]))

            # 다음 예측을 위해 윈도우 업데이트
            # 방법 1: 예측값을 다시 정규화하여 입력에 추가
            # 하지만 우리는 Q_in만 예측하므로, 전체 특성을 업데이트해야 함
            # 간단하게: 마지막 특성 벡터를 복사하고 Q_in만 업데이트

            # 마지막 시점의 특성 벡터 가져오기
            last_features = current_input[0, -1, :].cpu().numpy()  # (n_features,)

            # Q_in 위치 찾기 (첫 번째 특성이라고 가정)
            # 실제로는 FEATURE_NAMES에서 Q_in의 인덱스를 찾아야 함
            if "Q_in" in FEATURE_NAMES:
                q_in_idx = FEATURE_NAMES.index("Q_in")
                # 예측값을 정규화하여 업데이트
                last_features[q_in_idx] = pred_scaled.cpu().numpy()[0, 0]

            # 새로운 특성 벡터를 윈도우에 추가 (오른쪽으로 shift)
            new_features = torch.tensor(last_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, n_features)
            current_input = torch.cat([current_input[:, 1:, :], new_features], dim=1)  # (1, 48, n_features)

    return predictions

# ====== API 모델 정의 ======
class FlowInput(BaseModel):
    SYS_TIME: str
    flow_TankA: float
    flow_TankB: float
    level_TankA: float
    level_TankB: float
    Q_in: float

class TMSInput(BaseModel):
    SYS_TIME: str
    TOC_VU: float
    PH_VU: float
    SS_VU: float
    FLUX_VU: float
    TN_VU: float
    TP_VU: float

class AWSInput(BaseModel):
    SYS_TIME: str
    TA: float
    RN_15m: float
    RN_60m: float
    RN_12H: float
    RN_DAY: float
    HM: float
    TD: float
    distance: float

class PredinctInput(BaseModel):
    dataList: list[FlowInput] | list[TMSInput]
    awsList: dict[str, list[AWSInput]]  # {"stn_368": [...], "stn_541": [...], "stn_569": [...]}

class PredictIn(BaseModel):
    model_config = {"populate_by_name": True}
    request_id: str | None = None
    input: PredinctInput = Field(alias="in")

class PredictOut(BaseModel):
    request_id: str
    ok: bool
    output: dict | None = None
    latency_ms: int
    error: dict | None = None

# ====== API 엔드포인트 ======
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/ready")
def ready():
    return {
        "ok": True,
        "model_loaded": True,
        "model_version": "0.2.0",
        "model_name": "flow_lstm_model",
        "n_features": N_FEATURES,
        "window_size": 48,
        "horizon_unit": "30min"
    }

@app.post("/debug/flow")
async def debug_flow(request: Request):
    raw = await request.body()
    headers = dict(request.headers)

    print("=== HEADERS ===")
    for k in ["content-type", "content-length", "transfer-encoding", "connection", "upgrade", "host"]:
        print(f"{k}: {headers.get(k)}")

    print(f"=== RAW BODY LENGTH: {len(raw)} bytes ===")
    return {
        "content_type": headers.get("content-type"),
        "content_length": headers.get("content-length"),
        "transfer_encoding": headers.get("transfer-encoding"),
        "connection": headers.get("connection"),
        "upgrade": headers.get("upgrade"),
        "body_length": len(raw),
        "preview": raw[:500].decode("utf-8", errors="replace"),
    }

@app.post("/predict/tms", response_model=PredictOut)
def predict(x: PredictIn):
    try:
        t0 = time.perf_counter()
        rid = x.request_id or str(uuid.uuid4())
    
        tms_list = [item.model_dump() for item in x.input.dataList]
        # Backend sends lowercase keys (stn_368), fallback to uppercase (STN_368)
        aws368 = [item.model_dump() for item in (x.input.awsList.get("stn_368") or x.input.awsList.get("STN_368", []))]
        aws541 = [item.model_dump() for item in (x.input.awsList.get("stn_541") or x.input.awsList.get("STN_541", []))]
        aws569 = [item.model_dump() for item in (x.input.awsList.get("stn_569") or x.input.awsList.get("STN_569", []))]

        if not tms_list or not aws368 or not aws541 or not aws569:
            raise ValueError("input.dataList and input.awsList (STN_368, STN_541, STN_569) are required")

        if len(tms_list) != 1440 or len(aws368) != 1440 or len(aws541) != 1440 or len(aws569) != 1440:  # 최소 24시간 필요 (1분 단위 1440개 -> 30분 단위 48개)
            raise ValueError(f"tmsList and awsList require exactly 1440 records (24 hours), got {len(tms_list)}, {len(aws368)}, {len(aws541)}, {len(aws569)}")

        # 데이터 전처리
        input_tensor = preprocess_input(tms_list, aws368, aws541, aws569)

        # Autoregressive 예측
        # 1시간 후 = 2 steps (30분 × 2)
        pred_1h = autoregressive_predict(input_tensor, n_steps=2)
        pred_2h = autoregressive_predict(input_tensor, n_steps=4)
        pred_3h = autoregressive_predict(input_tensor, n_steps=6)
        pred_4h = autoregressive_predict(input_tensor, n_steps=8)
        pred_5h = autoregressive_predict(input_tensor, n_steps=10)
        pred_6h = autoregressive_predict(input_tensor, n_steps=12)
        pred_7h = autoregressive_predict(input_tensor, n_steps=14)
        pred_8h = autoregressive_predict(input_tensor, n_steps=16)
        pred_9h = autoregressive_predict(input_tensor, n_steps=18)
        pred_10h = autoregressive_predict(input_tensor, n_steps=20)
        pred_11h = autoregressive_predict(input_tensor, n_steps=22)
        pred_12h = autoregressive_predict(input_tensor, n_steps=24)

        latency = int((time.perf_counter() - t0) * 1000)

        return PredictOut(
            request_id=rid,
            ok=True,
            output={
                "predictions": {
                    "1h": pred_1h[-1],   # 1시간 후 최종 예측값
                    "2h": pred_2h[-1],   # 2시간 후 최종 예측값
                    "3h": pred_3h[-1],   # 3시간 후 최종 예측값
                    "4h": pred_4h[-1],   # 4시간 후 최종 예측값
                    "5h": pred_5h[-1],   # 5시간 후 최종 예측값
                    "6h": pred_6h[-1],   # 6시간 후 최종 예측값
                    "7h": pred_7h[-1],   # 7시간 후 최종 예측값
                    "8h": pred_8h[-1],   # 8시간 후 최종 예측값
                    "9h": pred_9h[-1],   # 9시간 후 최종 예측값
                    "10h": pred_10h[-1], # 10시간 후 최종 예측값
                    "11h": pred_11h[-1], # 11시간 후 최종 예측값
                    "12h": pred_12h[-1], # 12시간 후 최종 예측값
                },
                "trajectories": {
                    "12h": pred_12h, # 12시간까지의 전체 궤적 (30분 간격 24개 값)
                },
                "metadata": {
                    "window_size": 48,
                    "n_features": N_FEATURES,
                    "input_records": len(tms_list),
                    "resampled_steps": input_tensor.shape[1],
                }
            },
            latency_ms=latency,
            error=None
        )
    except Exception as e:
        import traceback
        error_detail = {
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=400, detail=error_detail)


@app.post("/predict/flow", response_model=PredictOut)
def predict(x: PredictIn):
    try:
        t0 = time.perf_counter()
        rid = x.request_id or str(uuid.uuid4())
    
        flow_list = [item.model_dump() for item in x.input.dataList]
        # Backend sends lowercase keys (stn_368), fallback to uppercase (STN_368)
        aws368 = [item.model_dump() for item in (x.input.awsList.get("stn_368") or x.input.awsList.get("STN_368", []))]
        aws541 = [item.model_dump() for item in (x.input.awsList.get("stn_541") or x.input.awsList.get("STN_541", []))]
        aws569 = [item.model_dump() for item in (x.input.awsList.get("stn_569") or x.input.awsList.get("STN_569", []))]

        if not flow_list or not aws368 or not aws541 or not aws569:
            raise ValueError("input.dataList and input.awsList (STN_368, STN_541, STN_569) are required")

        if len(flow_list) != 1440 or len(aws368) != 1440 or len(aws541) != 1440 or len(aws569) != 1440:  # 최소 24시간 필요 (1분 단위 1440개 -> 30분 단위 48개)
            raise ValueError(f"flowList and awsList require exactly 1440 records (24 hours), got {len(flow_list)}, {len(aws368)}, {len(aws541)}, {len(aws569)}")

        # 데이터 전처리
        input_tensor = preprocess_input(flow_list, aws368, aws541, aws569)

        # Autoregressive 예측
        # 1시간 후 = 2 steps (30분 × 2)
        pred_1h = autoregressive_predict(input_tensor, n_steps=2)
        pred_2h = autoregressive_predict(input_tensor, n_steps=4)
        pred_3h = autoregressive_predict(input_tensor, n_steps=6)
        pred_4h = autoregressive_predict(input_tensor, n_steps=8)
        pred_5h = autoregressive_predict(input_tensor, n_steps=10)
        pred_6h = autoregressive_predict(input_tensor, n_steps=12)
        pred_7h = autoregressive_predict(input_tensor, n_steps=14)
        pred_8h = autoregressive_predict(input_tensor, n_steps=16)
        pred_9h = autoregressive_predict(input_tensor, n_steps=18)
        pred_10h = autoregressive_predict(input_tensor, n_steps=20)
        pred_11h = autoregressive_predict(input_tensor, n_steps=22)
        pred_12h = autoregressive_predict(input_tensor, n_steps=24)

        latency = int((time.perf_counter() - t0) * 1000)

        return PredictOut(
            request_id=rid,
            ok=True,
            output={
                "predictions": {
                    "1h": pred_1h[-1],   # 1시간 후 최종 예측값
                    "2h": pred_2h[-1],   # 2시간 후 최종 예측값
                    "3h": pred_3h[-1],   # 3시간 후 최종 예측값
                    "4h": pred_4h[-1],   # 4시간 후 최종 예측값
                    "5h": pred_5h[-1],   # 5시간 후 최종 예측값
                    "6h": pred_6h[-1],   # 6시간 후 최종 예측값
                    "7h": pred_7h[-1],   # 7시간 후 최종 예측값
                    "8h": pred_8h[-1],   # 8시간 후 최종 예측값
                    "9h": pred_9h[-1],   # 9시간 후 최종 예측값
                    "10h": pred_10h[-1], # 10시간 후 최종 예측값
                    "11h": pred_11h[-1], # 11시간 후 최종 예측값
                    "12h": pred_12h[-1], # 12시간 후 최종 예측값
                },
                "trajectories": {
                    "12h": pred_12h, # 12시간까지의 전체 궤적 (30분 간격 24개 값)
                },
                "metadata": {
                    "window_size": 48,
                    "n_features": N_FEATURES,
                    "input_records": len(flow_list),
                    "resampled_steps": input_tensor.shape[1],
                }
            },
            latency_ms=latency,
            error=None
        )
    except Exception as e:
        import traceback
        error_detail = {
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=400, detail=error_detail)