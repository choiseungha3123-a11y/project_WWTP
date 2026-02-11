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

app = FastAPI(title="WWTP Prediction API", version="0.3.0")

# ====== 경로 설정 ======
MODEL_DIR = BASE_DIR / "model"
SAVE_DIR = MODEL_DIR / "save"
DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "recommand_features" / "save"

# ====== TMS 모델 정의 ======
class LSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        out_size=1,
        use_attention=True,
        head_hidden_size: int | None = None,
    ):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size

        if self.use_attention:
            self.layer_norm1 = nn.LayerNorm(lstm_out_size)
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_out_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm2 = nn.LayerNorm(lstm_out_size)

            head_size = head_hidden_size or lstm_out_size
            self.head = nn.Sequential(
                nn.Linear(lstm_out_size, head_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_size, out_size)
            )
        else:
            head_size = head_hidden_size or (lstm_out_size // 2)
            self.head = nn.Sequential(
                nn.Linear(lstm_out_size, head_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_size, out_size)
            )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            lstm_out_normed = self.layer_norm1(lstm_out)
            attn_out, _ = self.attention(lstm_out_normed, lstm_out_normed, lstm_out_normed)
            attn_out = attn_out + lstm_out
            attn_out = self.layer_norm2(attn_out)
            last = attn_out[:, -1, :]
        else:
            last = lstm_out[:, -1, :]

        yhat = self.head(last)
        return yhat

# ====== Flow 전용 LSTM (저장된 체크포인트 구조에 맞춤) ======
class FlowLSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size=128,
        num_layers=4,
        dropout=0.3,
        out_size=1,
        use_attention=True,
        num_heads=8,
    ):
        super(FlowLSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size

        if self.use_attention:
            self.layer_norm1 = nn.LayerNorm(lstm_out_size)
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_out_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.layer_norm2 = nn.LayerNorm(lstm_out_size)

            # 체크포인트의 head.* 구조(0,1,4,5,8 인덱스)에 맞춘 구성
            self.head = nn.Sequential(
                nn.Linear(lstm_out_size, lstm_out_size * 2),
                nn.LayerNorm(lstm_out_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_size * 2, lstm_out_size),
                nn.LayerNorm(lstm_out_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_size, out_size),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(lstm_out_size, lstm_out_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 2, out_size),
            )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            lstm_out_normed = self.layer_norm1(lstm_out)
            attn_out, _ = self.attention(lstm_out_normed, lstm_out_normed, lstm_out_normed)
            attn_out = attn_out + lstm_out
            attn_out = self.layer_norm2(attn_out)
            last = attn_out[:, -1, :]
        else:
            last = lstm_out[:, -1, :]

        yhat = self.head(last)
        return yhat

# ====== StandardScaler 정의 (LSTM_TMS.ipynb와 동일) ======
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

# ====== 모델 설정 ======
TMS_TARGETS = {
    "toc":  {"target_col": "TOC_VU",  "hidden_size": 128, "num_layers": 3, "dropout": 0.2, "use_attention": False},
    "ss":   {"target_col": "SS_VU",   "hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "use_attention": True},
    "tn":   {"target_col": "TN_VU",   "hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "use_attention": False},
    "tp":   {"target_col": "TP_VU",   "hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "use_attention": True},
    "flux": {"target_col": "FLUX_VU", "hidden_size": 128, "num_layers": 3, "dropout": 0.2, "use_attention": True},
    "ph":   {"target_col": "PH_VU",   "hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "use_attention": False},
}

FLOW_CONFIG = {"hidden_size": 128, "num_layers": 3, "dropout": 0.3, "use_attention": True, "num_heads": 8}

# ====== 공통 유틸 ======
def load_feature_names(csv_path: Path) -> list[str]:
    feat_df = pd.read_csv(csv_path)
    return feat_df["feature_name"].tolist()

def load_scaler(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_model_weights(model: nn.Module, ckpt_path: Path, map_location=None) -> nn.Module:
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def build_hourly_predictions(pred_12h: list[float]) -> dict:
    return {f"{h}h": pred_12h[h * 2 - 1] for h in range(1, 13)}

def extract_input_lists(x):
    data_list = [item.model_dump() for item in x.input.dataList]
    aws368 = [item.model_dump() for item in (x.input.awsList.get("stn_368") or x.input.awsList.get("STN_368", []))]
    aws541 = [item.model_dump() for item in (x.input.awsList.get("stn_541") or x.input.awsList.get("STN_541", []))]
    aws569 = [item.model_dump() for item in (x.input.awsList.get("stn_569") or x.input.awsList.get("STN_569", []))]
    return data_list, aws368, aws541, aws569

def validate_input_lengths(data_list, aws368, aws541, aws569):
    if not data_list or not aws368 or not aws541 or not aws569:
        raise ValueError("input.dataList and input.awsList (stn_368, stn_541, stn_569) are required")
    if len(data_list) != 1440 or len(aws368) != 1440 or len(aws541) != 1440 or len(aws569) != 1440:
        raise ValueError(
            f"dataList and awsList require exactly 1440 records (24 hours), "
            f"got {len(data_list)}, {len(aws368)}, {len(aws541)}, {len(aws569)}"
        )

# ====== 모델 및 스케일러 로드 ======
print("Loading models and scalers...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pickle 호환성을 위해 현재 모듈의 StandardScaler를 등록
sys.modules['__main__'].StandardScaler = StandardScaler

# --- TMS 모델 로드 (6개) ---
tms_models = {}
tms_x_scalers = {}
tms_y_scalers = {}
tms_features = {}

for target_name, cfg in TMS_TARGETS.items():
    feature_names = load_feature_names(FEATURE_DIR / f"{target_name}_recommended_features.csv")
    n_features = len(feature_names)
    tms_features[target_name] = feature_names

    mdl = LSTMRegressor(
        n_features=n_features,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        out_size=1,
        use_attention=cfg["use_attention"],
        head_hidden_size=cfg.get("head_hidden_size"),
    ).to(device)

    tms_models[target_name] = load_model_weights(mdl, SAVE_DIR / f"{target_name}_lstm_model.pth", device)
    tms_x_scalers[target_name] = load_scaler(SAVE_DIR / f"X_scaler_{target_name}.pkl")
    tms_y_scalers[target_name] = load_scaler(SAVE_DIR / f"y_scaler_{target_name}.pkl")

    print(f"  Loaded {target_name}: {n_features} features, attention={cfg['use_attention']}")

# --- Flow 모델 로드 ---
FLOW_FEATURE_NAMES = load_feature_names(FEATURE_DIR / "flow_recommended_features.csv")
FLOW_N_FEATURES = len(FLOW_FEATURE_NAMES)

flow_model = FlowLSTMRegressor(
    n_features=FLOW_N_FEATURES,
    hidden_size=FLOW_CONFIG["hidden_size"],
    num_layers=FLOW_CONFIG["num_layers"],
    dropout=FLOW_CONFIG["dropout"],
    out_size=1,
    use_attention=FLOW_CONFIG["use_attention"],
    num_heads=FLOW_CONFIG["num_heads"],
).to(device)

flow_model = load_model_weights(flow_model, SAVE_DIR / "flow_lstm_model.pth", device)
flow_x_scaler = load_scaler(SAVE_DIR / "X_scaler_flow.pkl")
flow_y_scaler = load_scaler(SAVE_DIR / "y_scaler_flow.pkl")

print(f"  Loaded flow: {FLOW_N_FEATURES} features")
print("All models loaded successfully")

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

# ====== 공통 데이터 전처리 함수 ======
def resample_to_30min(df: pd.DataFrame) -> pd.DataFrame:
    """1분 단위 데이터를 30분 단위로 리샘플링"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg_dict = {}
    for col in numeric_cols:
        if col.startswith("RN_") or col.startswith("AR_") or col == "FLUX_VU":
            agg_dict[col] = "sum"
        else:
            agg_dict[col] = "mean"

    return df[numeric_cols].resample("30min").agg(agg_dict)


def merge_input_data(data_list, aws368, aws541, aws569) -> pd.DataFrame:
    """
    백엔드 입력 데이터를 병합하여 30분 리샘플링된 DataFrame 반환

    Args:
        data_list: 1분 단위 24시간 데이터 (1440개 레코드)
        aws368, aws541, aws569: AWS 데이터

    Returns:
        df_resampled: 30분 리샘플링된 DataFrame (DatetimeIndex)
    """
    if not data_list or not aws368 or not aws541 or not aws569:
        raise ValueError("dataList and awsList are empty")

    df = pd.DataFrame(data_list)
    aws368 = pd.DataFrame(aws368)
    aws541 = pd.DataFrame(aws541)
    aws569 = pd.DataFrame(aws569)

    # 시간 컬럼 처리
    for name, frame in [("dataList", df), ("aws368", aws368), ("aws541", aws541), ("aws569", aws569)]:
        if "SYS_TIME" not in frame.columns:
            raise ValueError(f"SYS_TIME column is required in {name}")
        frame["SYS_TIME"] = pd.to_datetime(frame["SYS_TIME"], errors="coerce")

    df = df.set_index("SYS_TIME").sort_index()
    aws368 = aws368.set_index("SYS_TIME").sort_index()
    aws541 = aws541.set_index("SYS_TIME").sort_index()
    aws569 = aws569.set_index("SYS_TIME").sort_index()

    # AWS 데이터 병합
    aws = aws368.add_suffix("_368").join(
        aws541.add_suffix("_541"), how="inner"
    ).join(
        aws569.add_suffix("_569"), how="inner"
    )

    df = df.join(aws, how="inner")

    # FLUX_VU: 누적값 → 증분값 변환 (TMS 노트북과 동일, 리샘플링 전에 처리)
    if "FLUX_VU" in df.columns:
        flux = df["FLUX_VU"].copy()
        flux_diff = flux.diff()
        reset_mask = flux_diff < 0      # 일 초기화 지점
        flux_diff[reset_mask] = flux[reset_mask]
        flux_diff.iloc[0] = 0
        flux_diff = flux_diff.clip(lower=0)
        df["FLUX_VU"] = flux_diff

    # 30분 리샘플링
    df_resampled = resample_to_30min(df)

    return df_resampled


# ====== 공통 특성 엔지니어링 파이프라인 ======
def _apply_common_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for feature engineering")

    df_features = feat_eng.add_rain_features(df)
    df_features = feat_eng.add_station_agg_rain_features(df_features)
    df_features = feat_eng.add_weather_features(df_features)
    df_features = feat_eng.add_process_features(df_features)
    df_features = feat_eng.add_temporal_features(df_features)
    df_features = feat_eng.add_time_features(df_features)

    # 노트북과 동일: weekday/iso_week/hour_x_weekday 보장
    if "weekday" not in df_features.columns and "dayofweek" in df_features.columns:
        df_features["weekday"] = df_features["dayofweek"]
    if "iso_week" not in df_features.columns and isinstance(df_features.index, pd.DatetimeIndex):
        df_features["iso_week"] = df_features.index.isocalendar().week.astype(int).to_numpy()
    if "hour_x_weekday" not in df_features.columns:
        if "hour" in df_features.columns and "weekday" in df_features.columns:
            df_features["hour_x_weekday"] = df_features["hour"] * df_features["weekday"]
        elif "hour" in df_features.columns and "dayofweek" in df_features.columns:
            df_features["hour_x_weekday"] = df_features["hour"] * df_features["dayofweek"]

    df_features = df_features.ffill().fillna(0)
    return df_features


# ====== Flow 전처리 함수 ======
def apply_flow_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flow 모델용 특성 엔지니어링 (LSTM_FLOW.ipynb와 동일)

    1. flow_TankA/B lag 특성 추가 (min_lag=2)
    2. raw flow_TankA/B 제거 (lag 특성은 유지)
    3. raw Q_in(타겟) 제거
    4. 특성 엔지니어링 파이프라인 적용
    """
    # 1. flow_TankA/B lag 특성 추가 (제거 전에 실행)
    flow_lag_cols = [c for c in ["flow_TankA", "flow_TankB"] if c in df.columns]
    if flow_lag_cols:
        df = feat_eng.add_target_lag_features(df, flow_lag_cols, min_lag=2)

    # 2. raw flow_TankA/B 제거 (lag 특성은 유지)
    df = df.drop(columns=["flow_TankA", "flow_TankB"], errors='ignore')

    # 3. raw Q_in(타겟) 제거
    df_base = df.drop(columns=["Q_in"], errors='ignore')

    # 4. 특성 엔지니어링 파이프라인
    return _apply_common_feature_pipeline(df_base)


def preprocess_flow_input(data_list, aws368, aws541, aws569) -> torch.Tensor:
    """Flow 모델 입력 전처리: 병합 → 리샘플링 → 특성 엔지니어링 → 텐서"""
    df_resampled = merge_input_data(data_list, aws368, aws541, aws569)
    df_features = apply_flow_feature_engineering(df_resampled)

    for feat in FLOW_FEATURE_NAMES:
        if feat not in df_features.columns:
            df_features[feat] = 0.0

    X = df_features[FLOW_FEATURE_NAMES].values
    X_scaled = flow_x_scaler.transform(X)
    tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    return tensor


# ====== TMS 전처리 함수 ======
def apply_tms_feature_engineering(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """
    TMS 모델용 타겟별 특성 엔지니어링 (LSTM_TMS.ipynb와 동일)

    1. 타겟 lag 특성 추가 (min_lag=2)
    2. raw 타겟 컬럼 제거
    3. 특성 엔지니어링 파이프라인 적용
    """
    cfg = TMS_TARGETS[target_name]
    target_col = cfg["target_col"]

    df_work = df.copy()

    # 1. 타겟 lag 특성 추가 (분리 전에 실행)
    if target_col in df_work.columns:
        df_work = feat_eng.add_target_lag_features(df_work, [target_col], min_lag=2)

    # 2. raw 타겟 컬럼 제거 (lag 특성은 유지)
    if target_col in df_work.columns:
        df_work = df_work.drop(columns=[target_col])

    # 3. 특성 엔지니어링 파이프라인
    return _apply_common_feature_pipeline(df_work)


def make_tms_tensor(df_resampled: pd.DataFrame, target_name: str) -> torch.Tensor:
    """타겟별 특성 엔지니어링 → 추천 특성 선택 → 스케일링 → 텐서"""
    df_features = apply_tms_feature_engineering(df_resampled, target_name)
    feature_names = tms_features[target_name]

    for feat in feature_names:
        if feat not in df_features.columns:
            df_features[feat] = 0.0

    X = df_features[feature_names].values
    X_scaled = tms_x_scalers[target_name].transform(X)
    tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    return tensor


# ====== 자기회귀 예측 함수 ======
def autoregressive_predict(
    input_tensor: torch.Tensor,
    n_steps: int,
    pred_model: nn.Module,
    y_scaler_obj,
    target_feat_idx: int | None = None,
) -> list[float]:
    """
    Autoregressive 방식으로 다중 시점 예측

    Args:
        input_tensor: (1, 48, n_features) 초기 입력
        n_steps: 예측할 시점 수 (1h=2, 3h=6, 12h=24)
        pred_model: 예측에 사용할 모델
        y_scaler_obj: 역정규화에 사용할 y 스케일러
        target_feat_idx: 특성 벡터에서 타겟 lag 특성의 인덱스 (None이면 업데이트 안 함)

    Returns:
        predictions: 각 시점의 예측값 (역정규화된 실제 값)
    """
    predictions = []
    current_input = input_tensor.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            pred_scaled = pred_model(current_input.to(device))  # (1, 1)

            pred_original = y_scaler_obj.inverse_transform(pred_scaled.cpu().numpy())
            predictions.append(float(pred_original[0, 0]))

            # 마지막 시점의 특성 벡터 복사 후 윈도우 시프트
            last_features = current_input[0, -1, :].cpu().numpy()

            if target_feat_idx is not None:
                last_features[target_feat_idx] = pred_scaled.cpu().numpy()[0, 0]

            new_features = torch.tensor(last_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            current_input = torch.cat([current_input[:, 1:, :], new_features], dim=1)

    return predictions


def find_target_lag_idx(feature_names: list[str], target_col: str) -> int | None:
    """추천 특성 목록에서 타겟의 가장 작은 lag 특성 인덱스를 찾음"""
    prefix = f"{target_col}_tlag_"
    tlag_features = [(i, f) for i, f in enumerate(feature_names) if f.startswith(prefix)]
    if not tlag_features:
        return None
    tlag_features.sort(key=lambda x: int(x[1].split("_")[-1]))
    return tlag_features[0][0]


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

class FlowPredictInput(BaseModel):
    dataList: list[FlowInput]
    awsList: dict[str, list[AWSInput]]

class TMSPredictInput(BaseModel):
    dataList: list[TMSInput]
    awsList: dict[str, list[AWSInput]]

class FlowPredictIn(BaseModel):
    model_config = {"populate_by_name": True}
    request_id: str | None = None
    input: FlowPredictInput = Field(alias="in")

class TMSPredictIn(BaseModel):
    model_config = {"populate_by_name": True}
    request_id: str | None = None
    input: TMSPredictInput = Field(alias="in")

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
        "model_version": "0.3.0",
        "models_loaded": {
            "flow": {
                "n_features": FLOW_N_FEATURES,
            },
            "tms": {
                target_name: {
                    "n_features": len(tms_features[target_name]),
                    "use_attention": TMS_TARGETS[target_name]["use_attention"],
                }
                for target_name in TMS_TARGETS
            },
        },
        "window_size": 48,
        "horizon_unit": "30min",
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
def predict_tms(x: TMSPredictIn):
    try:
        t0 = time.perf_counter()
        rid = x.request_id or str(uuid.uuid4())

        tms_list, aws368, aws541, aws569 = extract_input_lists(x)
        validate_input_lengths(tms_list, aws368, aws541, aws569)

        # 공통: 데이터 병합 + 30분 리샘플링 (1회)
        df_resampled = merge_input_data(tms_list, aws368, aws541, aws569)

        all_predictions = {}
        all_trajectories = {}

        for target_name, cfg in TMS_TARGETS.items():
            # 타겟별 전처리 → 텐서
            input_tensor = make_tms_tensor(df_resampled, target_name)

            mdl = tms_models[target_name]
            y_sc = tms_y_scalers[target_name]
            feature_names = tms_features[target_name]
            target_feat_idx = find_target_lag_idx(feature_names, cfg["target_col"])

            # 12시간 = 24 steps (30분 × 24) 한 번에 예측
            pred_12h = autoregressive_predict(input_tensor, 24, mdl, y_sc, target_feat_idx)

            # 시간별 예측값 추출 (매 2 steps = 1시간)
            hourly_preds = build_hourly_predictions(pred_12h)

            all_predictions[target_name] = hourly_preds
            all_trajectories[target_name] = {"12h": pred_12h}

        latency = int((time.perf_counter() - t0) * 1000)

        return PredictOut(
            request_id=rid,
            ok=True,
            output={
                "predictions": all_predictions,
                "trajectories": all_trajectories,
                "metadata": {
                    "window_size": 48,
                    "targets": list(TMS_TARGETS.keys()),
                    "input_records": len(tms_list),
                    "resampled_steps": df_resampled.shape[0],
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
def predict_flow(x: FlowPredictIn):
    try:
        t0 = time.perf_counter()
        rid = x.request_id or str(uuid.uuid4())

        flow_list, aws368, aws541, aws569 = extract_input_lists(x)
        validate_input_lengths(flow_list, aws368, aws541, aws569)

        input_tensor = preprocess_flow_input(flow_list, aws368, aws541, aws569)

        # Flow는 flow_TankA/B lag 사용 (Q_in lag 없음)
        # Q_in을 flow_TankA/B로 역분해할 수 없으므로 lag 업데이트 안 함
        flow_target_idx = None

        # 12시간 = 24 steps 한 번에 예측
        pred_12h = autoregressive_predict(input_tensor, 24, flow_model, flow_y_scaler, flow_target_idx)

        # 시간별 예측값 추출
        hourly_preds = build_hourly_predictions(pred_12h)

        latency = int((time.perf_counter() - t0) * 1000)

        return PredictOut(
            request_id=rid,
            ok=True,
            output={
                "predictions": hourly_preds,
                "trajectories": {
                    "12h": pred_12h,
                },
                "metadata": {
                    "window_size": 48,
                    "n_features": FLOW_N_FEATURES,
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
