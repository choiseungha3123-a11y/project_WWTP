from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import uuid

app = FastAPI(title="Model Serving API", version="0.1.0")

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
    t0 = time.perf_counter()

    rid = x.request_id or str(uuid.uuid4())

    # TODO: 여기서 모델 추론 수행
    # 예: y = model(x.input)

    latency = int((time.perf_counter() - t0) * 1000)
    return PredictOut(
        request_id=rid,
        ok=True,
        output={"echo": x.input},
        latency_ms=latency,
        error=None
    )
