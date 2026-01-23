from fastapi import FastAPI
from pydantic import BaseModel
import time
import uuid

app = FastAPI(title="Model Serving API", version="0.1.0")

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
