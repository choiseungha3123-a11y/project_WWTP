# 하수 유입량 · 수질(TMS) 예측 및 이상 진단 AI 서비스

## 1. Overview
본 프로젝트는 하수처리장의 **미래 유입량**과 **수질(TMS)** 을 사전에 예측하고,  
운영 기준을 초과할 가능성이 있을 경우 **사전 경고 및 이상 진단**을 제공하는  
AI 기반 의사결정 지원 웹 서비스이다.

- **예측(Forecasting)**: 유입량, TMS 세부 지표
- **분석(Analytics)**: 시간·계절 패턴 및 기상 변수 상관 분석
- **진단(Diagnosis)**: 실시간 이상 여부 판정 및 알림

---

## 2. Project Goals
### 예측 목표
- 유입량 예측 정확도 **95%**
- TMS 세부 지표 예측 정확도 **90%**

### 분석 목표
- 시간별 / 일별 / 계절별 유입량 변동 패턴 분석
- 기상 요인(강우량, 기온 등)과 유입량의 상관 관계 분석

### 진단 목표
- 사용자 정의 기준 기반 실시간 이상 여부 판정
- 이상 발생 시 즉각적인 경고 제공

---

## 3. Features
### Forecasting
- 유입량 시계열 예측
- TMS 지표(TOC, PH, SS, FLUX, TN, TP) 예측
- 예측 구간(신뢰 구간) 시각화
- 기준값 초과 예상 시 사전 경고 표시

### Analytics
- KPI 대시보드
  - 평균 유입량
  - 변동 범위
  - 월별 / 계절별 추이
- 다변량 상관 분석 결과 시각화
- 기상 변수 ↔ 유입량 관계 분석

### Diagnosis
- Isolation Forest 기반 이상 탐지
- 유입량 및 TMS 지표 이상 여부 실시간 판단
- 이상 발생 시 알림 표시

---

## 4. Data
- **데이터 종류**
  - 유입량 시계열 데이터
  - TMS 수질 지표(TOC, PH, SS, FLUX, TN, TP)
  - 기상 데이터(강우량, 기온, 습도)

- **전처리**
  - 결측치 보간 및 제거
  - 이상치 사전 필터링
  - 시간 기준 정렬 및 시차(feature lag) 생성

- **데이터 분할**
  - 시간 순서 기준 Train / Validation / Test 분할
  - 미래 정보 누수(Time Leakage) 방지

---

## 5. Models & Methods
### Baseline
- 다중 회귀 분석
  - 과거 유입량, 기상 변수, 시간 특성 활용

### Ensemble Models
- Random Forest
- XGBoost
  - 비선형 관계 및 변수 상호작용 학습

### Deep Learning
- LSTM
  - 시계열 장기 의존성 학습
  - Sliding Window 기반 다중 시점 입력

### Anomaly Detection
- Isolation Forest
  - 정상 패턴 학습 후 이상 점수 기반 판별
  - 사용자 기준과 병행 적용

---

## 6. Evaluation
- **유입량 예측**
  - MAE, RMSE, MAPE
  - |실제값 − 예측값| / 실제값 ≤ 5% 비율을 정확도로 정의

- **TMS 예측**
  - 지표별 MAE / RMSE / MAPE
  - 목표 정확도 90% 기준 충족 여부 평가

- **이상 진단**
  - 이상 이벤트 탐지 사례 기반 검증
  - 알림 발생 적합성 검토

---

## 7. System Architecture
데이터 수집
↓
전처리 · 피처 생성
↓
모델 학습 / 예측
↓
이상 탐지
↓
웹 대시보드 · 알림

---

## 8. Getting Started
### Environment
- Python **3.14**
- PyTorch **2.10.0**
- scikit-learn **1.8.0**, XGBoost **3.1.3**
- fastAPI **0.128.0**, uvicorn **0.40.0**

### Installation
```bash
conda create -n {venv name} python=3.14
conda activate {venv name}

pip install numpy pandas seaborn scikit-learn torch fastapi uvicorn

uvicorn main:app --host 0.0.0.0 --port 8000 --reload(optional)
```

### End Point

서버 프로세스 확인
```
GET /health

Response (200 OK):
{
  "ok": true
}
```

모델 서비스 준비 상태 확인
```
GET /ready

Response (200 OK):
{
  "ok": true,
  "model_loaded": true, 
  "model_version": "0.1.0"
}
```

모델 추론 1회 수행(요청당 1회)
```
POST /predict
Content-Type: application/json

Request Body:
{
  "request_id": "test-001",
  "input": {
    "text": "hello"
  }
}

Response (200 OK):
{
  "request_id": "test-001",
  "ok": true,
  "output": {
    "echo": {
      "text": "hello"
    }
  },
  "latency_ms": 0,
  "error": null
}
```

---

## 9. Repository Structure
```powershell
코드 복사
├── data/                     # 원천 및 전처리 데이터
├── model/                    # 학습된 모델
├── notebook/                 # EDA 및 분석 노트북
│   ├── linear_regression/    # 선형 회귀 코드
│   ├── collect/              # 데이터 수집 코드
│   └── preprocess/           # 전처리 코드
├── src/                      # fastAPI 실행
├── TODO.md
└── README.md
```

---

