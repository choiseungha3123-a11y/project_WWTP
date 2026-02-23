# Streamlit 포트폴리오 데모 앱 설계 문서

> WWTP 수질 예측 AI — Streamlit 인터랙티브 데모
> 작성일: 2026-02-23

---

## 1. 목적 및 개요

### 목적

- 학습된 7개 LSTM 모델의 성능과 개발 과정을 포트폴리오로 시각화
- 면접관/관계자가 URL 하나로 바로 확인할 수 있는 인터랙티브 데모 제공
- 정적 PDF/PPT가 아닌 **실제 예측 데이터 기반 동적 차트**로 차별화

### 핵심 원칙

- 기존 `src/` 코드를 재사용 (중복 없이)
- 저장된 예측 CSV를 활용한 인터랙티브 시각화 (라이브 추론 불필요)
- Streamlit Cloud 배포 가능한 구조 유지

---

## 2. 활용 가능한 기존 자산

| 자산 | 경로 | 설명 |
|------|------|------|
| 예측 결과 CSV | `data/output/save/{target}_predictions.csv` | actual / predicted 2컬럼, 각 596~1856행 |
| 예측 분석 PNG | `results/DL/save/prediction_analysis_{target}.png` | matplotlib 분석 차트 (6개: flow 제외) |
| 학습 곡선 PNG | `results/DL/save/{target}_learning_curve.png` | 7개 모두 존재 |
| 모델 가중치 | `model/save/{target}_lstm_model.pth` | 7개 |
| 스케일러 | `model/save/{X,y}_scaler_{target}.pkl` | 14개 |
| 추천 피처 목록 | `data/recommand_features/save/{target}_recommended_features.csv` | 타겟별 2~17개 |
| 모델/피처 설정 | `src/config.py`, `src/loader.py`, `src/models.py` | 재사용 |

### 예측 CSV 샘플 수

| 타겟 | 행 수 (test 샘플) |
|------|-----------------|
| flow | 596 |
| tn   | 1,724 |
| toc  | 1,856 |
| ss / tp / flux / ph | 유사 규모 |

---

## 3. 디렉토리 구조

```
python/
└── demo/
    ├── DESIGN.md               ← 이 문서
    ├── app.py                  ← 메인 진입점 (홈 페이지)
    ├── pages/
    │   ├── 1_성능_대시보드.py   ← R² 비교 + 개선 과정
    │   ├── 2_예측_분석.py       ← Plotly 인터랙티브 차트
    │   └── 3_모델_정보.py       ← 아키텍처 요약 + 피처 목록
    └── utils/
        ├── __init__.py
        ├── data_loader.py      ← CSV/PNG 로딩 헬퍼
        └── metrics.py          ← R², RMSE, MAE 계산
```

### 실행

```bash
cd python
streamlit run demo/app.py
```

---

## 4. 각 페이지 상세 설계

---

### 4-1. `app.py` — 홈

**목적**: 프로젝트 첫인상, 전체 요약

#### 레이아웃

```
┌─────────────────────────────────────────────┐
│  🌊 WWTP 수질 예측 AI                        │
│  하수처리장 7개 수질 지표 LSTM 예측 시스템     │
│                                              │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │
│  │  7개  │  │ 0.90 │  │ 30분 │  │12시간│   │
│  │LSTM  │  │최고R²│  │해상도│  │예측폭│   │
│  └──────┘  └──────┘  └──────┘  └──────┘   │
│                                              │
│  ── 시스템 구성 ──                            │
│  데이터 수집 → 전처리 → LSTM 예측 → API 서비스 │
│                                              │
│  ── R² 성능 요약 (수평 bar chart) ──          │
│  TN   ████████████████████  0.90            │
│  PH   ████████████████      0.84            │
│  Flow ███████████████       0.82            │
│  SS   █████████████         0.67            │
│  TP   ████████████          0.64            │
│  FLUX ████████████          0.63            │
│  TOC  ██████████            0.58            │
│                                              │
│  [좌측 사이드바에서 페이지 이동]               │
└─────────────────────────────────────────────┘
```

#### 구현 포인트

- `st.metric()` 4개 나란히 배치 (st.columns)
- 성능 bar chart: `plotly.express.bar` (horizontal, 색상: R² 기준 green/yellow/orange)
- 타겟 표시명 매핑: `{"flow": "유입유량", "tn": "총질소(TN)", ...}`

---

### 4-2. `pages/1_성능_대시보드.py` — 모델 성능

**목적**: 7개 모델 성능 비교 + 개선 과정 스토리텔링 + 학습 곡선/예측 분석 PNG

#### 레이아웃

```
── [섹션 1] 최종 R² 비교 ──────────────────────
  Plotly grouped bar chart
  (최종 R² 값, 7개 타겟 나란히)

── [섹션 2] 개발 단계별 성능 변화 ──────────────
  Plotly line chart (stage × target)

  데이터:
  ┌──────────┬───────┬───────┬───────┬───────┐
  │ Stage    │ Flow  │  TN   │  PH   │ TOC   │
  ├──────────┼───────┼───────┼───────┼───────┤
  │ 베이스라인 │  0.30 │ -0.16 │ -0.17 │ -1.86 │
  │ Lag 피처  │  0.79 │  0.78 │  0.56 │  0.30 │
  │ HP 최적화 │  0.82 │  0.90 │  0.75 │  0.47 │
  │ 최종      │  0.82 │  0.90 │  0.84 │  0.58 │
  └──────────┴───────┴───────┴───────┴───────┘

── [섹션 3] 타겟별 상세 ─────────────────────────
  탭 버튼: [Flow] [TOC] [SS] [TN] [TP] [FLUX] [PH]

  선택 시:
  ┌──────────────────┐  ┌──────────────────┐
  │  학습 곡선 PNG    │  │  예측 분석 PNG    │
  │ (train/val loss) │  │ (시계열+산점도)   │
  └──────────────────┘  └──────────────────┘
```

#### 구현 포인트

- 섹션 2 stage 데이터: NOTE.md 기준으로 하드코딩 (dict)
- 섹션 3 탭: `st.tabs([...])` 사용
- PNG: `st.image(Path(...))` — flow 예측분석 PNG는 `results/DL/prediction_analysis_flow.png` (save 폴더 아님 주의)

---

### 4-3. `pages/2_예측_분석.py` — 인터랙티브 예측 분석

**목적**: 실제 예측 CSV 기반 인터랙티브 시각화 (핵심 페이지)

#### 레이아웃

```
  타겟 선택:  ○ 유입유량  ○ 총질소  ○ 부유물질  ...
             (st.radio 또는 st.selectbox)

  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
  │  R²  │  │ RMSE │  │  MAE │  │샘플수│
  │ 0.90 │  │ 1.23 │  │ 0.87 │  │ 1724 │
  └──────┘  └──────┘  └──────┘  └──────┘

  ── 실측 vs 예측 시계열 ─────────────────────
  [Plotly line chart]
  - 파란선: 실측값
  - 빨간점선: 예측값
  - x축: 샘플 인덱스 (시간 인덱스 없으므로)
  - hover: 실측/예측/오차 동시 표시
  - 범위 선택 슬라이더 (rangeslider)

  ── 하단 2열 ──────────────────────────────
  ┌────────────────────┐  ┌────────────────────┐
  │  산점도             │  │  오차 분포          │
  │  (실측 vs 예측)     │  │  (histogram)       │
  │  y=x 기준선        │  │  mean/std 표시     │
  │  색상: 밀도         │  │                    │
  └────────────────────┘  └────────────────────┘
```

#### 구현 포인트

- CSV 로딩: `pd.read_csv("data/output/save/{target}_predictions.csv")`
- 메트릭 계산:
  ```python
  r2  = r2_score(actual, predicted)
  rmse = np.sqrt(mean_squared_error(actual, predicted))
  mae  = mean_absolute_error(actual, predicted)
  ```
- 시계열 차트: `plotly.graph_objects.Figure` (실측=solid, 예측=dash)
- 산점도: `plotly.express.scatter` + `add_trace(go.Line([min,max],[min,max]))` (y=x선)
- 오차 분포: `plotly.express.histogram(errors, nbins=50)`
- flow 예측 CSV는 샘플 수가 596건으로 적음 → 정상

---

### 4-4. `pages/3_모델_정보.py` — 모델 아키텍처

**목적**: 모델 구조와 피처 정보를 테이블로 정리

#### 레이아웃

```
── 모델 구조 요약 ──────────────────────────────
  ┌──────┬────────┬───────┬─────────┬──────────┬──────┐
  │ 타겟  │ hidden │ layers│attention│ head구조 │피처수│
  ├──────┼────────┼───────┼─────────┼──────────┼──────┤
  │ flow │  256   │   3   │   ✓    │  4-layer │  10  │
  │ toc  │  256   │   2   │   ✗    │  4-layer │   6  │
  │ ss   │  512   │   4   │   ✗    │  4-layer │   8  │
  │ tn   │  512   │   4   │   ✗    │  3-layer │   2  │
  │ tp   │  512   │   4   │   ✗    │  3-layer │  10  │
  │ flux │  512   │   4   │   ✓    │  3-layer │  17  │
  │ ph   │  512   │   4   │   ✗    │  3-layer │   7  │
  └──────┴────────┴───────┴─────────┴──────────┴──────┘

── 타겟별 선택 피처 목록 ──────────────────────
  타겟 선택: [TN ▼]

  선택된 피처 (2개):
  - TN_VU_tlag_2   (과거 2스텝 lag)
  - TN_VU_tdiff_2  (2스텝 차분)

── LSTM 아키텍처 다이어그램 ────────────────────
  (ASCII 텍스트 블록으로 표현)

  Input (batch, seq=48, n_features)
      ↓
  LSTM (num_layers, hidden_size)
      ↓
  [Attention] ← 타겟에 따라 선택
      ↓
  FC Head (3~4 layer)
      ↓
  Output (1)
```

#### 구현 포인트

- 모델 구조 테이블: `pd.DataFrame` + `st.dataframe()` (style 적용)
- 피처 수: 각 추천 피처 CSV의 행 수 - 1 (헤더 제외)
- 피처 목록: CSV 읽어서 `feature_name` 컬럼 표시
- head 구조 판정 로직: `src/config.py`의 `TMS_TARGETS` 재사용

---

## 5. 공통 유틸리티

### `utils/data_loader.py`

```python
# 주요 함수 목록

load_predictions(target: str) -> pd.DataFrame
    # data/output/save/{target}_predictions.csv 로딩
    # 컬럼: actual, predicted

load_feature_names(target: str) -> list[str]
    # data/recommand_features/save/{target}_recommended_features.csv

get_png_path(kind: str, target: str) -> Path
    # kind: "prediction" | "learning_curve"
    # results/DL/save/ 경로 반환
    # flow prediction: results/DL/prediction_analysis_flow.png (예외 처리)
```

### `utils/metrics.py`

```python
# 주요 함수 목록

compute_metrics(actual, predicted) -> dict
    # {"r2": ..., "rmse": ..., "mae": ...}
```

---

## 6. 데이터 상수 (하드코딩)

### 타겟 표시명

```python
TARGET_LABELS = {
    "flow": "유입유량 (Flow)",
    "toc":  "총유기탄소 (TOC)",
    "ss":   "부유물질 (SS)",
    "tn":   "총질소 (TN)",
    "tp":   "총인 (TP)",
    "flux": "방류유량 (FLUX)",
    "ph":   "수소이온농도 (pH)",
}
```

### 최종 R² 성능 (NOTE.md 기준)

```python
FINAL_R2 = {
    "flow": 0.8166,
    "toc":  0.5762,
    "ss":   0.6712,
    "tn":   0.9011,
    "tp":   0.6378,
    "flux": 0.6296,
    "ph":   0.8432,
}
```

### 단계별 성능 (개선 과정 스토리)

```python
STAGE_R2 = {
    "베이스라인": {"flow": 0.30, "toc": -1.86, "ss": -0.52, "tn": -0.16, "tp": -2.15, "flux": -0.01, "ph": -0.17},
    "Lag 피처":   {"flow": 0.79, "toc":  0.29, "ss":  0.21, "tn":  0.78, "tp": -0.41, "flux":  0.23, "ph":  0.56},
    "HP 최적화":  {"flow": 0.82, "toc":  0.47, "ss":  0.67, "tn":  0.90, "tp":  0.63, "flux":  0.61, "ph":  0.84},
    "최종":       {"flow": 0.82, "toc":  0.58, "ss":  0.67, "tn":  0.90, "tp":  0.64, "flux":  0.63, "ph":  0.84},
}
```

---

## 7. 기술 스택

| 역할 | 패키지 | 비고 |
|------|--------|------|
| UI 프레임워크 | `streamlit` | 멀티페이지 구조 |
| 인터랙티브 차트 | `plotly` | `plotly.express` + `plotly.graph_objects` |
| 데이터 처리 | `pandas`, `numpy` | |
| 메트릭 계산 | `scikit-learn` | `r2_score`, `mean_squared_error`, `mean_absolute_error` |
| 모델 코드 재사용 | `src/config.py`, `src/models.py` | 중복 없이 import |

### 추가 설치 필요

```bash
pip install streamlit plotly
```
(pandas, numpy, scikit-learn, torch는 이미 설치되어 있음)

---

## 8. 구현 순서

| 순서 | 작업 | 예상 난이도 |
|------|------|-------------|
| 1 | `utils/data_loader.py` + `utils/metrics.py` 작성 | 낮음 |
| 2 | `app.py` 홈 페이지 (메트릭 카드 + R² bar chart) | 낮음 |
| 3 | `pages/2_예측_분석.py` (Plotly 인터랙티브 차트) | 중간 |
| 4 | `pages/1_성능_대시보드.py` (개선 과정 + PNG 표시) | 중간 |
| 5 | `pages/3_모델_정보.py` (테이블 + 피처 목록) | 낮음 |
| 6 | 전체 스타일 통일 (색상, 폰트, 레이아웃) | 낮음 |

---

## 9. 주의사항

### 경로 처리

- `demo/app.py`에서 `src/`를 import하려면 `sys.path`에 `python/` 루트 추가 필요
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  ```
- 상대 경로 대신 `Path(__file__).parent.parent` 기준 절대 경로 사용

### Flow 예측 분석 PNG 경로 예외

- TMS 6개: `results/DL/save/prediction_analysis_{target}.png`
- Flow: `results/DL/prediction_analysis_flow.png` (save 폴더 아님)

### 피처 CSV 경로

- `data/recommand_features/save/` (save 서브폴더 있음)
- `data/recommand_features/` (save 없음, 일부 타겟만 존재)
- 로딩 시 `save/` 경로 우선 사용

---

*작성일: 2026-02-23*
