# Streamlit 포트폴리오 데모 앱 설계 문서

> WWTP 수질 예측 AI — Streamlit 인터랙티브 데모
> 작성일: 2026-02-23 | 최종 업데이트: 2026-02-24

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
| 예측 분석 PNG | `results/DL/save/prediction_analysis_{target}.png` | matplotlib 분석 차트 (7개) |
| 학습 곡선 PNG | `results/DL/save/{target}_learning_curve.png` | 7개 모두 존재 |
| **이상 진단 PNG** | `results/DL/save/{target}_diagnosis.png` | Isolation Forest 이상 탐지 차트 (7개) |
| **HP 탐색 결과 CSV** | `results/DL/{target}_experiment_results.csv` | 타겟별 하이퍼파라미터 실험 결과 (7개) |
| **상관관계 PNG** | `results/correlation/flow.png`, `results/correlation/tms.png` | 피처-타겟 상관 분석 |
| **ML 비교 PNG** | `results/ML/flow_r2_comparison.png`, `results/ML/modelA_r2_comparison.png` 등 | ML 모델 비교 |
| 모델 가중치 | `model/save/{target}_lstm_model.pth` | 7개 |
| 스케일러 | `model/save/{X,y}_scaler_{target}.pkl` | 14개 |
| 추천 피처 목록 | `data/recommand_features/save/{target}_recommended_features.csv` | 타겟별 2~17개 |
| 모델/피처 설정 | `src/config.py`, `src/loader.py`, `src/models.py` | 재사용 |

### 예측 CSV 샘플 수

| 타겟 | 행 수 (test 샘플) |
|------|--------------------|
| flow | 596 |
| tn   | 1,724 |
| toc  | 1,856 |
| ss / tp / flux / ph | 유사 규모 |

---

## 3. 현재 구현 상태 (2026-02-24 기준)

### 디렉토리 구조

```
python/
└── demo/
    ├── DESIGN.md
    ├── app.py                   ← 홈 페이지 (완료)
    ├── pages/
    │   ├── 1_성능_대시보드.py   ← ML/DL 단계별 R² 비교 (완료)
    │   ├── 2_예측_분석.py       ← Plotly 인터랙티브 차트 (완료)
    │   ├── 3_모델_정보.py       ← 아키텍처 요약 + 피처 목록 (완료)
    │   └── 4_라이브_추론.py     ← CSV 업로드 → 실시간 추론 (완료)
    └── utils/
        ├── __init__.py
        ├── constants.py         ← FINAL_R2, STAGE_R2, TARGET_LABELS (완료)
        ├── data_loader.py       ← CSV/PNG 로딩 헬퍼 (완료)
        ├── metrics.py           ← R², RMSE, MAE 계산 (완료)
        └── live_infer.py        ← 실시간 추론 유틸 (완료)
```

### 실행

```bash
cd python
streamlit run demo/app.py
```

---

## 4. 구현된 페이지 상세

---

### 4-1. `app.py` — 홈 (완료)

- `st.metric()` 4개: 모델 수(7개), 최고 R²(0.90), 데이터 해상도(30분), 예측 폭(12시간)
- 수평 bar chart (horizontal, R² 기준 우수/양호/보통 색상)
- `TARGET_LABELS`, `FINAL_R2`, `TARGET_ORDER` → `constants.py` 참조

---

### 4-2. `pages/1_성능_대시보드.py` — 모델 성능 (완료)

**6개 섹션**:

| 섹션 | 내용 |
|------|------|
| 1) 최종 R² 비교 | 7개 타겟 Viridis 컬러 바 차트 |
| 2) 개발 단계별 성능 변화 | ML_baseline → ML_v2 → DL → DL_Lag → DL_HP → DL_최종 꺾은선 |
| 3) ML FLOW baseline vs V2 | HistGBR·Lasso·Ridge·XGBoost·RF 5개 모델 그룹형 바 차트 |
| 4) ML TMS baseline vs V2 | 6개 TMS 타겟 R² 비교 (R²=0 기준선 표시) |
| 5) 데이터 사용률 개선 | FLOW/TMS baseline 4.2% → V2 98.4%/90.4% |
| 6) DL 타겟별 상세 | 탭별 학습 곡선 + 예측 분석 PNG (`st.image`) |

**STAGE_R2 구조** (`constants.py`):
- `ML_baseline`, `ML_v2`, `DL`, `DL_Lag 피처`, `DL_HP 최적화`, `DL_최종`

---

### 4-3. `pages/2_예측_분석.py` — 인터랙티브 예측 분석 (완료)

- selectbox: 타겟 선택
- 메트릭 4개: R², RMSE, MAE, 샘플 수
- 시계열 차트: 실측(파란선) vs 예측(빨간점선), rangeslider, hovermode=x unified
- 산점도: actual vs predicted, 오차 색상(RdBu), y=x 기준선
- 오차 분포 히스토그램 (nbins=50) + 평균/표준편차 caption

---

### 4-4. `pages/3_모델_정보.py` — 모델 아키텍처 (완료)

- 모델 구조 요약 테이블 (`src/config.py` 재사용)
- 타겟별 추천 피처 목록 selectbox
- LSTM 아키텍처 다이어그램 (ASCII, 타겟별 attention/head 동적 표시)

---

### 4-5. `pages/4_라이브_추론.py` — 라이브 추론 (완료)

- 실제 모델·스케일러 로컬 로드 (FastAPI 불필요)
- 입력 방식: CSV 업로드 또는 수동 입력 (48스텝)
- 템플릿 CSV 다운로드 + 실제 데이터 세그먼트 선택 (`live_infer_templates/real_segment/`)
- 12시간(30분 간격) autoregressive 예측 궤적 시각화

---

## 5. 추가 제안 — 신규 페이지

---

### 5-1. `pages/5_하이퍼파라미터_탐색.py` — HP 탐색 분석 ★ 추천

**목적**: 타겟별 하이퍼파라미터 그리드 탐색 과정과 인사이트를 시각화

**활용 가능한 기존 자산**: `results/DL/{target}_experiment_results.csv` (7개 파일)

#### 레이아웃

```
── 타겟 선택 ─────────────────────────────────────
  selectbox: [TN ▼]

── 탐색 개요 (metric 카드 4개) ──────────────────
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ 실험 조합 │ │  최고 R² │ │  최저 R² │ │  개선폭  │
  │  36개    │ │  0.9019  │ │  0.8920  │ │  +0.08%  │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘

── 파라미터별 R² 분포 ───────────────────────────
  (box plot 또는 violin plot)
  - x축: hidden_size / num_layers / lr 선택 라디오
  - y축: R²

── 상위 10개 조합 결과표 ──────────────────────
  ┌────────┬────────┬──────┬────────────┬────────┐
  │ hidden │ layers │  lr  │ dropout    │   R²   │
  ├────────┼────────┼──────┼────────────┼────────┤
  │  512   │   2    │ 2e-3 │    0.2     │ 0.9019 │
  │  ...   │  ...   │ ...  │    ...     │  ...   │
  └────────┴────────┴──────┴────────────┴────────┘

── 노트북 → 최적 파라미터 비교 (before/after) ──
  (하드코딩, NOTE.md 기반)
  ┌──────────────┬──────────┬──────────┐
  │ 파라미터     │  노트북  │  최적    │
  ├──────────────┼──────────┼──────────┤
  │ hidden_size  │   512    │   512    │
  │ num_layers   │    4     │    2     │
  │ lr           │  2e-4    │  2e-3    │
  │ R² 개선      │  0.9011  │  0.9019  │
  └──────────────┴──────────┴──────────┘

── 타겟 간 공통 인사이트 ──────────────────────
  (st.info 또는 expander)
  - lr=2e-3이 대부분 타겟에서 최적 (flux 제외)
  - attention 메커니즘: TP/TOC에서 성능 하락 유발
  - layers=1이 TP/TOC/PH에 적합, layers=2가 FLOW/SS/FLUX에 적합
```

#### 구현 포인트

- CSV 컬럼: `experiment_results.csv` 실제 컬럼명 확인 후 매핑
- 파라미터별 박스플롯: `px.box(df, x=selected_param, y="r2")`
- before/after 테이블: NOTE.md 기반 dict 하드코딩 (7개 타겟)
- `st.tabs([TARGET_LABELS[t] for t in TARGET_ORDER])` 탭 또는 selectbox

```python
HP_BEFORE_AFTER = {
    "flow": {"before": {"hidden":512, "layers":3, "lr":"2e-4", "r2":0.8166},
             "after":  {"hidden":512, "layers":2, "lr":"2e-3", "r2":0.8425}},
    "toc":  {"before": {"hidden":256, "layers":2, "lr":"1e-3", "r2":0.4731},
             "after":  {"hidden":384, "layers":1, "lr":"1e-3", "r2":0.5574}},
    "ss":   {"before": {"hidden":512, "layers":4, "lr":"2e-4", "r2":0.6712},
             "after":  {"hidden":256, "layers":2, "lr":"2e-3", "r2":0.6906}},
    "tn":   {"before": {"hidden":512, "layers":4, "lr":"2e-4", "r2":0.9011},
             "after":  {"hidden":512, "layers":2, "lr":"2e-3", "r2":0.9019}},
    "tp":   {"before": {"hidden":512, "layers":4, "lr":"2e-4", "r2":0.6252},
             "after":  {"hidden":384, "layers":1, "lr":"1e-3", "r2":0.6378}},
    "flux": {"before": {"hidden":512, "layers":4, "lr":"2e-4", "r2":0.6296},
             "after":  {"hidden":256, "layers":2, "lr":"5e-4", "r2":0.6241}},
    "ph":   {"before": {"hidden":512, "layers":4, "lr":"2e-4", "r2":0.8432},
             "after":  {"hidden":512, "layers":1, "lr":"2e-3", "r2":0.8574}},
}
```

---

### 5-2. `pages/5_이상_진단.py` (또는 6번) — 이상 탐지

**목적**: Isolation Forest 기반 이상 진단 결과 시각화

**활용 가능한 기존 자산**: `results/DL/save/{target}_diagnosis.png` (7개 파일)

#### 레이아웃

```
── 섹션 1: 이상 탐지 개요 ────────────────────
  Isolation Forest 방법론 설명 (st.info)

── 섹션 2: 타겟별 이상 진단 탭 ──────────────
  탭: [Flow] [TOC] [SS] [TN] [TP] [FLUX] [PH]

  선택 시:
  ┌───────────────────────────────────────────┐
  │  diagnosis.png                            │
  │  (실측값 시계열 + 이상 탐지 점수 표시)    │
  └───────────────────────────────────────────┘
  - 이상치 비율, 감지 기준 텍스트 설명

── 섹션 3: 도메인 기준 경보 임계값 ──────────
  (하드코딩 또는 사용자 입력)
  ┌──────┬────────────┬────────────┐
  │ 지표 │ 배출허용기준│ 경보 임계값│
  ├──────┼────────────┼────────────┤
  │ TOC  │  15 mg/L  │  30 mg/L   │
  │ TN   │  20 mg/L  │  40 mg/L   │
  │ TP   │  0.2 mg/L │  0.4 mg/L  │
  └──────┴────────────┴────────────┘
```

#### 구현 포인트

- `st.tabs()` + `st.image(str(diag_path))`
- `diagnosis.png` 파일이 있으면 표시, 없으면 `st.warning`
- 도메인 기준 테이블은 하드코딩 (README.md 기준 참고)

---

### 5-3. LSTM vs Transformer 비교 섹션 추가 (1_성능_대시보드.py)

**목적**: Transformer 실험을 별도 페이지 대신 성능_대시보드 내 섹션으로 추가

**활용**: NOTE.md 기반 하드코딩 (Transformer 성능은 저장 파일 없음)

```
── 7) LSTM vs Transformer 비교 ──────────────
  (NOTE.md 기준 실험 결과 하드코딩)
  설명: "전 타겟에서 LSTM이 Transformer보다 우수 → LSTM 선택"

  ┌────────┬───────┬─────────────┐
  │ 타겟   │ LSTM  │ Transformer │
  ├────────┼───────┼─────────────┤
  │ TOC    │ 0.56  │ (실험 결과) │
  │ SS     │ 0.69  │ ...         │
  │ TN     │ 0.90  │ ...         │
  └────────┴───────┴─────────────┘
  주석: Transformer 실험 결과는 notebook/DL/transformer_TMS.ipynb 기준
```

#### 구현 포인트

- TRANSFORMER_R2 dict 하드코딩 (NOTE.md에서 발췌)
- `px.bar(barmode="group")` LSTM vs Transformer 나란히

---

## 6. 추가 제안 — 기존 페이지 개선

---

### 6-1. `app.py` — 홈 개선

**현재**: "데이터 수집 → 전처리 → LSTM 예측 → API 서비스" 텍스트

**개선**: Plotly Sankey 다이어그램 또는 단계별 st.columns 카드로 교체

```python
# 시스템 아키텍처 카드형 표시
steps = [
    ("📥 데이터 수집", "1분 단위, 24시간\n유입량+TMS+기상 3개소"),
    ("⚙️ 전처리", "30분 리샘플링\n피처 엔지니어링"),
    ("🧠 LSTM 추론", "Autoregressive\n12시간 예측"),
    ("📡 API 서비스", "FastAPI\n/predict/flow, /predict/tms"),
    ("📊 대시보드", "Streamlit\n인터랙티브 시각화"),
]
cols = st.columns(len(steps))
for col, (icon_title, desc) in zip(cols, steps):
    col.info(f"**{icon_title}**\n\n{desc}")
```

---

### 6-2. `pages/2_예측_분석.py` — 예측 분석 개선

**현재 미구현 지표**: MAPE

**개선 1**: 메트릭 5개로 확장 (R², RMSE, MAE, **MAPE**, **5% 이내 비율**)

```python
# README.md 정의: |실제값 - 예측값| / 실제값 ≤ 5% 비율을 정확도로 정의
mape = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100
within_5pct = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan)) <= 0.05) * 100
```

**개선 2**: 오차 상위 구간 하이라이팅

```
── 오차 상위 5% 구간 ────────────────────────
  [시계열 차트에 오차 큰 구간 음영 표시]
  - 실측-예측 오차 절대값 기준 상위 5% 구간에 빨간 배경
```

**개선 3**: 오차 자동 상관 분석 (Autocorrelation Plot)

---

### 6-3. `pages/3_모델_정보.py` — 모델 정보 개선

**추가**: 피처 엔지니어링 파이프라인 단계 시각화

```
── 피처 엔지니어링 파이프라인 ───────────────
  (순서도 텍스트 블록)
  1. add_target_lag_features (lag/rolling/diff/EWMA)
  2. 타겟 원본 컬럼 제거
  3. add_rain_features
  4. add_station_agg_rain_features
  5. add_weather_features
  6. add_process_features
  7. add_temporal_features
  8. add_time_features
  9. ffill → fillna(0)
```

---

## 7. 공통 유틸리티 (현재 구현)

### `utils/constants.py`

```python
TARGET_ORDER    # list[str] — 7개 타겟 순서
TARGET_LABELS   # dict — 한글 표시명
FINAL_R2        # dict — 최종 R² (constants.py 기준)
STAGE_R2        # dict — 단계별 R² (ML_baseline/ML_v2/DL/DL_Lag 피처/DL_HP 최적화/DL_최종)
```

### `utils/data_loader.py`

```python
load_predictions(target)   → pd.DataFrame (actual, predicted)
load_feature_names(target) → list[str]
get_png_path(kind, target) → Path
  # kind: "prediction" | "learning_curve" | "diagnosis"
  # results/DL/save/ 경로 반환
```

### `utils/metrics.py`

```python
compute_metrics(actual, predicted) → dict
    # {"r2": ..., "rmse": ..., "mae": ...}
```

### `utils/live_infer.py`

```python
load_runtime_artifacts(target)           → dict (model, scalers, feature_names)
run_inference(df, artifacts, horizon)    → (one_step, list[float])
validate_and_align_input(df, feat_names) → pd.DataFrame
build_sequence_from_single_row(...)      → pd.DataFrame
build_trajectory_df(preds)              → pd.DataFrame
```

---

## 8. 데이터 상수 (하드코딩)

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

### 최종 R² 성능 (현재 저장 모델 기준)

```python
FINAL_R2 = {
    "flow": 0.8425,
    "toc":  0.5574,
    "ss":   0.6906,
    "tn":   0.9011,
    "tp":   0.6201,
    "flux": 0.6241,
    "ph":   0.8574,
}
```

### 단계별 성능 (개선 과정 스토리)

```python
STAGE_R2 = {
    "ML_baseline": {"flow": 0.57, "toc": -0.66, "ss": -1.00, "tn": -0.79, "tp": -6.72, "flux": 0.96, "ph": 0.39},
    "ML_v2":       {"flow": 0.72, "toc": -0.09, "ss": -2.12, "tn": -0.25, "tp": -5.09, "flux": 0.00, "ph": -0.26},
    "DL":          {"flow": 0.30, "toc": -1.86, "ss": -0.52, "tn": -0.16, "tp": -2.15, "flux": -0.01, "ph": -0.17},
    "DL_Lag 피처": {"flow": 0.79, "toc":  0.29, "ss":  0.21, "tn":  0.78, "tp": -0.41, "flux":  0.23, "ph":  0.56},
    "DL_HP 최적화":{"flow": 0.82, "toc":  0.47, "ss":  0.67, "tn":  0.90, "tp":  0.63, "flux":  0.61, "ph":  0.84},
    "DL_최종":     {"flow": 0.84, "toc":  0.55, "ss":  0.69, "tn":  0.90, "tp":  0.62, "flux":  0.62, "ph":  0.85},
}
```

---

## 9. 기술 스택

| 역할 | 패키지 | 비고 |
|------|--------|------|
| UI 프레임워크 | `streamlit` | 멀티페이지 구조 |
| 인터랙티브 차트 | `plotly` | `plotly.express` + `plotly.graph_objects` |
| 데이터 처리 | `pandas`, `numpy` | |
| 메트릭 계산 | `scikit-learn` | `r2_score`, `mean_squared_error`, `mean_absolute_error` |
| 모델 코드 재사용 | `src/config.py`, `src/models.py` | 중복 없이 import |

```bash
pip install streamlit plotly
# pandas, numpy, scikit-learn, torch는 이미 설치되어 있음
```

---

## 10. 구현 우선순위

| 순서 | 작업 | 자산 | 난이도 | 포트폴리오 가치 |
|------|------|------|--------|----------------|
| 1 | **5_하이퍼파라미터_탐색.py** 신규 | `experiment_results.csv` (7개) | 중간 | ★★★★★ |
| 2 | **2_예측_분석.py** MAPE·5%이내 비율 추가 | 기존 CSV | 낮음 | ★★★★ |
| 3 | **1_성능_대시보드.py** Transformer 비교 섹션 추가 | 하드코딩 | 낮음 | ★★★ |
| 4 | **5_이상_진단.py** 신규 | `diagnosis.png` (7개) | 낮음 | ★★★ |
| 5 | **app.py** 시스템 아키텍처 카드형 개선 | 코드만 | 낮음 | ★★ |
| 6 | **3_모델_정보.py** 피처 파이프라인 단계 추가 | 코드만 | 낮음 | ★★ |

---

## 11. 주의사항

### 경로 처리

- `demo/app.py`에서 `src/`를 import하려면 `sys.path`에 `python/` 루트 추가 필요
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  ```
- 상대 경로 대신 `Path(__file__).resolve().parents[N]` 기준 절대 경로 사용

### PNG 경로 규칙

- TMS 6개: `results/DL/save/prediction_analysis_{target}.png`
- Flow 예측분석: `results/DL/save/prediction_analysis_flow.png` (save 포함)
- 이상 진단: `results/DL/save/{target}_diagnosis.png`
- 학습 곡선: `results/DL/save/{target}_learning_curve.png`

### 피처 CSV 경로

- `data/recommand_features/save/` (save 서브폴더 있음)
- 로딩 시 `save/` 경로 우선 사용

### 라이브 추론 템플릿

- 기본 템플릿: `demo/live_infer_templates/{target}_live_infer_template.csv` (48행 × n_features, 0값)
- 실 데이터 세그먼트: `demo/live_infer_templates/real_segment/{target}_live_infer_template_real_20250928.csv`

---

*초안: 2026-02-23 | 최종 업데이트: 2026-02-24*
