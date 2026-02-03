# src/ML 실행 순서 및 모듈 설명

## 📋 목차
1. [전체 실행 흐름](#전체-실행-흐름)
2. [모듈별 상세 설명](#모듈별-상세-설명)
3. [파이프라인 종류](#파이프라인-종류)
4. [실행 예시](#실행-예시)

---

## 🔄 전체 실행 흐름

### 기본 파이프라인 (`run_pipeline`)

```
1. io.py
   ├─ load_csvs()           # CSV 파일 로드
   ├─ prep_flow()           # FLOW 데이터 전처리
   ├─ prep_aws()            # AWS 데이터 전처리
   └─ set_datetime_index()  # 시간 인덱스 설정
   
2. io.py
   └─ merge_sources_on_time()  # 데이터 병합
   
3. preprocess.py
   ├─ resample_hourly()         # 리샘플링 (1시간 단위)
   ├─ impute_missing()          # 결측치 처리
   └─ detect_and_handle_outliers()  # 이상치 처리
   
4. features.py
   ├─ build_features()          # 피처 생성
   │  ├─ add_time_features()    # 시간 특성
   │  ├─ add_lag_features()     # Lag 특성
   │  ├─ add_rolling_features() # Rolling 통계
   │  └─ add_model_specific_features()  # 모델별 도메인 특성
   └─ make_supervised_dataset() # X, y 분리
   
5. split.py
   └─ time_split()              # 시계열 데이터 분할 (train/valid/test)
   
6. models.py
   └─ build_model_zoo()         # 여러 ML 모델 생성
      ├─ LinearRegression
      ├─ Ridge
      ├─ Lasso
      ├─ RandomForest
      ├─ GradientBoosting
      ├─ XGBoost
      └─ HistGradientBoosting
   
7. metrics.py
   ├─ fit_and_evaluate()        # 모델 학습 및 평가
   ├─ compute_metrics()         # 성능 지표 계산
   └─ plot_metric_table()       # 결과 시각화
```

### 개선 파이프라인 (`run_improved_pipeline`)

```
1~4. [기본 파이프라인과 동일]

5. feature_selection.py
   └─ select_top_features()     # 피처 선택 (상위 N개)
      └─ SelectKBest (mutual_info_regression)
   
6. scaling.py
   └─ scale_data()              # 데이터 스케일링
      └─ StandardScaler
   
7. split.py
   └─ time_split()              # 데이터 분할
   
8. models.py
   └─ build_model_zoo_with_optuna()  # Optuna 하이퍼파라미터 튜닝
      ├─ XGBoost (최적화)
      └─ HistGradientBoosting (최적화)
   
9. metrics.py
   ├─ fit_and_evaluate()        # 모델 학습 및 평가
   └─ compute_metrics()         # 성능 지표 계산
   
10. save_results.py
    ├─ save_metrics()           # 지표 저장
    ├─ save_predictions()       # 예측값 저장
    └─ save_model()             # 모델 저장
    
11. visualization.py
    ├─ plot_learning_curve()    # 학습 곡선
    └─ plot_r2_comparison()     # R² 비교 차트
```

### Sliding Window 파이프라인 (`run_sliding_window_pipeline`)

```
1~4. [기본 파이프라인과 동일]

5. sliding_window.py
   └─ create_sliding_windows()  # 슬라이딩 윈도우 생성
      # (window_size, horizon, stride)
   
6. feature_selection.py
   └─ select_top_features()     # 피처 선택
   
7. scaling.py
   └─ scale_data()              # 스케일링
   
8~11. [개선 파이프라인과 동일]
```

---

## 📦 모듈별 상세 설명

### 1. **io.py** - 데이터 입출력
**역할**: CSV 파일 로드 및 기본 전처리

**주요 함수**:
- `load_csvs(data_root)`: 데이터 디렉토리에서 CSV 파일 로드
- `prep_flow(df)`: FLOW 데이터 전처리 (Q_in 생성)
- `prep_aws(df1, df2, df3)`: AWS 데이터 병합 및 전처리
- `set_datetime_index(df, time_col)`: 시간 인덱스 설정
- `merge_sources_on_time(dfs, how)`: 여러 데이터 소스 병합

**입력**: CSV 파일 경로
**출력**: DataFrame 딕셔너리

---

### 2. **preprocess.py** - 데이터 전처리
**역할**: 결측치, 이상치 처리 및 리샘플링

**주요 함수**:
- `resample_hourly(df, rule, agg)`: 시간 단위 리샘플링
- `impute_missing_with_strategy(df, config)`: 결측치 처리
  - 단기: Forward Fill
  - 중기: EWMA
  - 장기: Rolling Median 또는 NaN 유지
- `detect_and_handle_outliers(df, config)`: 이상치 탐지 및 처리
  - 도메인 지식 기반
  - 통계 기반 (IQR, Z-score)

**입력**: 병합된 DataFrame
**출력**: 전처리된 DataFrame

---

### 3. **features.py** - 피처 엔지니어링
**역할**: 시간, Lag, Rolling, 도메인 특화 피처 생성

**주요 함수**:
- `build_features(df, target_cols, mode, cfg)`: 전체 피처 생성 파이프라인
- `add_time_features(df)`: 시간 특성 (hour, dayofweek, season 등)
- `add_lag_features(df, cols, lags)`: Lag 특성 (1, 2, 3, 6, 12, 24시간)
- `add_rolling_features(df, cols, windows)`: Rolling 통계 (mean, std, min, max)
- `add_model_specific_features(df, mode)`: 모델별 도메인 특성
  - **FLOW**: 수위-유량, 강우 공간 통합
  - **ModelA/B/C**: TMS 상호작용, 강수-TMS 상호작용
- `make_supervised_dataset(df, target_cols)`: X, y 분리

**입력**: 전처리된 DataFrame
**출력**: 피처가 추가된 DataFrame, X, y

---

### 4. **split.py** - 데이터 분할
**역할**: 시계열 데이터를 train/valid/test로 분할

**주요 함수**:
- `time_split(X, y, config)`: 시간 순서 유지하며 분할
  - Train: 70%
  - Valid: 15%
  - Test: 15%

**입력**: X, y
**출력**: (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

---

### 5. **feature_selection.py** - 피처 선택
**역할**: 중요한 피처만 선택 (차원 축소)

**주요 함수**:
- `select_top_features(X_train, y_train, n_features)`: 상위 N개 피처 선택
  - SelectKBest + mutual_info_regression

**입력**: X_train, y_train, n_features
**출력**: 선택된 피처 이름 리스트

---

### 6. **scaling.py** - 데이터 스케일링
**역할**: 피처 스케일 정규화

**주요 함수**:
- `scale_data(X_train, X_valid, X_test)`: StandardScaler 적용

**입력**: 분할된 데이터
**출력**: 스케일링된 데이터 + Scaler 객체

---

### 7. **models.py** - 모델 생성
**역할**: ML 모델 생성 및 하이퍼파라미터 튜닝

**주요 함수**:
- `build_model_zoo()`: 기본 모델 세트 생성
  - LinearRegression, Ridge, Lasso
  - RandomForest, GradientBoosting
  - XGBoost, HistGradientBoosting
  
- `build_model_zoo_with_optuna()`: Optuna로 최적화된 모델 생성
  - XGBoost (튜닝)
  - HistGradientBoosting (튜닝)

**입력**: 학습 데이터, 설정
**출력**: 모델 딕셔너리

---

### 8. **metrics.py** - 모델 평가
**역할**: 모델 학습 및 성능 평가

**주요 함수**:
- `fit_and_evaluate(models, splits)`: 모든 모델 학습 및 평가
- `compute_metrics(y_true, y_pred)`: 성능 지표 계산
  - MSE, RMSE, MAE, R², MAPE
- `plot_metric_table(metric_table)`: 결과 테이블 시각화

**입력**: 모델, 데이터 분할
**출력**: 성능 지표 테이블, 학습된 모델

---

### 9. **sliding_window.py** - 슬라이딩 윈도우
**역할**: 시계열 패턴 학습을 위한 윈도우 생성

**주요 함수**:
- `create_sliding_windows(X, y, window_size, horizon, stride)`: 윈도우 생성
  - window_size: 과거 몇 시간 볼 것인지
  - horizon: 미래 몇 시간 후 예측
  - stride: 윈도우 이동 간격

**입력**: X, y, 윈도우 설정
**출력**: X_seq (3D), y_seq (2D)

---

### 10. **save_results.py** - 결과 저장
**역할**: 모델, 예측값, 지표 저장

**주요 함수**:
- `save_metrics(metrics, path)`: 성능 지표 저장 (JSON)
- `save_predictions(y_true, y_pred, path)`: 예측값 저장 (CSV)
- `save_model(model, path)`: 모델 저장 (pickle)

**입력**: 결과 데이터, 저장 경로
**출력**: 파일 저장

---

### 11. **visualization.py** - 시각화
**역할**: 학습 결과 시각화

**주요 함수**:
- `plot_learning_curve(model, X, y)`: 학습 곡선
- `plot_r2_comparison(metric_table)`: R² 비교 차트
- `plot_predictions(y_true, y_pred)`: 예측 vs 실제 비교

**입력**: 모델, 데이터, 지표
**출력**: 그래프 (matplotlib)

---

## 🚀 파이프라인 종류

### 1. 기본 파이프라인
**특징**: 여러 ML 모델 비교, 빠른 실행
**사용 시기**: 초기 탐색, 빠른 프로토타이핑

```python
from src.ML.pipeline import run_pipeline

out = run_pipeline(
    dfs,
    mode="flow",
    time_col_map={"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"},
    resample_rule="1h",
    resample_agg="mean",
    random_state=42
)
```

### 2. 개선 파이프라인
**특징**: Optuna 튜닝, 피처 선택, 스케일링
**사용 시기**: 최종 모델 학습, 성능 최적화

```python
from src.ML.pipeline import run_improved_pipeline

out = run_improved_pipeline(
    dfs,
    mode="flow",
    time_col_map={"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"},
    resample_rule="1h",
    n_top_features=50,
    cv_splits=3,
    n_trials=50,
    save_dir="results/ML"
)
```

### 3. Sliding Window 파이프라인
**특징**: 시계열 패턴 학습, LSTM과 유사한 입력 구조
**사용 시기**: 시간적 의존성이 중요한 경우

```python
from src.ML.pipeline import run_sliding_window_pipeline

out = run_sliding_window_pipeline(
    dfs,
    mode="flow",
    window_size=24,  # 24시간 과거 데이터
    horizon=1,       # 1시간 후 예측
    stride=1,        # 매 시간마다
    n_top_features=50,
    save_dir="results/ML"
)
```

---

## 💡 실행 예시

### CLI 실행
```bash
# 기본 파이프라인
python scripts/ML/train.py --mode flow --data-root data/actual --resample 1h

# 개선 파이프라인
python scripts/ML/train.py --mode flow --improved --n-features 50 --n-trials 50

# Sliding Window 파이프라인
python scripts/ML/train.py --mode flow --sliding-window --window-size 24 --horizon 1
```

### 노트북 실행
```python
# notebook/ML/train_ml_models.ipynb 참고
PIPELINE_TYPE = "improved"  # "basic", "improved", "sliding_window"
MODE = "flow"
RESAMPLE = "1h"
```

---

## 📊 모드별 타겟 및 입력 데이터

| 모드 | 타겟 | 입력 데이터 |
|------|------|-------------|
| **flow** | Q_in | FLOW (level) + AWS (강수, 기상) |
| **tms** | 전체 6개 TMS | FLOW + AWS + TMS |
| **modelA** | TOC_VU, SS_VU | FLOW + AWS + TMS (PH, FLUX, TN, TP) |
| **modelB** | TN_VU, TP_VU | FLOW + AWS + TMS (PH, FLUX, TOC, SS) |
| **modelC** | FLUX_VU, PH_VU | FLOW + AWS + TMS (TOC, SS, TN, TP) |

---

## 🔍 디버깅 팁

1. **데이터 로드 실패**: `io.py`의 `load_csvs()` 확인
2. **결측치 많음**: `preprocess.py`의 `impute_missing()` 설정 조정
3. **성능 낮음**: `features.py`의 도메인 특성 추가 또는 `feature_selection.py`로 피처 선택
4. **과적합**: `models.py`의 하이퍼파라미터 조정 또는 정규화 강화
5. **학습 느림**: 기본 파이프라인 사용 또는 `n_trials` 감소

---

## 📝 참고 자료

- 각 모듈의 docstring 참고
- `scripts/ML/train.py`: CLI 실행 예시
- `notebook/ML/train_ml_models.ipynb`: 노트북 실행 예시
