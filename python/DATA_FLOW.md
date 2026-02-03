# scripts/train.py 데이터 처리 및 예측 진행 순서

## 📋 개요

`scripts/train.py`는 3가지 파이프라인을 제공합니다:
1. **기본 파이프라인** (`run_pipeline`)
2. **개선 파이프라인** (`run_improved_pipeline`) - Optuna, 피처 선택, Scaling
3. **Sliding Window 파이프라인** (`run_sliding_window_pipeline`) - 시계열 윈도우 기반

---

## 🔄 공통 초기 단계 (train.py)

### 1단계: CLI 인자 파싱
```bash
python scripts/train.py --mode flow --improved --n-features 50
```

**주요 인자:**
- `--mode`: 예측 모드 (flow/tms/modelA/modelB/modelC)
- `--improved`: 개선 파이프라인 사용
- `--sliding-window`: Sliding Window 파이프라인 사용
- `--resample`: 리샘플링 규칙 (기본: 1h)
- `--n-features`: 선택할 피처 개수 (기본: 50)
- `--cv-splits`: 교차 검증 분할 수 (기본: 3)
- `--n-trials`: Optuna 시도 횟수 (기본: 50)

### 2단계: 데이터 로드 (src/io.py)
```python
# CSV 파일 로드
df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569 = load_csvs(data_root)

# 전처리
df_flow = prep_flow(df_flow)      # FLOW 데이터 정리
df_aws = prep_aws(...)             # AWS 기상 데이터 병합

dfs = {"flow": df_flow, "tms": df_tms, "aws": df_aws}
```

**로드되는 데이터:**
- `FLOW_Actual.csv`: 유입 유량 데이터 (Q_in, flow_TankA/B, level_TankA/B)
- `TMS_Actual.csv`: 수질 데이터 (TOC, PH, SS, FLUX, TN, TP)
- `AWS_368.csv`, `AWS_541.csv`, `AWS_569.csv`: 기상 데이터 (온도, 습도, 강수량 등)

### 3단계: 분할 설정
```python
split_cfg = SplitConfig(
    train_ratio=0.6,   # 60% 학습
    valid_ratio=0.2,   # 20% 검증
    test_ratio=0.2     # 20% 테스트
)
```

---

## 📊 파이프라인별 상세 처리 순서

## 1️⃣ 기본 파이프라인 (run_pipeline)

### 전처리 순서
```
원본 데이터
    ↓
[1단계] 시간축 정합
    ↓
[2단계] 결측치 보간 (1차)
    ↓
[3단계] 이상치 처리
    ↓
[2단계] 결측치 재보간 (2차)
    ↓
[4단계] 리샘플링
    ↓
[5단계] 파생 특성 생성
    ↓
[6단계] Train/Valid/Test 분리
    ↓
[7단계] 모델 학습 및 평가
```

### 상세 설명

#### [1단계] 시간축 정합 (src/io.py)
```python
# 각 데이터프레임을 DatetimeIndex로 변환
dfs_indexed = {}
for name, df in dfs.items():
    if not isinstance(df.index, pd.DatetimeIndex):
        df = set_datetime_index(df, time_col=time_col_map[name])
    df = df.sort_index()  # 시간순 정렬
    df = df[~df.index.duplicated(keep='first')]  # 중복 제거
    dfs_indexed[name] = df

# 데이터 병합 (outer join)
df_all = merge_sources_on_time(dfs_indexed, how="outer")
```

**결과:** 모든 데이터가 하나의 DataFrame으로 병합됨

#### [2단계] 결측치 보간 (src/preprocess.py)
```python
# ImputationConfig 설정
config = ImputationConfig(
    method="ffill",           # Forward Fill
    ewma_span=12,            # EWMA 윈도우 크기
    max_gap_hours=6,         # 최대 보간 간격
    use_ewma_for_long_gaps=True
)

df_all = impute_missing_with_strategy(df_all, freq="1h", config=config)
```

**전략:**
- 짧은 결측 (≤6시간): Forward Fill
- 긴 결측 (>6시간): EWMA (Exponential Weighted Moving Average)
- 보간 마스크 추가 (`_imputed` 컬럼)

#### [3단계] 이상치 처리 (src/preprocess.py)
```python
# OutlierConfig 설정
config = OutlierConfig(
    method="iqr",            # IQR 방법
    iqr_multiplier=3.0,      # IQR * 3.0
    clip=False               # 이상치를 NaN으로 변환
)

df_all = detect_and_handle_outliers(df_all, config=config)
```

**방법:**
- IQR (Interquartile Range) 기반 탐지
- 이상치 → NaN 변환 → 재보간
- 이상치 마스크 추가 (`_outlier` 컬럼)

#### [4단계] 리샘플링 (src/preprocess.py)
```python
df_hourly = resample_hourly(df_all, rule="1h", agg="mean")
```

**목적:** 시간 간격 통일 (예: 10분 → 1시간)

#### [5단계] 파생 특성 생성 (src/features.py)
```python
# 모드에 따라 제외할 컬럼 결정
exclude_cols = get_exclude_features(mode, target_cols)

df_feat = build_features(
    df_hourly=df_hourly,
    target_cols=target_cols,
    exclude_cols=exclude_cols,
    cfg=feature_cfg
)
```

**생성되는 특성:**

1. **Rolling 통계 (이동 평균/표준편차)**
   - 윈도우: 3, 6, 12, 24시간
   - 예: `TA_368_roll_mean_3h`, `RN_368_roll_std_6h`

2. **Lag 특성 (과거 값)**
   - 시차: 1, 3, 6, 12, 24시간
   - 예: `TA_368_lag_1h`, `HM_368_lag_6h`

3. **시간 특성**
   - `hour`: 시간 (0-23)
   - `day_of_week`: 요일 (0-6)
   - `month`: 월 (1-12)
   - `is_weekend`: 주말 여부 (0/1)

4. **차분 특성**
   - 1시간 차분: `TA_368_diff_1h`

**데이터 누수 방지:**
- FLOW 모드: TMS 데이터 전체 제외 (미래 정보)
- TMS 모드: FLOW 데이터 제외 (미래 정보)
- ModelA/B/C: 예측 대상만 제외, 나머지 TMS는 입력으로 사용

#### [6단계] 지도학습 데이터셋 생성 (src/features.py)
```python
X, y = make_supervised_dataset(
    df_feat, 
    target_cols=target_cols,
    exclude_cols=exclude_cols,
    dropna=True  # NaN 행 제거
)
```

**결과:**
- `X`: 입력 특성 (예: 200개 피처)
- `y`: 타겟 변수 (예: Q_in)

#### [7단계] 데이터 분할 (src/split.py)
```python
splits = time_split(X, y, cfg=split_cfg)

X_train, y_train = splits["train"]  # 60%
X_valid, y_valid = splits["valid"]  # 20%
X_test, y_test = splits["test"]     # 20%
```

**시계열 분할:** 시간 순서 유지 (과거 → 미래)

#### [8단계] 모델 학습 (src/models.py)
```python
# 모델 Zoo 생성
zoo = build_model_zoo(random_state=42)
# 포함 모델: RandomForest, XGBoost, HistGradientBoosting

# 각 모델 학습 및 평가
for model_name, model in zoo.items():
    model = wrap_multioutput_if_needed(model, y)  # 다중 타겟 처리
    model.fit(X_train, y_train)
    
    # 예측
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    y_pred_test = model.predict(X_test)
    
    # 평가
    metrics = compute_metrics(y_test, y_pred_test)
    # R², RMSE, MAE, MAPE
```

---

## 2️⃣ 개선 파이프라인 (run_improved_pipeline)

### 전처리 순서
```
원본 데이터
    ↓
[1-5단계] 기본 파이프라인과 동일
    ↓
[6단계] Train/Valid/Test 분리
    ↓
[7단계] 스케일링 (Train 기준) ⭐ 추가
    ↓
[8단계] 피처 선택 (Train 기준) ⭐ 추가
    ↓
[9단계] Optuna 하이퍼파라미터 최적화 ⭐ 추가
    ↓
모델 학습 및 평가
```

### 추가 단계 상세 설명

#### [7단계] 스케일링 (src/scaling.py)
```python
from sklearn.preprocessing import StandardScaler

X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_data(
    X_train, X_valid, X_test
)

# StandardScaler: (X - mean) / std
# Train 데이터로 fit, Valid/Test는 transform만
```

**목적:** 특성 스케일 통일 (평균 0, 표준편차 1)

**데이터 누수 방지:**
- Train 데이터로만 scaler fit
- Valid/Test는 Train의 통계량 사용

#### [8단계] 피처 선택 (src/feature_selection.py)
```python
top_features = select_top_features(
    X_train_scaled, 
    y_train,
    n_features=50,  # 상위 50개 선택
    random_state=42
)

X_train_selected = X_train_scaled[top_features]
X_valid_selected = X_valid_scaled[top_features]
X_test_selected = X_test_scaled[top_features]
```

**방법:**
1. RandomForest로 feature importance 계산
2. 중요도 상위 N개 선택

**데이터 누수 방지:**
- Train 데이터로만 중요도 계산
- Valid/Test는 선택된 피처만 사용

#### [9단계] Optuna 하이퍼파라미터 최적화 (src/models.py)
```python
# Optuna 래퍼 모델 생성
zoo = build_model_zoo_with_optuna(
    cv_splits=3,      # TimeSeriesSplit
    n_trials=50,      # 50번 시도
    random_state=42
)

# 각 모델별 최적화
for model_name, optuna_model in zoo.items():
    # Optuna가 자동으로 하이퍼파라미터 탐색
    optuna_model.fit(X_train_selected, y_train)
    
    # 최적 파라미터로 재학습
    best_params = optuna_model.best_params_
    
    # XGBoost는 Early Stopping 추가
    if model_name == "XGBoost":
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(
            X_train_selected, y_train,
            eval_set=[(X_valid_selected, y_valid)],
            early_stopping_rounds=20,
            verbose=False
        )
```

**최적화 대상 파라미터:**

**XGBoost:**
- `max_depth`: 3-10
- `learning_rate`: 0.01-0.3
- `n_estimators`: 100-1000
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0

**HistGradientBoosting:**
- `max_depth`: 3-15
- `learning_rate`: 0.01-0.3
- `max_iter`: 100-500

**RandomForest:**
- `n_estimators`: 100-500
- `max_depth`: 10-50
- `min_samples_split`: 2-20

---

## 3️⃣ Sliding Window 파이프라인 (run_sliding_window_pipeline)

### 전처리 순서
```
원본 데이터
    ↓
[1-5단계] 기본 파이프라인과 동일
    ↓
[6단계] Sliding Window 생성 ⭐ 추가
    ↓
[7단계] 윈도우 단위 데이터 분할
    ↓
[8단계] 윈도우 평탄화 (2D 변환)
    ↓
[9단계] 스케일링
    ↓
[10단계] 피처 선택
    ↓
모델 학습 및 평가
```

### 추가 단계 상세 설명

#### [6단계] Sliding Window 생성 (src/sliding_window.py)
```python
X_seq, y_seq = create_sliding_windows(
    X, y,
    window_size=24,  # 과거 24시간
    horizon=1,       # 1시간 후 예측
    stride=1         # 1시간씩 이동
)
```

**변환:**
```
원본 데이터 (2D):
X: (10000 샘플, 200 피처)
y: (10000 샘플, 1 타겟)

↓ Sliding Window

3D 시퀀스 데이터:
X_seq: (9975 윈도우, 24 시간, 200 피처)
y_seq: (9975 윈도우, 1 타겟)
```

**예시:**
```
윈도우 1: [시간 0-23] → 시간 24 예측
윈도우 2: [시간 1-24] → 시간 25 예측
윈도우 3: [시간 2-25] → 시간 26 예측
...
```

#### [8단계] 윈도우 평탄화 (src/sliding_window.py)
```python
# 3D → 2D 변환 (ML 모델용)
X_train_flat = flatten_windows_for_ml(X_train_seq)

# (9975, 24, 200) → (9975, 4800)
# 24시간 * 200피처 = 4800 피처
```

**특성 이름 생성:**
```python
feature_names = create_feature_names_for_flattened_windows(
    original_features, 
    window_size=24
)

# 예: ['TA_368_t-23', 'TA_368_t-22', ..., 'TA_368_t-0']
```

---

## 📈 모델 평가 지표

### 계산되는 지표 (src/metrics.py)
```python
metrics = compute_metrics(y_true, y_pred)

# 반환값:
{
    "R2_mean": 0.85,           # 결정계수 (높을수록 좋음)
    "RMSE_mean": 12.5,         # 평균 제곱근 오차 (낮을수록 좋음)
    "MAE_mean": 8.3,           # 평균 절대 오차 (낮을수록 좋음)
    "MAPE_mean(%)": 5.2        # 평균 절대 백분율 오차 (낮을수록 좋음)
}
```

### 지표 의미
- **R² (결정계수)**: 모델이 데이터를 얼마나 잘 설명하는가 (0-1, 1이 완벽)
- **RMSE**: 예측 오차의 크기 (타겟과 같은 단위)
- **MAE**: 절대 오차의 평균 (이상치에 덜 민감)
- **MAPE**: 백분율 오차 (상대적 성능 평가)

---

## 💾 결과 저장

### 저장되는 파일 (src/save_results.py)
```
results/ML/
├── predictions/
│   ├── {mode}_train_predictions.csv      # 학습 데이터 예측값
│   ├── {mode}_valid_predictions.csv      # 검증 데이터 예측값
│   └── {mode}_test_predictions.csv       # 테스트 데이터 예측값
├── sequences/                             # Sliding Window만
│   ├── {mode}_X_seq.npz                  # 3D 시퀀스 데이터
│   └── {mode}_y_seq.npz
├── models/
│   ├── {model_name}_{mode}.pkl           # 학습된 모델
│   └── scaler_{mode}.pkl                 # 스케일러
├── {mode}_r2_comparison.png              # R² 비교 그래프
├── {mode}_{model}_learning_curve.png     # 학습 곡선
└── analysis_report.md                     # 분석 보고서
```

---

## 🎯 모드별 예측 대상 및 입력 데이터

### Flow 모드 (유량 예측)
```python
mode = "flow"
target = ["Q_in"]  # 유입 유량

# 입력 데이터
inputs = [
    "AWS 기상 데이터",      # TA, HM, RN, WS, WD 등
    "level_TankA",          # 탱크 A 수위
    "level_TankB"           # 탱크 B 수위
]

# 제외 데이터 (데이터 누수 방지)
excluded = [
    "모든 TMS 지표",        # 미래 정보
    "flow_TankA",           # Q_in의 구성 요소
    "flow_TankB"            # Q_in의 구성 요소
]
```

### TMS 모드 (전체 수질 예측)
```python
mode = "tms"
targets = ["TOC_VU", "PH_VU", "SS_VU", "FLUX_VU", "TN_VU", "TP_VU"]

# 입력 데이터
inputs = ["AWS 기상 데이터"]

# 제외 데이터
excluded = ["모든 FLOW 데이터"]  # 미래 정보
```

### ModelA (유기물/입자 예측)
```python
mode = "modelA"
targets = ["TOC_VU", "SS_VU"]  # 유기물, 부유물질

# 입력 데이터
inputs = [
    "AWS 기상 데이터",
    "PH_VU", "FLUX_VU", "TN_VU", "TP_VU"  # 나머지 TMS 지표
]

# 제외 데이터
excluded = [
    "TOC_VU", "SS_VU",      # 예측 대상
    "모든 FLOW 데이터"       # 미래 정보
]
```

### ModelB (영양염 예측)
```python
mode = "modelB"
targets = ["TN_VU", "TP_VU"]  # 총질소, 총인

# 입력 데이터
inputs = [
    "AWS 기상 데이터",
    "TOC_VU", "PH_VU", "SS_VU", "FLUX_VU"  # 나머지 TMS 지표
]

# 제외 데이터
excluded = [
    "TN_VU", "TP_VU",       # 예측 대상
    "모든 FLOW 데이터"       # 미래 정보
]
```

### ModelC (공정 상태 예측)
```python
mode = "modelC"
targets = ["FLUX_VU", "PH_VU"]  # 유량계, pH

# 입력 데이터
inputs = [
    "AWS 기상 데이터",
    "TOC_VU", "SS_VU", "TN_VU", "TP_VU"  # 나머지 TMS 지표
]

# 제외 데이터
excluded = [
    "FLUX_VU", "PH_VU",     # 예측 대상
    "모든 FLOW 데이터"       # 미래 정보
]
```

---

## 🚀 실행 예시

### 1. 기본 파이프라인
```bash
python scripts/train.py --mode flow --resample 1h
```

### 2. 개선 파이프라인 (권장)
```bash
python scripts/train.py \
    --mode flow \
    --improved \
    --n-features 50 \
    --cv-splits 3 \
    --n-trials 50 \
    --resample 1h
```

### 3. Sliding Window 파이프라인
```bash
python scripts/train.py \
    --mode flow \
    --sliding-window \
    --improved \
    --window-size 24 \
    --horizon 1 \
    --n-features 50
```

---

## 📊 출력 예시

```
============================================================
WWTP 예측 모델 학습 (개선 파이프라인)
============================================================
모드: FLOW
데이터 경로: data/actual
리샘플링: 1h
피처 선택: 상위 50개
교차 검증: 3 splits
Optuna 시도: 50 trials
============================================================

[1/8] 데이터 로드 중...

[3/8] 개선된 파이프라인 실행 중...

============================================================
개선된 파이프라인 (Optuna) - 모드: FLOW
============================================================

[1/9] 시간축 정합 중...
[2/9] 결측치 보간 중 (1차)...
[3/9] 이상치 탐지 및 처리 중...
[4/9] 리샘플링 중 (1h)...
[5/9] 파생 특성 생성 중...
데이터셋 크기: 26193 샘플, 203 피처
[6/9] 데이터 분할 중...
  Train: 15715 샘플
  Valid: 5238 샘플
  Test:  5240 샘플
[7/9] 스케일링 중 (Train 기준)...
[8/9] 특성 선택 중 (상위 50개, Train 기준)...
  선택된 피처: 50개
[9/9] 모델 학습 및 평가 중...

============================================================
모델 학습: XGBoost
Optuna 최적화 중 (50 trials)...
============================================================
  최적 파라미터: {'max_depth': 7, 'learning_rate': 0.05, ...}
  최적 MSE: 125.34
  Early stopping: 234번째 반복

  Train - R²: 0.9234, RMSE: 8.52
  Valid - R²: 0.8567, RMSE: 11.23
  Test  - R²: 0.8432, RMSE: 12.15

============================================================
최종 결과 (Test Set)
============================================================
     model  R2_mean  RMSE_mean  MAPE_mean(%)
   XGBoost   0.8432      12.15          4.23
   HistGBR   0.8201      13.45          4.87
RandomForest 0.7856      15.23          5.45

============================================================
최고 성능 모델
============================================================
모델: XGBoost
Test R²: 0.8432
Test RMSE: 12.15

결과 저장 위치: results/ML

저장된 파일:
  📊 예측값: 3개 파일
  🤖 모델: 3개 파일

============================================================
학습 완료!
============================================================
```

---

## 🔍 핵심 포인트

### 데이터 누수 방지
1. **시간 순서 유지**: Train → Valid → Test (과거 → 미래)
2. **스케일링**: Train으로만 fit, Valid/Test는 transform
3. **피처 선택**: Train으로만 중요도 계산
4. **미래 정보 제외**: 
   - FLOW 예측 시 TMS 데이터 제외
   - TMS 예측 시 FLOW 데이터 제외

### 성능 향상 기법
1. **결측치 처리**: Forward Fill + EWMA
2. **이상치 처리**: IQR 기반 탐지 및 제거
3. **파생 특성**: Rolling, Lag, 시간 특성
4. **피처 선택**: 중요도 기반 상위 N개 선택
5. **하이퍼파라미터 최적화**: Optuna + TimeSeriesSplit
6. **Early Stopping**: XGBoost 과적합 방지

### 시계열 특화
1. **시간 순서 분할**: 과거 데이터로 미래 예측
2. **Sliding Window**: 과거 N시간 → 미래 예측
3. **Lag 특성**: 과거 값을 입력으로 사용
4. **Rolling 통계**: 이동 평균/표준편차
