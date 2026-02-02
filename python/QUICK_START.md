# 빠른 시작 가이드

> 💡 **이 문서는 프로젝트의 모든 기능을 다루는 통합 가이드입니다.**  
> 빠른 시작부터 고급 기능(Sliding Window, 결과 저장)까지 모두 포함되어 있습니다.

## 📑 목차

1. [빠른 시작](#-빠른-시작)
2. [사용법](#-사용법)
3. [프로젝트 구조](#-프로젝트-구조)
4. [파이프라인 비교](#-파이프라인-비교)
5. [지원 모델](#-지원-모델)
6. [주요 옵션](#-주요-옵션)
7. [Sliding Window 작동 원리](#-sliding-window-작동-원리)
8. [결과 저장 및 로드](#-결과-저장-및-로드)
9. [예상 출력](#-예상-출력)
10. [주의사항](#️-주의사항)
11. [TMS 모델 선택 가이드](#-tms-모델-선택-가이드)
12. [모델별 특성 엔지니어링](#-모델별-특성-엔지니어링)
13. [상세 문서](#-상세-문서)

---

## 🚀 빠른 시작

### 1단계: 의존성 설치

```bash
pip install -r requirements.txt
```

### 2단계: 학습 실행

**기본 파이프라인 (빠른 실험):**
```bash
python scripts/train.py --mode flow --data-root data/actual
```

**개선된 파이프라인 (최고 성능):**
```bash
python scripts/train.py --mode flow --improved --n-features 50 --cv-splits 3
```

**Sliding Window 파이프라인 (시계열 패턴 학습):**
```bash
python scripts/train.py --mode flow --sliding-window --window-size 24
```

## 📚 사용법

### CLI로 학습

**기본 파이프라인:**
```bash
# FLOW 모드 (유량 예측)
python scripts/train.py --mode flow --data-root data/actual

# TMS 모드 (전체 수질 예측 - 6개 지표)
python scripts/train.py --mode tms --data-root data/actual

# Model A (유기물/입자 계열: TOC_VU + SS_VU)
python scripts/train.py --mode modelA --data-root data/actual

# Model B (영양염 계열: TN_VU + TP_VU)
python scripts/train.py --mode modelB --data-root data/actual

# Model C (공정 상태 계열: FLUX_VU + PH_VU)
python scripts/train.py --mode modelC --data-root data/actual

# 시각화 포함
python scripts/train.py --mode flow --data-root data/actual --plot

# 커스텀 설정
python scripts/train.py \
  --mode modelA \
  --data-root data/actual \
  --resample 5min \
  --train-ratio 0.7 \
  --valid-ratio 0.15 \
  --test-ratio 0.15 \
  --random-state 42
```

**개선된 파이프라인:**
```bash
# 기본 개선 파이프라인
python scripts/train.py --mode flow --improved

# Model A (Optuna 최적화)
python scripts/train.py --mode modelA --improved --n-features 50

# 커스텀 설정
python scripts/train.py \
  --mode modelB \
  --improved \
  --n-features 50 \
  --cv-splits 3 \
  --n-trials 50 \
  --resample 1h \
  --save-dir results/ML/custom

# TMS 모드 (6개 지표 개별 모델 학습)
python scripts/train.py --mode tms --improved --n-features 100
```

**Sliding Window 파이프라인:**
```bash
# 기본 사용 (과거 24시간 → 다음 시간 예측, 결과 자동 저장)
python scripts/train.py --mode flow --sliding-window --window-size 24

# 윈도우 크기 변경 (과거 48시간)
python scripts/train.py --mode flow --sliding-window --window-size 48

# 예측 horizon 변경 (3시간 후 예측)
python scripts/train.py --mode flow --sliding-window --window-size 24 --horizon 3

# Sliding Window + Optuna 최적화
python scripts/train.py --mode flow --sliding-window --improved \
  --window-size 24 --n-features 50 --n-trials 50

# ModelA (TOC+SS 예측)
python scripts/train.py --mode modelA --sliding-window --improved \
  --window-size 24 --n-features 50

# 윈도우 이동 간격 조정 (메모리 절약)
python scripts/train.py --mode flow --sliding-window \
  --window-size 48 --stride 2 --n-features 30

# 결과 저장 옵션
python scripts/train.py --mode flow --sliding-window --window-size 24 \
  --sequence-format npz  # NPZ 형식 (기본, 권장)

python scripts/train.py --mode flow --sliding-window --window-size 24 \
  --no-save-sequences --no-save-model  # 예측값만 저장

python scripts/train.py --mode flow --sliding-window --window-size 24 \
  --no-save  # 저장 안 함
```

### Python 코드에서 사용

**기본 파이프라인:**

```python
from src.io import load_csvs, prep_flow, prep_aws
from src.pipeline import run_pipeline
from src.features import FeatureConfig
from src.split import SplitConfig

# 데이터 로드
df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569 = load_csvs("data/actual")
df_flow = prep_flow(df_flow)
df_aws = prep_aws(df_aws_368, df_aws_541, df_aws_569)

dfs = {"flow": df_flow, "tms": df_tms, "aws": df_aws}
time_col_map = {"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"}

# 커스텀 설정
feature_cfg = FeatureConfig(
    lag_hours=[1, 2, 3, 6, 12, 24],
    roll_hours=[3, 6, 12, 24]
)
split_cfg = SplitConfig(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)

# 파이프라인 실행
result = run_pipeline(
    dfs,
    mode="flow",
    time_col_map=time_col_map,
    resample_rule="1h",
    resample_agg="mean",
    feature_cfg=feature_cfg,
    split_cfg=split_cfg,
    random_state=42
)

# 결과 확인
print(result["metric_table"])
print(result["continuity"])
```

**개선된 파이프라인:**

```python
from src.io import load_csvs, prep_flow, prep_aws
from src.pipeline import run_improved_pipeline
from src.features import FeatureConfig
from src.split import SplitConfig

# 데이터 로드
df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569 = load_csvs("data/actual")
df_flow = prep_flow(df_flow)
df_aws = prep_aws(df_aws_368, df_aws_541, df_aws_569)

dfs = {"flow": df_flow, "tms": df_tms, "aws": df_aws}
time_col_map = {"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"}

# 개선된 파이프라인 실행
result = run_improved_pipeline(
    dfs,
    mode="flow",
    time_col_map=time_col_map,
    resample_rule="1h",
    n_top_features=50,
    cv_splits=3,
    n_trials=50,
    random_state=42,
    save_dir="results/ML/improved"
)

# 결과 확인
print(result["metric_table"])
print(f"선택된 피처: {len(result['top_features'])}개")
```

**Sliding Window 파이프라인:**

```python
from src.io import load_csvs, prep_flow, prep_aws
from src.pipeline import run_sliding_window_pipeline
from src.split import SplitConfig

# 데이터 로드
df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569 = load_csvs("data/actual")
df_flow = prep_flow(df_flow)
df_aws = prep_aws(df_aws_368, df_aws_541, df_aws_569)

dfs = {"flow": df_flow, "tms": df_tms, "aws": df_aws}
time_col_map = {"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"}

# Sliding Window 파이프라인 실행
result = run_sliding_window_pipeline(
    dfs,
    mode="flow",
    window_size=24,        # 과거 24시간
    horizon=1,             # 다음 시간 예측
    stride=1,              # 매 시간마다 윈도우 생성
    time_col_map=time_col_map,
    resample_rule="1h",
    n_top_features=50,
    cv_splits=3,
    n_trials=50,
    random_state=42,
    save_dir="results/ML/sliding_window",
    save_results=True,     # 결과 저장
    save_predictions=True, # 예측값 저장
    save_sequences=True,   # 시퀀스 데이터 저장
    save_model=True,       # 모델 저장
    sequence_format="npz"  # NPZ 형식
)

# 결과 확인
print(result["metric_table"])
print(f"원본 데이터: {len(result['X_original'])} 샘플")
print(f"윈도우 생성 후: {len(result['X_seq'])} 윈도우")
print(f"선택된 피처: {len(result['top_features'])}개")

# 저장된 파일 확인
if result.get("saved_files"):
    print("\n저장된 파일:")
    print(f"  예측값: {result['saved_files']['predictions']}")
    print(f"  시퀀스: {result['saved_files']['sequences']}")
    print(f"  모델: {result['saved_files']['models']}")
```

**저장된 결과 로드 및 사용:**

```python
from src.save_results import load_sequence_dataset
import pickle
import pandas as pd

# 1. 시퀀스 데이터 로드
data = load_sequence_dataset('results/ML/sliding_window/sequences/sequence_all_20240202_143022.npz')
X = data['X']  # (n_windows, window_size, n_features)
y = data['y']  # (n_windows, n_targets)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 2. 모델 로드
with open('results/ML/sliding_window/models/XGBoost_20240202_143022.pkl', 'rb') as f:
    model = pickle.load(f)

# 3. 스케일러 로드
with open('results/ML/sliding_window/models/scaler_20240202_143022.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 4. 예측값 로드
df_pred = pd.read_csv('results/ML/sliding_window/predictions/predictions_test_20240202_143022.csv',
                      index_col=0, parse_dates=True)
print(df_pred.head())

# 5. 새 데이터 예측
# X_new_scaled = scaler.transform(X_new)
# y_pred = model.predict(X_new_scaled)
```

## 📁 프로젝트 구조

```
src/
├── __init__.py              # 패키지 초기화
├── io.py                   # 데이터 로드 및 전처리
├── preprocess.py           # 결측치 처리, 리샘플링, 이상치 처리
├── features.py             # 피처 엔지니어링
├── split.py                # 데이터 분할
├── models.py               # 모델 정의 (기본 + Optuna)
├── feature_selection.py    # 피처 선택
├── scaling.py              # StandardScaler
├── metrics.py              # 평가 지표
├── visualization.py        # Learning Curve 시각화
├── sliding_window.py       # Sliding Window 생성 및 변환
├── save_results.py         # 결과 저장 (예측값, 시퀀스, 모델)
└── pipeline.py             # 파이프라인 (기본 + 개선 + Sliding Window)

scripts/
└── train.py                # 통합 학습 스크립트
```

## 🔄 파이프라인 비교

| 기능 | 기본 파이프라인 | 개선된 파이프라인 | Sliding Window 파이프라인 |
|------|----------------|------------------|--------------------------|
| 입력 방식 | 단일 시점 | 단일 시점 | **과거 N시간 윈도우** |
| 시계열 패턴 | ❌ 약함 | ❌ 약함 | ✅ **강함** |
| 모델 | 6개 기본 모델 | 7개 모델 + Optuna | 7개 모델 + Optuna |
| 스케일링 | ❌ | ✅ StandardScaler | ✅ StandardScaler |
| 피처 선택 | ❌ | ✅ 중요도 기반 | ✅ 중요도 기반 |
| 하이퍼파라미터 튜닝 | ❌ | ✅ Optuna | ✅ Optuna |
| 교차 검증 | ❌ | ✅ TimeSeriesSplit | ✅ TimeSeriesSplit |
| XGBoost | ❌ | ✅ Early Stopping | ✅ Early Stopping |
| 다중 타겟 | MultiOutput 래퍼 | 개별 모델 학습 | 개별 모델 학습 |
| 시각화 | 기본 | Learning Curve 추가 | Learning Curve 추가 |
| 데이터 샘플 | 원본 그대로 | 원본 그대로 | **감소 (window_size만큼)** |
| 특성 수 | 원본 | 선택된 N개 | **원본 × window_size → 선택된 N개** |
| 메모리 사용 | 💾 적음 | 💾💾 보통 | 💾💾💾 **많음** |
| 속도 | ⚡⚡⚡ 빠름 | ⚡⚡ 보통 | ⚡ **느림** |
| 성능 | ⭐⭐ 보통 | ⭐⭐⭐ 좋음 | ⭐⭐⭐⭐ **매우 좋음** |

**추천:**
- 빠른 실험: `python scripts/train.py --mode flow`
- 최고 성능 (단일 시점): `python scripts/train.py --mode flow --improved`
- 최고 성능 (시계열): `python scripts/train.py --mode flow --sliding-window --improved --window-size 24`

## 🤖 지원 모델

### 기본 파이프라인
1. LinearRegression
2. Ridge
3. Lasso
4. ElasticNet
5. RandomForest
6. HistGradientBoosting

### 개선된 파이프라인 (Optuna 포함)
1. **LinearRegression** - 파라미터 없음
2. **Ridge** - alpha 튜닝
3. **Lasso** - alpha, max_iter 튜닝
4. **ElasticNet** - alpha, l1_ratio, max_iter 튜닝
5. **RandomForest** - n_estimators, max_depth, min_samples_split 등 튜닝
6. **HistGradientBoosting** - learning_rate, max_depth, early_stopping 튜닝
7. **XGBoost** - learning_rate, max_depth, subsample 등 튜닝 + Early Stopping

## 💡 주요 옵션

### 공통 옵션
- `--mode`: 예측 모드
  - `flow`: 유량 예측 (Q_in)
  - `tms`: 전체 TMS 지표 (6개)
  - `modelA`: 유기물/입자 계열 (TOC_VU, SS_VU)
  - `modelB`: 영양염 계열 (TN_VU, TP_VU)
  - `modelC`: 공정 상태 계열 (FLUX_VU, PH_VU)
- `--data-root`: 데이터 디렉토리 경로 (기본: data/actual)
- `--resample`: 리샘플링 규칙 (5min, 1h 등)
- `--train-ratio`: 학습 데이터 비율 (기본: 0.6)
- `--valid-ratio`: 검증 데이터 비율 (기본: 0.2)
- `--test-ratio`: 테스트 데이터 비율 (기본: 0.2)
- `--random-state`: 랜덤 시드 (기본: 42)

### 기본 파이프라인 전용
- `--how`: 데이터 병합 방식 (inner/outer/left/right)
- `--agg`: 집계 방법 (mean 또는 auto)
- `--plot`: 최고 성능 모델 시각화

### 개선된 파이프라인 전용
- `--improved`: 개선된 파이프라인 활성화 (필수)
- `--n-features`: 선택할 피처 개수 (기본: 50)
- `--cv-splits`: TimeSeriesSplit 분할 수 (기본: 3)
- `--n-trials`: Optuna 시도 횟수 (기본: 50)
- `--save-dir`: 결과 저장 디렉토리 (기본: results/ML)

### Sliding Window 파이프라인 전용
- `--sliding-window`: Sliding Window 파이프라인 활성화 (필수)
- `--window-size`: 과거 몇 개의 시간 스텝을 볼 것인지 (기본: 24시간)
- `--horizon`: 미래 몇 스텝 후를 예측할 것인지 (기본: 1 = 다음 시간)
- `--stride`: 윈도우 이동 간격 (기본: 1 = 매 시간마다)
- `--use-3d`: 3D 입력 모델 사용 (LSTM 등, 현재 미지원)

### 결과 저장 옵션
- `--no-save`: 모든 결과 저장 안 함
- `--no-save-predictions`: 예측값 저장 안 함
- `--no-save-sequences`: 시퀀스 데이터 저장 안 함
- `--no-save-model`: 모델 저장 안 함
- `--sequence-format`: 시퀀스 저장 형식 (npz/pickle/csv, 기본: npz)

## 📊 예상 출력

**기본 파이프라인:**
```
============================================================
WWTP 예측 모델 학습 (기본 파이프라인)
============================================================
모드: flow
데이터 경로: data/actual
리샘플링: 5min
============================================================

[1/8] 데이터 로드 중...
[3/8] 파이프라인 실행 중...

============================================================
데이터셋 크기
============================================================
전체: 8760 샘플
학습: 5256 샘플
검증: 1752 샘플
테스트: 1752 샘플
피처 수: 150

============================================================
모델 성능 (테스트 데이터)
============================================================
              model   R2_mean  RMSE_mean  MAPE_mean(%)
  HistGBR           0.950000   0.150000          5.000
  RandomForest      0.945000   0.155000          5.200
  Ridge             0.920000   0.180000          6.000
  LinearRegression  0.918000   0.182000          6.100
  ElasticNet        0.915000   0.185000          6.200
  Lasso             0.910000   0.190000          6.500
```

**개선된 파이프라인:**
```
============================================================
WWTP 예측 모델 학습 (개선 파이프라인)
============================================================
모드: flow
피처 선택: 상위 50개
교차 검증: 3 splits
Optuna 시도: 50 trials
============================================================

데이터셋 크기: 8760 샘플, 150 피처

피처 선택 중 (상위 50개)...
데이터 스케일링 중...

============================================================
모델 학습: XGBoost
============================================================
  단일 타겟 학습...
  최적 파라미터: {'learning_rate': 0.05, 'max_depth': 5, ...}
  Early stopping: 287번째 반복

  Train - R²: 0.9850, RMSE: 0.12
  Valid - R²: 0.9520, RMSE: 0.15
  Test  - R²: 0.9480, RMSE: 0.16

============================================================
최종 결과 (Test Set)
============================================================
              model   R2_mean  RMSE_mean  MAPE_mean(%)
  XGBoost           0.948000   0.160000          4.800
  HistGBR           0.945000   0.165000          5.000
  RandomForest      0.940000   0.170000          5.200

최고 성능 모델: XGBoost
Test R²: 0.9480
Test RMSE: 0.16

결과 저장 위치: results/ML/improved
```

## 🎯 다음 단계

1. 다른 모드 시도 (`tms`, `all`)
2. 하이퍼파라미터 조정
3. 피처 엔지니어링 실험
4. 결과 분석 및 시각화

## ⚠️ 주의사항

- **TMS 모드**: 6개 지표(TOC_VU, PH_VU, SS_VU, FLUX_VU, TN_VU, TP_VU)를 각각 개별 모델로 학습
- **TMS 모델 그룹**: 
  - `modelA` (유기물/입자): TOC_VU, SS_VU 예측 시 나머지 4개 TMS 지표를 입력으로 사용
  - `modelB` (영양염): TN_VU, TP_VU 예측 시 나머지 4개 TMS 지표를 입력으로 사용
  - `modelC` (공정 상태): FLUX_VU, PH_VU 예측 시 나머지 4개 TMS 지표를 입력으로 사용
- **개선된 파이프라인**: Optuna로 인해 학습 시간이 오래 걸릴 수 있음
- **피처 선택**: 너무 적은 피처는 성능 저하, 너무 많은 피처는 과적합 가능성
- **Sliding Window**: 
  - 데이터 샘플 수가 window_size + horizon - 1만큼 감소
  - 특성 수가 window_size배 증가 (평탄화 시)
  - 메모리 사용량이 크게 증가 (stride 조정으로 완화 가능)
  - 학습 시간이 길어짐 (n_trials, cv_splits 조정 권장)
  - 시계열 패턴이 강할수록 효과적 (일일 주기, 강우 이벤트 등)
- **결과 저장**:
  - 기본적으로 모든 결과 자동 저장 (예측값, 시퀀스, 모델)
  - NPZ 형식 권장 (빠르고 용량 작음)
  - 파일명에 타임스탬프 포함 (덮어쓰기 방지)
  - 디스크 공간 확인 필요 (큰 데이터셋은 수백 MB 차지)

## 💡 TMS 모델 선택 가이드

TMS 지표들을 그룹화하여 예측하면 성능이 향상됩니다:

1. **Model A (유기물/입자 계열)**
   - 예측 대상: TOC_VU (총유기탄소), SS_VU (부유물질)
   - 입력 데이터: AWS 기상 데이터 + **나머지 TMS 지표 (PH_VU, FLUX_VU, TN_VU, TP_VU)**
   - 특징: 유입/침전/생물 반응에서 함께 움직이며, 강우/유량 이벤트에 동일한 영향을 받음
   - 핵심: FLUX(유량)와 영양염(TN, TP)이 TOC/SS 예측에 중요한 정보 제공

2. **Model B (영양염 계열)**
   - 예측 대상: TN_VU (총질소), TP_VU (총인)
   - 입력 데이터: AWS 기상 데이터 + **나머지 TMS 지표 (TOC_VU, PH_VU, SS_VU, FLUX_VU)**
   - 특징: 생물학적 영양염 제거(BNR) 구간에서 공정 조건을 공유하여 제거 성능이 연동됨
   - 핵심: FLUX(유량)와 유기물(TOC, SS)이 영양염 예측에 중요한 정보 제공

3. **Model C (공정 상태 계열)**
   - 예측 대상: FLUX_VU (유량), PH_VU (pH)
   - 입력 데이터: AWS 기상 데이터 + **나머지 TMS 지표 (TOC_VU, SS_VU, TN_VU, TP_VU)**
   - 특징: pH는 생물 반응과 연동되고, FLUX는 공정 부하/활성의 대표 지표
   - 핵심: 수질 지표(TOC, SS, TN, TP)가 공정 상태(FLUX, pH) 예측에 중요한 정보 제공

4. **FLOW 모델**
   - 예측 대상: Q_in (유입량)
   - 입력 데이터: **AWS 기상 데이터만 사용** (TMS 지표는 전혀 사용 안 함)
   - 특징: 강우량과 기상 조건으로 유입량 예측
   - 핵심: TMS 데이터는 유입 후 측정되므로 실시간 예측에 사용 불가

### 데이터 누수 방지 전략

실시간 예측 시나리오를 고려하여:
- **FLOW 모델**: TMS 지표는 유입 후 측정되므로 입력에서 완전히 제외
- **TMS 모델 (A, B, C)**: 
  - FLOW 데이터는 유입 후 측정되므로 입력에서 제외
  - **예측 대상 TMS 지표만 제외**, 나머지 TMS 지표는 입력으로 사용
  - 예: ModelA는 TOC/SS를 예측하지만, PH/FLUX/TN/TP는 입력으로 사용 가능
- **모든 모델**: 예측 대상 변수의 현재/과거 정보는 사용하지 않음

**권장 사용법:**
```bash
# 유입량 예측 (AWS 데이터만 사용)
python scripts/train.py --mode flow --improved

# 각 TMS 모델 그룹별로 학습 (AWS + 나머지 TMS 지표 사용)
python scripts/train.py --mode modelA --improved  # TOC+SS 예측, PH/FLUX/TN/TP 입력
python scripts/train.py --mode modelB --improved  # TN+TP 예측, TOC/PH/SS/FLUX 입력
python scripts/train.py --mode modelC --improved  # FLUX+PH 예측, TOC/SS/TN/TP 입력
```


## 🎨 모델별 특성 엔지니어링

각 모델은 노트북 설계에 따라 **완전히 다른 입력 데이터와 특화 특성**을 사용합니다.

### ModelFLOW (Q_in 예측) - 165개 이상 특성

**입력 데이터**: AWS 기상 + level_TankA/B (수위)  
**제외**: TMS 지표, flow_TankA/B (데이터 누수)

**특화 특성**:
- **수위-유량**: level_sum/diff, lag (1~36시간), rolling (평균/표준편차/IQR/추세)
- **강우 공간 통합**: mean/max/min/std/spread (3개 관측소)
- **ARI 지수**: tau6, tau12, tau24 (선행강우지수, 지수 감쇠 누적)
- **건조/습윤**: wet_flag, dry_spell_minutes (First flush 효과)
- **강우×수위**: rain_x_levelsum_lag1 (포화 상태 유입 급증)

**핵심 메커니즘**: 수위 → 유량 (물리적 인과 관계)

---

### ModelA (TOC+SS 예측) - 100개 이상 특성

**입력 데이터**: AWS 기상 + PH, FLUX, TN, TP  
**제외**: TOC, SS (예측 대상)

**특화 특성**:
- **강수**: 단기 집중도, AR_3/6/12/24H + log1p, rain_start/end, post_rain_6H, API 지수
- **기상**: VPD, 기상 안정성 (TA/HM_std_3H/6H)
- **TMS 부하**: TOC_proxy_load, SS_proxy_load (FLUX × PH/영양염)
- **영양염 비율**: TN/TP, log(TN+TP), PH×TN, PH×TP
- **공정 플래그**: pH_zone, TN_high_flag, TP_spike_flag
- **강수-TMS**: RN60×SS(t-1), (TN/TP)×PH, dry×RN15

**핵심 메커니즘**: 강수 → 우수 유입 → 토사/유기물 동반 유입

---

### ModelB (TN+TP 예측) - 160개 이상 특성

**입력 데이터**: AWS 기상 + PH, FLUX, SS, TOC  
**제외**: TN, TP (예측 대상)

**특화 특성**:
- **강수/기상**: ModelA와 동일 (단기 집중도, API, VPD, 기상 안정성)
- **시계열 메모리**: 10/30/60분 lag, 30min/1H/3H rolling (PH/FLUX/SS/TOC)
- **TMS 부하**: SS_load, TOC_load, FLUX×(SS+TOC)
- **상호작용**: PH×TOC, SS×FLUX, TOC/SS
- **변화율**: ΔPH, ΔFLUX, ΔSS, ΔTOC, |ΔFLUX|
- **Spike flags**: SS/TOC/PH/FLUX_spike_z2 (공정 이상 감지)

**핵심 메커니즘**: 유기물 부하 → 영양염 제거 효율

---

### ModelC (FLUX+PH 예측) - 170개 이상 특성

**입력 데이터**: AWS 기상 + TOC, SS, TN, TP  
**제외**: FLUX, PH (예측 대상)

**특화 특성**:
- **강수/기상**: ModelA/B와 동일
- **시계열 메모리**: 10/30/60분 lag, 30min/1H/3H rolling (TN/TP/SS/TOC)
- **조성/비율**: TOC/SS, SS/TOC, TN/TP, TP/TN, TOC/TN, TN/TOC (6가지)
- **상호결합**: TOC×SS, TN×TP (비선형 관계)
- **Spike flags**: TN/TP/SS/TOC_spike_z2
- **강수-TMS**: RN15/60×SS/TOC (희석/충격 효과)
- **온도-TMS**: TA×TN, TA×TOC (생물학적 반응)

**핵심 메커니즘**: 수질 조성 → 공정 상태 (역방향 예측)

---

### 입력 데이터 비교표

| 모델 | 예측 대상 | 입력 TMS | 입력 FLOW | 특성 개수 |
|------|----------|---------|----------|----------|
| **ModelFLOW** | Q_in | ❌ | level만 | 165개 |
| **ModelA** | TOC, SS | PH, FLUX, TN, TP | ❌ | 100개 |
| **ModelB** | TN, TP | PH, FLUX, SS, TOC | ❌ | 160개 |
| **ModelC** | FLUX, PH | TOC, SS, TN, TP | ❌ | 170개 |

**주의사항**:
- 모든 모델은 AWS 기상 데이터 사용
- 예측 대상 변수는 입력에서 제외 (데이터 누수 방지)
- 마스크 컬럼(`_is_missing`, `_imputed_*`, `_outlier_*`)은 lag/rolling 제외
- Rolling 특성은 shift(1) 후 계산 (미래 정보 누수 방지)

---

## 🔧 코드 최적화

`src/features.py`는 중복 코드를 제거하고 유틸리티 함수로 최적화되었습니다:

**추가된 유틸리티 함수**:
- `calculate_rolling_std()`: 기상 안정성 계산 (코드 71% 감소)
- `calculate_spike_flags()`: 공정 이상 감지 (코드 71% 감소)
- `calculate_derivatives()`: 변화율 계산 (코드 33% 감소)
- `calculate_ari()`: 선행강우지수 계산 (코드 82% 감소)

**효과**:
- 전체 코드 라인 수: 78줄 → 22줄 (72% 감소)
- 중복 코드: 10곳 → 4개 유틸리티 함수로 통합
- 유지보수성, 일관성, 가독성 대폭 향상

---

## 📖 상세 문서

### 개발 문서
- `NOTE.md`: 개발 노트 및 변경 이력
- `TODO.md`: 할 일 목록 및 향후 계획

### 특성 엔지니어링 문서
- `MODELFLOW_FEATURES_ADDED.md`: ModelFLOW 특성 상세 (수위-유량, ARI, First flush)
- `MODELA_FEATURES_ADDED.md`: ModelA 특성 상세 (부하, 영양염 비율, 공정 플래그)
- `MODELB_FEATURES_ADDED.md`: ModelB 특성 상세 (시계열 메모리, 부하, spike flags)
- `MODELC_FEATURES_ADDED.md`: ModelC 특성 상세 (조성 비율, 상호결합, 온도 상호작용)
- `FEATURE_DESIGN_CORRECTION.md`: 설계 수정 내역 (노트북 기반 정확한 설계)
- `FLOW_MODE_FIX.md`: FLOW 모드 입력 데이터 수정 (데이터 누수 방지)
- `FEATURES_OPTIMIZATION.md`: 코드 최적화 내역 (유틸리티 함수 추출)

### 코드 문서
- 각 모듈의 docstring 참조 (`src/*.py`)
- 함수별 상세 설명은 코드 내 주석 참조

---

## 🎓 학습 자료

### 시계열 예측
- [Time Series Forecasting with Sliding Windows](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
- [Understanding LSTM Input Shape](https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/)

### 하이퍼파라미터 최적화
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)

---

## 💬 문의 및 지원

문제가 발생하면 다음을 확인하세요:
1. 데이터 경로가 올바른지 (`data/actual/`)
2. 필요한 패키지가 설치되었는지 (`requirements.txt`)
3. Python 버전이 3.8 이상인지
4. 메모리가 충분한지 (Sliding Window는 많이 사용)

---

## 📝 라이선스

이 프로젝트는 내부 연구용으로 개발되었습니다.

---

**마지막 업데이트**: 2024-02-02  
**버전**: 2.0 (Sliding Window + 결과 저장 기능 추가)

---

## 💾 결과 저장 및 로드

### 자동 저장 (Sliding Window 파이프라인)

Sliding Window 파이프라인은 기본적으로 다음 결과를 자동 저장합니다:

1. **예측값** (CSV) - Train/Valid/Test 각각
2. **시퀀스 데이터** (NPZ/Pickle/CSV) - 원본 윈도우
3. **모델 및 메타데이터** (Pickle) - 최고 성능 모델

```bash
# 기본 사용 (모든 결과 자동 저장)
python scripts/train.py --mode flow --sliding-window --window-size 24
```

**저장 위치:**
```
results/ML/
├── predictions/
│   ├── predictions_train_20240202_143022.csv
│   ├── predictions_valid_20240202_143022.csv
│   └── predictions_test_20240202_143022.csv
├── sequences/
│   └── sequence_all_20240202_143022.npz
└── models/
    ├── XGBoost_20240202_143022.pkl
    ├── scaler_20240202_143022.pkl
    ├── features_20240202_143022.txt
    └── metadata_20240202_143022.pkl
```

### 선택적 저장

```bash
# 예측값만 저장
python scripts/train.py --mode flow --sliding-window --window-size 24 \
  --no-save-sequences --no-save-model

# 시퀀스 데이터만 저장
python scripts/train.py --mode flow --sliding-window --window-size 24 \
  --no-save-predictions --no-save-model

# 저장 안 함
python scripts/train.py --mode flow --sliding-window --window-size 24 --no-save
```

### 저장 형식 선택

```bash
# NPZ 형식 (기본, 권장 - 빠르고 용량 작음)
python scripts/train.py --mode flow --sliding-window --window-size 24 \
  --sequence-format npz

# Pickle 형식
python scripts/train.py --mode flow --sliding-window --window-size 24 \
  --sequence-format pickle

# CSV 형식 (사람이 읽기 쉬움, 용량 큼)
python scripts/train.py --mode flow --sliding-window --window-size 24 \
  --sequence-format csv
```

### 저장된 결과 로드

#### 1. 예측값 로드 (CSV)

```python
import pandas as pd
import matplotlib.pyplot as plt

# 예측값 로드
df = pd.read_csv('results/ML/predictions/predictions_test_20240202_143022.csv',
                 index_col=0, parse_dates=True)

# 컬럼: Q_in, Q_in_pred, Q_in_error, Q_in_error_pct
print(df.head())
print(df.describe())

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Q_in'], label='Actual', alpha=0.7)
plt.plot(df.index, df['Q_in_pred'], label='Predicted', alpha=0.7)
plt.legend()
plt.show()
```

#### 2. 시퀀스 데이터 로드 (NPZ)

```python
from src.save_results import load_sequence_dataset

# 시퀀스 데이터 로드
data = load_sequence_dataset('results/ML/sequences/sequence_all_20240202_143022.npz')

X = data['X']              # (n_windows, window_size, n_features)
y = data['y']              # (n_windows, n_targets)
feature_names = data['feature_names']
target_cols = data['target_cols']
window_size = data['window_size']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Window size: {window_size}")

# 특정 윈도우 확인
print(f"첫 번째 윈도우 X: {X[0].shape}")  # (window_size, n_features)
print(f"첫 번째 윈도우 y: {y[0]}")        # (n_targets,)
```

#### 3. 모델 로드 및 예측

```python
import pickle

# 모델 로드
with open('results/ML/models/XGBoost_20240202_143022.pkl', 'rb') as f:
    model = pickle.load(f)

# 스케일러 로드
with open('results/ML/models/scaler_20240202_143022.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 특성 리스트 로드
with open('results/ML/models/features_20240202_143022.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

# 메타데이터 로드
with open('results/ML/models/metadata_20240202_143022.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"모델: {metadata['best_model_name']}")
print(f"Test R²: {metadata['test_r2']:.4f}")
print(f"Window size: {metadata['window_size']}")

# 새 데이터 예측
# X_new: 새 데이터 (선택된 특성만)
# X_new_scaled = scaler.transform(X_new[feature_names])
# y_pred = model.predict(X_new_scaled)
```

### 파일 크기 비교

| 형식 | 파일 크기 | 로드 속도 | 권장 |
|------|----------|----------|------|
| **NPZ** | ~20 MB | ⚡⚡⚡ 빠름 | ✅ 권장 |
| **Pickle** | ~25 MB | ⚡⚡ 보통 | ⚠️ 호환성 주의 |
| **CSV** | ~150 MB | ⚡ 느림 | ❌ 비권장 |

**권장사항:**
- 일반적인 경우: **NPZ** 사용 (빠르고 용량 작음)
- Python 객체 저장: **Pickle** 사용
- Excel 분석: **CSV** 사용 (소규모 데이터만)

### 사용 시나리오

#### 시나리오 1: 모델 재사용

```python
# 저장된 모델로 새 데이터 예측
import pickle

# 모델 및 스케일러 로드
with open('results/ML/models/XGBoost_20240202_143022.pkl', 'rb') as f:
    model = pickle.load(f)
with open('results/ML/models/scaler_20240202_143022.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 새 데이터 예측
X_new_scaled = scaler.transform(X_new)
y_pred = model.predict(X_new_scaled)
```

#### 시나리오 2: 시퀀스 데이터 분석

```python
from src.save_results import load_sequence_dataset
import matplotlib.pyplot as plt

# 시퀀스 데이터 로드
data = load_sequence_dataset('results/ML/sequences/sequence_all_20240202_143022.npz')
X = data['X']  # (n_windows, window_size, n_features)

# 특정 특성의 시간 패턴 시각화
feature_idx = 0
plt.figure(figsize=(12, 6))
for i in range(10):  # 처음 10개 윈도우
    plt.plot(X[i, :, feature_idx], alpha=0.5)
plt.title(f'Feature {data["feature_names"][feature_idx]} - Time Pattern')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.show()
```

#### 시나리오 3: 예측 결과 분석

```python
import pandas as pd
import matplotlib.pyplot as plt

# 예측값 로드
df = pd.read_csv('results/ML/predictions/predictions_test_20240202_143022.csv',
                 index_col=0, parse_dates=True)

# 오차 분석
print("오차 통계:")
print(df[['Q_in_error', 'Q_in_error_pct']].describe())

# 오차 분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 오차 히스토그램
axes[0].hist(df['Q_in_error'], bins=50, edgecolor='black')
axes[0].set_title('Error Distribution')
axes[0].set_xlabel('Error')

# 오차 시계열
axes[1].plot(df.index, df['Q_in_error'])
axes[1].set_title('Error Over Time')
axes[1].axhline(y=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()
```

---

## 🔍 Sliding Window 작동 원리

### 개념

**Sliding Window**는 시계열 데이터에서 **과거 N개의 시간 스텝을 하나의 입력으로 묶어서** 미래를 예측하는 방식입니다.

```
기존 방식 (단일 시점):
  시간 t의 특성들 → 시간 t의 타겟 예측

Sliding Window 방식:
  시간 [t-23, t-22, ..., t-1, t]의 특성들 → 시간 t+1의 타겟 예측
  (24시간 윈도우)
```

### 데이터 분석 과정에서의 작동 순서

#### 1단계: 전처리 (기존과 동일)

```
원본 데이터 (10,000 시간)
  ↓ [시간축 정합]
  ↓ [결측치 보간]
  ↓ [이상치 처리]
  ↓ [리샘플링]
  ↓ [파생 특성 생성]
전처리 완료 (10,000 샘플, 100 특성)
```

#### 2단계: Sliding Window 생성 ⭐

```python
# 예시: window_size=24, horizon=1, stride=1

원본 데이터:
시간    temp  humidity  rain  → 타겟(유량)
0시     20    60        0       100
1시     21    58        0       105
2시     22    55        2       110
...
23시    23    52        5       120
24시    24    50        3       115
25시    25    48        1       118

↓ Sliding Window 적용

윈도우 1 (0-23시 → 24시 예측):
  입력: [[20,60,0], [21,58,0], ..., [23,52,5]]  (24개 시간 스텝)
  타겟: 115  (24시 유량)

윈도우 2 (1-24시 → 25시 예측):
  입력: [[21,58,0], [22,55,2], ..., [24,50,3]]  (24개 시간 스텝)
  타겟: 118  (25시 유량)

...

결과: (9,976 윈도우, 24 시간, 100 특성)
```

**핵심 포인트:**
- 원본 10,000 샘플 → 9,976 윈도우 (24개 감소)
- 각 윈도우는 과거 24시간의 패턴을 포함
- stride=1이면 매 시간마다 윈도우 생성 (최대 데이터 활용)

#### 3단계: 평탄화 (ML 모델용)

```python
# LSTM/RNN은 3D 입력 사용, 일반 ML 모델은 2D 필요

3D 윈도우 데이터:
  (9,976 윈도우, 24 시간, 100 특성)

↓ 평탄화 (flatten)

2D 데이터:
  (9,976 샘플, 2,400 특성)
  # 24 × 100 = 2,400개 특성

특성 이름 예시:
  temp_t-23, humidity_t-23, rain_t-23,  # 23시간 전
  temp_t-22, humidity_t-22, rain_t-22,  # 22시간 전
  ...
  temp_t-1, humidity_t-1, rain_t-1,     # 1시간 전
  temp_t0, humidity_t0, rain_t0         # 현재
```

**핵심 포인트:**
- 특성 수가 window_size배 증가 (100 → 2,400)
- 각 특성은 시간 정보를 포함 (t-23, t-22, ..., t0)
- 메모리 사용량 대폭 증가

#### 4단계: 데이터 분할

```python
# 시계열 순서 유지 (셔플 안 함)

9,976 윈도우
  ↓ Train (60%): 5,986 윈도우
  ↓ Valid (20%): 1,995 윈도우
  ↓ Test (20%):  1,995 윈도우
```

#### 5단계: 스케일링

```python
# Train 데이터로 fit, Valid/Test는 transform만

Train: (5,986, 2,400) → StandardScaler.fit_transform()
Valid: (1,995, 2,400) → StandardScaler.transform()
Test:  (1,995, 2,400) → StandardScaler.transform()
```

#### 6단계: 피처 선택 ⭐

```python
# 2,400개 특성 → 상위 50개 선택 (RandomForest 중요도)

Train: (5,986, 2,400) → (5,986, 50)
Valid: (1,995, 2,400) → (1,995, 50)
Test:  (1,995, 2,400) → (1,995, 50)

선택된 특성 예시:
  temp_t-1, temp_t-2, temp_t-3,      # 최근 온도
  rain_t0, rain_t-1, rain_t-6,       # 최근 강수
  humidity_t-12, humidity_t-24,      # 주기적 습도
  ...
```

**핵심 포인트:**
- 2,400개 중 중요한 50개만 선택
- 시간 패턴이 중요한 특성이 자동 선택됨
- 메모리 사용량 대폭 감소

#### 7단계: 모델 학습

```python
# Optuna로 하이퍼파라미터 최적화

for model in [Ridge, Lasso, RandomForest, XGBoost, ...]:
    # TimeSeriesSplit 교차 검증
    best_params = optuna_optimize(model, Train, Valid)
    
    # 최적 파라미터로 학습
    model.fit(Train, y_train)
    
    # 평가
    y_pred = model.predict(Test)
    r2 = r2_score(y_test, y_pred)
```

### 시각적 비교

#### 기존 방식 (단일 시점)

```
입력:
  [현재 시점의 특성들]
  
예측:
  현재 유량
  
문제점:
  - 과거 패턴 무시
  - 시간적 의존성 학습 불가
```

#### Sliding Window 방식

```
입력:
  [23시간 전, 22시간 전, ..., 1시간 전, 현재]
  
예측:
  다음 시간 유량
  
장점:
  - 과거 추세 학습
  - 주기성 포착 (일일 패턴)
  - 시간적 의존성 학습
```

### 실제 예시: 강우 이벤트

```
시나리오: 강우 후 유량 증가 예측

기존 방식:
  현재 강수량: 5mm → 유량 예측: 120 m³/h
  (과거 강수 이력 무시)

Sliding Window (24시간):
  0-6시간 전: 강수 없음
  6-12시간 전: 강수 시작 (2mm)
  12-18시간 전: 강수 증가 (5mm)
  18-24시간 전: 강수 지속 (3mm)
  현재: 강수 감소 (1mm)
  
  → 모델이 "강수가 지속되었고 이제 감소 중"이라는 패턴 학습
  → 유량 예측: 150 m³/h (더 정확)
```

### 파라미터 영향

#### window_size (윈도우 크기)

```
window_size=12 (12시간):
  - 단기 패턴 포착
  - 데이터 손실 적음
  - 특성 수: 100 × 12 = 1,200개

window_size=24 (24시간):
  - 일일 패턴 포착 ⭐ 권장
  - 데이터 손실 보통
  - 특성 수: 100 × 24 = 2,400개

window_size=48 (48시간):
  - 장기 추세 포착
  - 데이터 손실 많음
  - 특성 수: 100 × 48 = 4,800개
```

#### horizon (예측 시점)

```
horizon=1 (다음 시간):
  - 단기 예측
  - 높은 정확도
  - 실시간 제어용

horizon=3 (3시간 후):
  - 중기 예측
  - 보통 정확도
  - 운영 계획용

horizon=6 (6시간 후):
  - 장기 예측
  - 낮은 정확도
  - 전략 계획용
```

#### stride (윈도우 이동 간격)

```
stride=1 (매 시간):
  - 최대 데이터 활용
  - 윈도우 수: 9,976개
  - 학습 시간: 길음

stride=2 (2시간마다):
  - 데이터 절반 사용
  - 윈도우 수: 4,988개
  - 학습 시간: 짧음
  - 메모리 절약
```

### 성능 향상 메커니즘

#### 1. 시간적 의존성 학습

```
예: 유량은 6시간 전 강수량과 강한 상관관계

기존 방식:
  현재 강수량만 사용 → 상관관계 0.6

Sliding Window:
  6시간 전 강수량 포함 → 상관관계 0.85
```

#### 2. 주기성 포착

```
예: 유량은 24시간 주기 (일일 패턴)

기존 방식:
  주기성 학습 불가

Sliding Window (24시간):
  어제 같은 시간 패턴 학습 → 정확도 향상
```

#### 3. 추세 학습

```
예: 강수 후 유량 증가 추세

기존 방식:
  현재 값만 → 추세 무시

Sliding Window:
  과거 12시간 추세 → 미래 예측 개선
```

### 실전 팁

#### 1. 윈도우 크기 선택

```bash
# 일일 패턴이 중요한 경우 (권장)
python scripts/train.py --mode flow --sliding-window --window-size 24

# 빠른 변화 포착
python scripts/train.py --mode flow --sliding-window --window-size 12

# 장기 추세 포착
python scripts/train.py --mode flow --sliding-window --window-size 48
```

#### 2. 메모리 관리

```bash
# 메모리 부족 시: stride 증가 + 피처 감소
python scripts/train.py --mode flow --sliding-window \
  --window-size 48 --stride 2 --n-features 30
```

#### 3. 성능 최적화

```bash
# 최고 성능: Sliding Window + Optuna
python scripts/train.py --mode flow --sliding-window --improved \
  --window-size 24 --n-features 50 --n-trials 100
```

### 예상 성능 향상

```
기존 방식:
  Test R²: 0.85
  Test RMSE: 15.2

Sliding Window (24시간):
  Test R²: 0.89 (+4.7%)
  Test RMSE: 12.8 (-15.8%)

Sliding Window + Optuna:
  Test R²: 0.92 (+8.2%)
  Test RMSE: 11.5 (-24.3%)
```

---