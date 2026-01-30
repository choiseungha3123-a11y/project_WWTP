# 빠른 시작 가이드

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
└── pipeline.py             # 파이프라인 (기본 + 개선)

scripts/
└── train.py                # 통합 학습 스크립트
```

## 🔄 파이프라인 비교

| 기능 | 기본 파이프라인 | 개선된 파이프라인 |
|------|----------------|------------------|
| 모델 | 6개 기본 모델 | 7개 모델 + Optuna |
| 스케일링 | ❌ | ✅ StandardScaler |
| 피처 선택 | ❌ | ✅ 중요도 기반 |
| 하이퍼파라미터 튜닝 | ❌ | ✅ Optuna |
| 교차 검증 | ❌ | ✅ TimeSeriesSplit |
| XGBoost | ❌ | ✅ Early Stopping |
| 다중 타겟 | MultiOutput 래퍼 | 개별 모델 학습 |
| 시각화 | 기본 | Learning Curve 추가 |
| 속도 | 빠름 | 느림 (튜닝 포함) |
| 성능 | 기본 | 최적화됨 |

**추천:**
- 빠른 실험: `python scripts/train.py --mode flow`
- 최고 성능: `python scripts/train.py --mode flow --improved`

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

각 모델의 특성 엔지니어링에 대한 상세 설명은 다음 문서를 참조하세요:

- `MODELFLOW_FEATURES_ADDED.md`: ModelFLOW 특성 상세 (수위-유량, ARI, First flush)
- `MODELA_FEATURES_ADDED.md`: ModelA 특성 상세 (부하, 영양염 비율, 공정 플래그)
- `MODELB_FEATURES_ADDED.md`: ModelB 특성 상세 (시계열 메모리, 부하, spike flags)
- `MODELC_FEATURES_ADDED.md`: ModelC 특성 상세 (조성 비율, 상호결합, 온도 상호작용)
- `FEATURE_DESIGN_CORRECTION.md`: 설계 수정 내역 (노트북 기반 정확한 설계)
- `FLOW_MODE_FIX.md`: FLOW 모드 입력 데이터 수정 (데이터 누수 방지)
- `FEATURES_OPTIMIZATION.md`: 코드 최적화 내역 (유틸리티 함수 추출)

