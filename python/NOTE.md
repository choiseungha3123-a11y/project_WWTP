# 개발 노트

> 프로젝트 진행 과정 및 주요 변경 사항을 기록합니다.

---

## 📅 2026년 2월 6일

### 📂 작업 파일
```
notebook/DL/LSTM_FLOW.ipynb
notebook/DL/LSTM_TMS.ipynb
notebook/DL/analyze_predictions.py
notebook/feature/WF_feature_selection.py
notebook/feature/feature_engineering.py
archive
```

### LSTM_FLOW 모델 성능 향상

**이유**:
- flow에 대한 LSTM 모델 성능도 R2 기준 0.3 수준

**내용**:
- 하이퍼파라미터 수정
  - hidden_size: 64 -> 128
  - num_layers: 2 -> 4
  - dropout: 0.2 -> 0.3
  - batch_size: 16 -> 64
  - learning_rate: 0.001 -> 0.0003
- resample 1h -> 30min
- Layer Normalization 추가
- Multi-head Attention 강화(4 -> 8)
- MSELoss -> HuberLoss
- ReduceLROnPlateau -> ConsinAnnealingWarmRestarts(주기적 restart로 local minima 탈출)
- Gradient Clipping 추가

**결과**:
- flow 예측 성능을 0.62로 올림
- 구간별 flow 예측 성능 불균형
  - 낮은 flow(< 320): MAPE 27.05%
  - 중간 flow(320 - 400): MAPE 5.17%
  - 높은 flow(> 400): MAPE 6.40%

### LSTM_TMS 모델 성능 향상 시도

**이유**:
- tms에 대한 LSTM 모델 성능은 R2 기준 거의 마이너스 값

**내용**:
- OUTLIER_CONFIG(zscore, both=False)
  - TOC_VU: R2 -1.8612
  - SS_VU: R2 -0.5181
  - TN_VU: R2 -0.1558
  - TP_VU: R2 -2.1524
  - FLUX_VU: R2 -0.0079
  - PH_VU: R2 -0.1669


### LSTM 파일 MAPE 값 수정

**내용**:
- 현재 y 값을 2D로 고정했기 때문에 y.dim() == 1 조건에 의해 MAPE 값을 계산하지 않음
- MAPE 값이 비정상적으로 크고 clamp_min(1e-6)이 너무 작아서 MAPE를 폭발시킴
- threshold를 설정해서 0.1 이상인 경우만 MAPE에 포함시킴
- 카운터 기반 평균 계산
- MAPE plot에 대한 한글 폰트 깨짐 해결

### LSTM 파일 리펙토링

**이유**:
- LSTM.ipynb 안의 코드가 특성 엔지니어링 코드에 의해 너무 길어짐

**내용**:
- feature_engineering.py 생성
- NaN/inf 값 사전 검증, 특성 이름과 X shape 일치 확인, X와 y 샘플 수 일치 확인
- 누적 중요도 threshold 달성 관련 오류 수정

### archive 폴더 생성

**이유**:
- ML과 DL 파일 버전을 올리면서 새로운 파일이 많이 생김
- 현재 사용 중인 파일만 각 폴더에 유지하고 싶음

- archive/README.md 참조

### analyze_predictions.py 생성

- test 데이터에 대한 예측값을 results/DL/{mode}_predictions.csv에 저장
- 실측값과 예측값의 차이를 시각적으로 확인하기 위해 작성
  - 시계열 비교(실측 vs. 예측)
  - 산점도 (실측 vs. 예측)
  - 예측 에러 시계열
  - 구간별 예측 정확도(MAPE)

### ✅ 다음 할 일 (2026/02/09)
- [] LSTM 성능 개선
- [] LSTM 특성에 target 컬럼의 lagging 추가

---

## 📅 2026년 2월 5일

### 📂 작업 파일
```
notebook/DL/LSTM.ipynb
notebook\feature\WF_feature_selection.py
```

### LSTM 모델 수정

**내용**:
- 어제 완성하지 못한 LSTM 모델 완성
- 1시간이 아닌 30분 단위로 리샘플링
- flow, modelA, modelB, modelC 모드 설정 시 모드에 해당하는 종속 변수에 대한 예측
- LSTM에 Attention 추가
- Walk-Forward Validation을 통해 특성 선택
- 데이터 누수 확인
- 실제값과 예측값 저장 및 선택된 특성 저장
- 학습 완료 모델 저장

### 홍봉희 교수님 미팅

**내용**:
- AWS 데이터에 하수처리장과 AWS 간 거리 데이터 추가
- 제일 간단한 유입유량 모델 먼저 완성
- TMS 모델의 경우 6개의 모델 만들기
  1) 입력 데이터로 종속 변수가 들어가지 않는 모델
  2) 입력 데이터로 종속 변수가 들어가는 모델 -> 예측 시기가 짧은 입력된 종속 변수에 +/-로 값을 예측할테니 lagging하여 예측
- TMS 모델의 경우 유입된 하수가 하수처리 과정을 거쳐 TMS에 측정됨으로, 하수처리 과정의 transformer 모델을 통해 학습한 후 LSTM을 통해 최종 TMS 지표를 예측하는 transformer + LSTM 모델을 구현

**수정 사항**:
- AWS 데이터에 거리 정보(위경도 기반) 추가
  - 368: 1.02km
  - 541: 4.61km
  - 569: 1.24km
- 모드가 4개 -> 7개 변경(flow, toc, ss, tn, tp, flux, ph)
- tms 지표 예측 시, 유입유량 데이터 활용

### ✅ 다음 할 일 (2026/02/06)
- [X] LSTM 성능 개선

---

## 📅 2026년 2월 4일

### 📂 작업 파일
```
notebook/DL/LSTM.ipynb
```

### LSTM 모델 데이터 로드, 전처리, 학습, 평가 코드 작성

**이유**:
- 기존 src/DL 폴더의 코드가 작동하지 않아서 새롭게 작성

**결과**:
- 여전히 loss 값이 nan으로 고정되는 현상이 나타남

### 각 지표에 대한 기준 잡기 및 회의

**이유**:
- 각 지표에 대한 이상치 알림을 만들기 위해 기준을 잡을 필요가 있었음
- 운영 시스템 점수에 대한 계산식을 세울 필요가 있음

**결과**:
- 법령으로 정해진 수치에 대해 기준을 잡음
- 0.2((예상 초과 확률, 지표 7개 중 기준을 넘는 개수) * 100) + 0.4(기준 대비 여유도) + 0.15(7개의 지표에 대한 예측 - 실측 0~100 스케일링) + 0.15(데이터 신뢰도) + 0.1(외부요인, 계절/강우)

### ✅ 다음 할 일 (2026/02/05)
- [X] LSTM 고치기

---

## 📅 2026년 2월 3일

### 📂 작업 파일
```
src/ML/features.py                # 수정 (타겟 lag 피처 추가)
src/DL/features.py                # 완성 (도메인 특화 피처 포팅)
src/DL/model_config.py            # 생성 (4개 모델 사양)
src/DL/preprocessing.py           # 수정 (장기 결측 처리 개선)
src/ML/preprocess.py              # 수정 (장기 결측 처리 개선)
scripts/DL/*.py
```

### 🔧 결측치 및 이상치 처리 전략 개선 ⭐⭐

**변경 이유**: 
- 장기 결측을 NaN으로 유지하면 학습 시 데이터 손실 발생
- 이상치를 NaN으로만 변환하면 추가 결측치 발생

**1. 결측치 처리 개선**

**새로운 전략**:
- **단기 결측 (1-3시간)**: Forward Fill (변경 없음)
- **중기 결측 (4-12시간)**: EWMA (span=6, 변경 없음)
- **장기 결측 (12시간+)**: **EWMA (span=24)** ← 변경!
  - 기존: NaN 유지 → 데이터 손실
  - 개선: 장기 span EWMA로 채우기 → 데이터 보존
  - span을 4배로 늘려 더 부드러운 보간

**2. 이상치 처리 개선**

**새로운 전략**:
- 이상치 탐지: 도메인 지식 + 통계적 방법 (변경 없음)
- 이상치 대체: **EWMA (span=12)로 대체** ← 변경!
  - 기존: NaN으로 변환 → 추가 결측치 발생
  - 개선: EWMA로 즉시 대체 → 연속성 유지
  - 시간 가중 이동평균으로 부드럽게 대체

**적용 범위**:
- `src/DL/preprocessing.py`: LSTM 파이프라인
- `src/ML/preprocess.py`: ML 파이프라인

**마스크 추가**:
- 결측치: `{col}_imputed_long_ewma` (장기 EWMA로 채움)
- 이상치: `{col}_outlier_replaced_ewma` (EWMA로 대체됨)

**장점**:
- 데이터 손실 최소화
- 학습 가능한 샘플 수 증가
- 시계열 연속성 유지
- 과거 추세를 반영한 부드러운 보간/대체

### 🎯 타겟 Lag 피처 구현 (Autoregressive Features) ⭐

**배경**: 시계열 예측에서 과거 타겟 값은 매우 강력한 예측 변수이나, 기존 코드는 타겟을 입력에서 완전히 제외

**작업 내용**:
- `build_features()`: 타겟 lag/rolling 피처 자동 생성 (`add_target_lags=True`)
  - Lag 피처: `y_flowA_lag1`, `y_flowA_lag2`, ..., `y_flowA_lag24`
  - Rolling 피처: `y_flowA_rmean3`, `y_flowA_rstd3`, `y_flowA_rmin3`, `y_flowA_rmax3` 등
  - `shift(1)` 적용으로 미래 정보 누수 방지
- `make_supervised_dataset()`: 타겟 lag 피처를 X에 포함 (`keep_target_lags=True`)
  - 현재 시점 타겟은 제외 (데이터 누수 방지)
  - 과거 타겟 값만 입력으로 사용 (autoregressive)

### 🧠 LSTM 도메인 특화 피처 엔지니어링 완성 ⭐⭐⭐

**배경**: LSTM 파이프라인에 ML 모델에서 사용하던 도메인 특화 피처가 없어 성능 저하 우려

**작업 내용**:
- `src/DL/features.py` 완전 재작성 (1,000+ 라인)
  - `src/ML/features.py`의 모든 도메인 특화 함수 포팅
  - 모델별 특성 생성 자동화 (`model_mode` 파라미터)

**구현된 도메인 특화 피처 함수**:

1. **유틸리티 함수**:
   - `calculate_rolling_std()`: 기상 안정성 계산
   - `calculate_spike_flags()`: 이상치 플래그 (z-score)
   - `calculate_derivatives()`: 변화율 계산
   - `calculate_ari()`: 선행강우지수 (ARI)

2. **강수 피처** (`add_rain_features()`):
   - 기본: 변화율, 선행강수량 (AR_3H/6H/12H/24H), 건조기간
   - ModelA/B/C 전용: 단기 집중도, API 지수, rain_start/end 플래그, post_rain_6H

3. **기상 피처** (`add_weather_features()`):
   - 기본: 이슬점 강하, 온도-습도 상호작용, 온도 변화율
   - ModelA/B/C 전용: VPD (증기압 부족), 기상 안정성 (rolling_std)

4. **TMS 상호작용 피처** (`add_tms_interaction_features()`):
   - **ModelA 전용**: TOC/SS_proxy_load, 영양염 비율, pH_zone, TN_high_flag, TP_spike_flag, 변화율
   - **ModelB 전용**: SS/TOC_load, PH×TOC, SS×FLUX, TOC/SS 비율, spike flags
   - **ModelC 전용**: 조성 비율 (TOC/SS, TN/TP, TOC/TN), 상호결합, spike flags

5. **강수-TMS 상호작용** (`add_rain_tms_interaction_features()`):
   - **ModelA**: RN_60m×SS(t-1), (TN/TP)×PH, dry×RN_15m
   - **ModelB**: dry×RN_15m, RN_60m×SS_lag1h
   - **ModelC**: RN_15m/60m×SS/TOC, TA×TN/TOC
   - **Flow**: rain×level_sum_lag1

6. **수위-유량 피처** (`add_level_flow_features()` - Flow 전용):
   - level_sum/diff, lag (1/2/3/6/12/36시간)
   - rolling 통계: 평균/표준편차/최소/최대/IQR/추세

7. **강우 공간 피처** (`add_rain_spatial_features()` - Flow 전용):
   - 공간 통계: mean/max/min/std/spread
   - 단기 집중도, ARI (tau=0.5/1/2), wet_flag, dry_spell

**통합 함수** (`create_features()`):
- 모델 모드에 따라 자동으로 적절한 도메인 피처 생성
- 기본 시간/lag/rolling 피처 + 도메인 특화 피처
- 타겟 컬럼 자동 제외 (데이터 누수 방지)

**테스트 결과**:
```
✓ Flow 모드: 1,243개 새 피처 생성
✓ ModelA 모드: 621개 새 피처 생성
✓ 모든 도메인 함수 정상 작동 확인
```

### 📋 모델 사양 정의 완료

**작업 내용**:
- `src/DL/model_config.py` 생성
- 4개 모델 사양 정의:
  - **flow**: Q_in 예측 (TMS 데이터 미사용)
  - **modelA**: TOC_VU, SS_VU 예측 (FLUX 입력 사용)
  - **modelB**: TN_VU, TP_VU 예측 (FLUX 입력 사용)
  - **modelC**: FLUX_VU, PH_VU 예측 (FLUX 제외)

### 기존 train.py 코드에서 LSTM 분리

**배경**: 학습 시 잦은 오류 발생

**작업 내용**:
- ML과 DL 폴더 구분
- LSTM 파이프라인 코드 새롭게 작성

### ✅ 다음 할 일 (2026/02/04)
- [X] LSTM 모델 새롭게 작성
---

## 📅 2026년 2월 2일

### 📂 작업 파일
```
src/sliding_window.py          # 새로 생성
src/save_results.py            # 새로 생성
src/pipeline.py                # 수정
src/preprocess.py              # 수정
scripts/train.py               # 수정
```

### 🔄 1. Sliding Window 기능 구현

**배경**: 시계열 데이터에서 과거 N개의 시간 스텝을 입력으로 사용하여 미래 예측 성능 향상

**작업 내용**:
- `src/sliding_window.py` 모듈 생성
  - `create_sliding_windows()`: 시계열 데이터를 sliding window로 변환
  - `flatten_windows_for_ml()`: 3D 윈도우를 2D로 평탄화 (일반 ML 모델용)
  - `create_feature_names_for_flattened_windows()`: 평탄화된 특성 이름 생성
  - `split_windowed_data()`: 윈도우 데이터 분할
- `src/pipeline.py`에 `run_sliding_window_pipeline()` 추가
- `scripts/train.py`에 Sliding Window 옵션 통합
  - `--sliding-window`: Sliding Window 파이프라인 활성화
  - `--window-size`: 과거 몇 개의 시간 스텝을 볼 것인지 (기본: 24)
  - `--horizon`: 미래 몇 스텝 후를 예측할 것인지 (기본: 1)
  - `--stride`: 윈도우 이동 간격 (기본: 1)
  - `--use-3d`: 3D 입력 모델 사용 (현재 미지원)

**Sliding Window 파이프라인 단계**:
1. 시간축 정합 → 결측치 보간 → 이상치 처리 → 리샘플링 → 파생 특성 생성
2. **Sliding Window 생성** (과거 N시간 → 미래 예측)
3. 데이터 분할 (Train/Valid/Test)
4. 평탄화 (ML 모델용 2D 변환)
5. 스케일링 (StandardScaler)
6. 피처 선택 (RandomForest 중요도)
7. 모델 학습 (Optuna 최적화)

### 💾 2. 결과 저장 기능 구현

**배경**: 학습 결과(예측값, 시퀀스 데이터, 모델)를 재사용하기 위해 저장 필요

**작업 내용**:
- `src/save_results.py` 모듈 생성
  - `save_predictions()`: 예측값을 CSV로 저장 (train/valid/test)
  - `save_sequence_dataset()`: 시퀀스 데이터를 NPZ/Pickle/CSV로 저장
  - `load_sequence_dataset()`: 저장된 시퀀스 데이터 로드
  - `save_model_and_metadata()`: 모델, 스케일러, 메타데이터 저장
  - `save_all_results()`: 모든 결과를 한 번에 저장
- `src/pipeline.py`의 `run_sliding_window_pipeline()`에 저장 기능 통합
- `scripts/train.py`에 저장 옵션 추가
  - `--no-save`: 모든 결과 저장 안 함
  - `--no-save-predictions`: 예측값 저장 안 함
  - `--no-save-sequences`: 시퀀스 데이터 저장 안 함
  - `--no-save-model`: 모델 저장 안 함
  - `--sequence-format`: 시퀀스 저장 형식 (npz/pickle/csv, 기본: npz)

### 🔧 3. 전처리 개선 (결측치 및 이상치 처리) ⭐

**배경**: 
- 기존: 장기 결측은 NaN 유지 → 모델 학습 시 샘플 손실
- 기존: 이상치를 NaN으로 변환 → 추가 결측치 발생

**작업 내용**:
- `src/preprocess.py` 수정
  - **결측치 처리 개선**:
    - 단기 결측 (1-3시간): Forward Fill (기존 유지)
    - 중기 결측 (4-12시간): EWMA (기존 유지)
    - 장기 결측 (12시간+): **Rolling Median으로 대체** (변경!)
      - `ImputationConfig.rolling_window`: 24시간 (기본값)
      - `center=True`로 앞뒤 데이터 모두 사용
      - 중앙값 기반으로 안정적 보간
  - **이상치 처리 개선**:
    - 이상치 탐지: 도메인 지식 + 통계적 방법 (기존 유지)
    - 이상치 대체: **EWMA로 대체** (변경!)
      - `OutlierConfig.ewma_span`: 12시간 (기본값)
      - 시간 가중 이동평균으로 부드럽게 대체
      - NaN 생성 없이 연속성 유지


### 📚 4. 문서 통합 및 정리

**배경**: 중복된 문서가 많아 유지보수가 어려움

**작업 내용**:
- 3개의 중복 문서 삭제
  - `TRAIN_USAGE.md` (train.py 사용 가이드)
  - `SLIDING_WINDOW_README.md` (Sliding Window 상세 가이드)
  - `SAVE_RESULTS_GUIDE.md` (결과 저장 가이드)
- 모든 핵심 내용을 `QUICK_START.md`에 통합
- `NOTE.md`와 `TODO.md`는 활발히 사용 중이므로 유지

**통합된 QUICK_START.md 구조**:
1. 빠른 시작
2. 사용법 (CLI + Python 코드)
3. 프로젝트 구조
4. 파이프라인 비교 (기본/개선/Sliding Window)
5. 지원 모델
6. 주요 옵션
7. **Sliding Window 작동 원리** (7단계 상세 설명)
8. **결과 저장 및 로드** (예측값/시퀀스/모델)
9. 예상 출력
10. 주의사항
11. TMS 모델 선택 가이드
12. 모델별 특성 엔지니어링
13. 상세 문서

**결과**:
- 문서 개수: 5개 → 2개 (메인 문서만)
- 중복 제거로 유지보수 부담 감소
- 모든 정보가 한 곳에 집중되어 접근성 향상

### 📝 5. NOTE.md 구조 개선

**작업 내용**:
- 문서 헤더 추가 (제목 + 설명)
- 날짜 형식 통일: "📅 2026년 1월 30일"
- 섹션별 이모지 아이콘 추가
  - 📂 작업 파일
  - 🔍 데이터 분석
  - 🔧 기술적 수정
  - 🤖 머신러닝
  - 🎯 전략/목표
  - 🔄 리팩토링
  - 🧠 딥러닝
  - 📊 데이터 처리
  - 🔬 실험
  - 🚀 개선
  - 💾 데이터 저장
  - 📚 문서화
  - ✅ 다음 할 일
- 정보 계층화 및 가독성 향상
- **볼드체**로 중요 키워드 강조
- 체크리스트 형식의 "다음 할 일" 섹션

**결과**:
- 일관된 구조로 정보 찾기 쉬워짐
- 시각적 구분으로 가독성 대폭 향상
- 내용은 전혀 수정하지 않고 구조만 개선

### 💡 주요 성과

**기능 구현**:
- ✅ Sliding Window 모듈 완성 (`src/sliding_window.py`)
- ✅ 결과 저장 모듈 완성 (`src/save_results.py`)
- ✅ **LSTM 딥러닝 모듈 완성** (`src/models_dl.py`) ⭐
- ✅ **LSTM 파이프라인 통합** (`src/pipeline.py`, `scripts/train.py`)
- ✅ 시계열 예측 성능 향상 기대

**문서 정리**:
- ✅ 중복 문서 3개 삭제
- ✅ 통합 가이드 1개로 집중
- ✅ 유지보수 효율성 향상

**구조 개선**:
- ✅ NOTE.md 가독성 대폭 향상
- ✅ 일관된 양식 적용
- ✅ 정보 접근성 개선

### ✅ 다음 할 일 (2026/02/03)
- [X] 타겟 lag 피처 구현 (autoregressive 특성)
- [X] LSTM 파이프라인 실험 및 성능 평가

---

## 📅 2026년 1월 30일

### 📂 작업 파일
```
note/preprocess/correlation.ipynb
note/preprocess/preprocess_casual_mask.ipynb
scripts/*.py
src/*.py
```

### 🔍 1. 데이터 간 상관 관계 분석

**결과 위치**: `results/correlation/*.png`

#### TMS Spearman 상관분석 결과

**강한 상관관계 (|r| > 0.4)**
- **SS_VU ↔ TP_VU**: 0.48 (중간 정도의 양의 상관)
  - 부유물질과 총인이 함께 증가하는 경향
- **PH_VU ↔ TP_VU**: -0.39 (중간 정도의 음의 상관)
  - pH가 높을수록 총인이 감소하는 경향

**약한~중간 상관관계 (0.2 < |r| < 0.4)**
- **PH_VU ↔ TN_VU**: -0.28
- **FLUX_VU ↔ TP_VU**: -0.26
- **PH_VU ↔ SS_VU**: -0.23

**매우 약한 상관관계 (|r| < 0.2)**
- **TOC_VU**: 다른 변수들과 거의 상관관계 없음 (-0.08 ~ 0.15)
- **FLUX_VU**: 대부분의 변수와 약한 상관관계
- 나머지 변수 쌍들도 대부분 매우 약한 상관관계

**주요 인사이트**
1. SS(부유물질)와 TP(총인)가 함께 증가하는 경향이 가장 뚜렷
2. PH가 높을수록 TP와 TN(총질소)이 감소하는 경향
3. TOC(총유기탄소)와 FLUX(유량)는 다른 수질 지표들과 독립적으로 작동
4. 전반적으로 변수 간 상관관계가 약해 **다중공선성 문제는 크지 않을 것으로 예상**

#### FLOW Spearman 상관분석 결과

**매우 강한 상관관계 (|r| > 0.7)**
- **flow_TankA ↔ flow_TankB**: 0.72 (강한 양의 상관)
  - 두 탱크의 유입량이 함께 변동하는 경향
- **level_TankA ↔ level_TankB**: 1.00 (완벽한 양의 상관)
  - 두 탱크의 수위가 동일하게 변동 (동일한 데이터일 가능성)

**중간 상관관계 (0.4 < |r| < 0.7)**
- **flow_TankA ↔ level_TankA**: 0.42
- **flow_TankA ↔ level_TankB**: 0.42
- **flow_TankB ↔ level_TankA**: 0.48
- **flow_TankB ↔ level_TankB**: 0.48

**주요 인사이트**
1. **TankA와 TankB의 수위가 완벽하게 일치** (r=1.00)
   - 두 탱크가 연결되어 있거나 동일한 센서 데이터일 가능성
2. **두 탱크의 유입량이 강하게 연동** (r=0.72)
   - 동일한 수원에서 유입되거나 유사한 패턴을 보임
3. **유입량과 수위 간 중간 정도의 양의 상관관계** (r=0.42~0.48)
   - 유입량이 증가하면 수위도 증가하는 경향이 있으나 완벽하지는 않음
   - 배출량이나 다른 요인도 수위에 영향을 미침

### 🔧 2. 결측치 보간 전략 수정

**변경 이유**: 기존 선형 보간은 미래값을 사용하여 데이터 누수 발생

**새로운 전략**:
- **단기 결측**: ffill (Forward Fill)
- **중기 결측**: 시간가중 EWMA / EMA
- **장기 결측**: NaN 유지

**마스크 추가**:
- `is_missing`: 원본 결측
- `imputed_ffill`: ffill로 보간
- `imputed_ewma`: EWMA로 보간
- `imputed_NaN`: 장기 결측로 결측 상태

### 🤖 3. 머신러닝 파이프라인 최적화

**주요 변경 사항**:
- `baseline.ipynb` → `baseline.py` (파이프라인 형태로 전환)
- 코드 리뷰 반영: 정규화, 이상치 처리, Optuna 최적화, TimeSeriesSplit, Feature Importance 기반 특성 선택
- 결측치 보간 전략 추가 (기존 baseline은 전부 드랍)

**전처리 파이프라인**:
1. 시간축 정합
2. 결측치 보간 (ffill/EWMA 전략)
3. 이상치 처리 (NaN 변환 후 재보간)
4. 리샘플링 (10분/1시간)
5. 파생 특성 생성 (rolling/lag/시간 특성)
6. 데이터 분할 (train/valid/test)
7. 모델 학습 및 평가

**특성 엔지니어링**: 모델별로 다르게 적용

### ✅ 다음 할 일 (2026/01/31)
- [X] 모델 실행하고 결과 확인

---

## 📅 2026년 1월 29일

### 📂 작업 파일
```
notebook/preprocess/feature/modelA.ipynb
notebook/preprocess/feature/modelB.ipynb
notebook/preprocess/feature/modelC.ipynb
notebook/preprocess/feature/modelFLOW.ipynb
notebook/DL/flow_lstm_model.py
scripts/*.py
src/*.py
```

### 🎯 1. 수질 모델 특성 엔지니어링 (계속)

#### 공통 문제 정의
- **예측 단위**: 원본 1분 시계열 → 5분 리샘플링
- **예측 목표**: 시점 t에서 t+h(5분 × h) 이후의 종속 변수 예측
- **핵심 제약**: 데이터 누수 방지 (종속 변수의 현재값은 입력 특성에 사용하지 않으며, 과거 정보만 사용)

#### 모델별 결과 파일
- **모델 A**: 유기물/입자 계열 (TOC/SS) → `data/processed/modelA_dataset.csv`
- **모델 B**: 영양염 계열 (TN/TP) → `data/processed/modelB_dataset.csv`
- **모델 C**: 공정 상태 계열 (FLUX/PH) → `data/processed/modelC_dataset.csv`
- **모델 FLOW**: 유입유량 (Q_in) → `data/processed/modelFLOW_dataset.csv`

### 🔄 2. 유입유량 모델 코드 리팩토링

**목표**: `.ipynb` → `.py` 변환 (코드 리뷰용)

**요구사항**:
- 파이프라인 활용
- main에 모든 코드를 넣지 않기

### 🧠 3. 유입유량 모델 딥러닝 (LSTM) 실험

**데이터**: `modelFLOW_dataset.csv` 사용

#### 실험 결과

| 실험 | 모델 구조 | Test R² | Test RMSE | 과적합 정도 | 학습 시간 |
|------|----------|---------|-----------|------------|----------|
| **실험 1** | 2층-64유닛 | **0.4032** ⭐ | 59.39 | 중간 (0.41) | 18 에포크 |
| **실험 2** | 4층-128유닛 | 0.0450 ❌ | 75.13 | 심각 (0.81) | 26 에포크 |
| **실험 3** | 3층-128유닛+정규화 | 0.3597 | 61.52 | 중간 (0.43) | 90 에포크 |

#### 성공 요인
- **실험 1**: 적절한 모델 복잡도로 가장 좋은 일반화 성능
- **실험 3**: 정규화로 학습 안정성 확보, Val-Test 일관성 개선

#### 실패 요인
- **실험 2**: 과도한 모델 복잡도로 심각한 과적합 발생
- **공통**: Train-Test 성능 격차 (시간적 분포 변화)

#### 🔧 즉시 개선 가능한 사항
1. **특성 선택**: 203개 → 50-100개로 축소
2. **시계열 교차 검증**: TimeSeriesSplit 적용
3. **실험 1 기반 미세 조정**: 드롭아웃 0.3, 배치 크기 64

### ✅ 다음 할 일 (2026/01/30)
- [x] LSTM 개선

---

## 📅 2026년 1월 28일

### 📂 작업 파일
```
notebook/preprocess/preprocess.ipynb
notebook/preprocess/show.ipynb
notebook/preprocess/feature/modelA.ipynb
```

### 🎯 TMS 모델 분리 전략

**문제점**: 하나의 모델로 TMS 지표 6개를 예측하기에는 성능이 낮음

**해결책**: 3개 모델로 분리

- **모델 A**: 유기물/입자 계열 (TOC + SS)
  - 유입/침전/생물 반응에서 같이 움직이는 경우가 많음
  - 강우/유량 이벤트에도 같은 영향을 받음
  - 노트북: `notebook/feature/modelA.ipynb`

- **모델 B**: 영양염 계열 (TN + TP)
  - 질소/인은 생물학적 영양염 제거 구간(BNR)에서 공정조건을 함께 공유
  - 제거 성능이 같이 흔들리는 경우가 많음
  - 노트북: `notebook/feature/modelB.ipynb`

- **모델 C**: 공정 상태 계열 (FLUX + pH)
  - pH는 생물반응과 연동
  - FLUX는 공정 부하/활성의 대표지표라서 상태로 같이 해석이 쉬움
  - 노트북: `notebook/feature/modelC.ipynb`
  - **참고**: pH는 변동 폭이 작고 센서 특성이 달라서 성능이 안 나오면 pH만 단독 모델로 분리 가능

### 📊 데이터 전처리 및 EDA

#### 1. 데이터 시각화 (`show.ipynb`)

**확인 항목**:
- 기초 통계량
- Boxplot
- Distribution
- 시계열 변화

**결과 위치**:
- `results/boxplot/*.png`
- `results/distribution/*.png`
- `results/timeseries/*.png`

#### 2. 결측치 처리 전략 수정

**기존 방식 (01/27)**: 선형 보간만 사용

**새로운 방식**: 구간 길이에 따라 다르게 처리
- **짧은 결측**: 선형 보간 (limit = 3)
- **중간 결측**: 스플라인 보간 (limit = 12)
- **남은 결측**: forward/backward fill

**결과 위치**: `results/preprocess/*.png` (처리 전후 비교)

### ✅ 다음 할 일 (2026/01/29)
- [X] Feature engineering 계속하기
- [X] 딥러닝 돌려보기

---

## 📅 2026년 1월 27일

### 📂 작업 파일
```
notebook/ML/primary/baseline.ipynb
notebook/ML/improved/v1/improved_baseline.py
notebook/ML/improved/v2/improved_v2_baseline.py
```

### 🔬 Baseline 모델 실험

#### 데이터 기간
```
flow: 2025/09/02 23:53:00 ~ 2025/12/03 10:39:00
tms:  2024/08/26 15:09:00 ~ 2025/09/29 05:23:00
aws:  2024/08/01 00:00:00 ~ 2026/01/25 05:09:00
```

#### 파이프라인
1. 가용한 데이터의 전체 기간 확인 (flow, tms, aws)
2. 결측치 제거
3. 1시간 단위로 리샘플링
4. 시간 특성 추가
   - 시간, 요일, 월, 주말 여부, 시간대 구분
   - 주기성 (sin/cos)
   - Lagging
   - 1시간/2시간/24시간 이동평균
5. 연속된 데이터인지 확인
6. Train/Valid/Test 분리 (0.6, 0.2, 0.2)
7. 모델 선택 (LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, HistGBR)
8. 성능 평가 (MAE, RMSE, MAPE, R²)
9. 성능 시각화

#### 결과
**결과 위치**: `notebook/ML/primary/baseline.ipynb` 하단

**성능 요약**:
- ✅ flow(Q_in), tms(FLUX_VU): 성능 좋음
- ❌ 나머지 변수: 성능 낮음

### 🚀 Improved Baseline V1

#### 개선 사항
1. 결측치 제거
2. StandardScaler 적용
3. GridSearchCV로 하이퍼파라미터 튜닝
4. 피처 선택 (중요도 기반)
5. TimeSeriesSplit 교차 검증
6. XGBoost 추가, 모델 개수 축소 (Ridge, Lasso, RandomForest, HistGBR)

**상세 문서**: `README_v1.md`

#### 결과
**결과 위치**:
- 보고서: `results/ML/v1/analysis_report.md`
- 그래프: `results/ML/v1/*.png`

**문제점**:
- 데이터 손실 95.8%: 13,848개 → 576개만 사용
- 심각한 과적합: Train R² 0.97 → Test R² -1.31
- 피처 부족: 수질 관련 직접 피처 없음

### 🎯 Improved Baseline V2

#### 개선 사항
1. 전처리 과정에서 1분 간격 확인 후 선형 보간으로 채우기
2. 도메인 피처 추가
3. 정규화 강화

**상세 문서**: `README_v2.md`

#### 결과
**결과 위치**:
- 보고서: `results/ML/v2/analysis_report.md`
- 그래프: `results/ML/v2/*.png`

**성능 개선**:

**Q_in 예측 극적 개선**
```
V1: R² 0.01 (거의 예측 불가)
V2: R² 0.56 (+6,213% 개선!) ✅
```

**일부 수질 변수 개선**
```
TN_VU:   R² -0.79 → 0.30  (+138%)
SS_VU:   R² -0.99 → 0.01  (+101%)
TP_VU:   R² -6.73 → -0.27 (+96%)
FLUX_VU: R² 0.96 유지 ✅
```

**데이터 활용도**
```
V1: 4.2% → V2: 94.3% (22배 증가)
```

**남은 문제점**:
- 종속 변수가 여러 개인 경우 성능이 낮음
- Q_in 혹은 FLUX_VU의 경우 머신러닝치고 꽤 괜찮은 성능

### ✅ 다음 할 일 (2026/01/28)
- [X] 데이터 EDA
- [] 딥러닝 모델 구상 및 구현

---

## 📅 2026년 1월 26일

### 📂 작업 파일
```
notebook/ML/linear/flow_baseline.ipynb
notebook/ML/linear/tms_baseline.ipynb
```

### 🔬 Sliding Window 기반 선형 모델 실험

#### 실험 설정
1. 데이터가 모든 시간 구간이 1분 단위로 나뉘어져 있는지 확인하고 빈 값이 있다면 보간으로 채워넣기
2. Sliding window 크기: 한 달 간격, step = 10분
3. 10분 간격으로 데이터 예측
4. 예측값은 실제값과 함께 시간 인덱스로 CSV 저장
   - `pred_Q_in_10min.csv`
   - `pred_tms_10min.csv`
5. 모델 성능 평가 (MAE, RMSE, MAPE, R²)
6. 시각화

#### 실험 결과

**Q_in (flow_TankA + flow_TankB)**
```
Window size: 30 day, Step: 10 min, Pred: 10 min
R²:        0.9100
RMSE:      23.02
MAE:       11.44
MAPE(%):   4584243.0
```

**TOC_VU**
```
Window size: 30 day, Step: 10 min, Pred: 10 min
R²:        -4.60
RMSE:      7.40
MAE:       6.13
MAPE(%):   2567921.5
```

**PH_VU**
```
Window size: 30 day, Step: 10 min, Pred: 10 min
R²:        -18.15
RMSE:      0.66
MAE:       0.11
MAPE(%):   21978.68
```

**SS_VU**
```
Window size: 30 day, Step: 10 min, Pred: 10 min
R²:        -5.31
RMSE:      2.63
MAE:       0.91
MAPE(%):   399477.41
```

**FLUX_VU**
```
Window size: 30 day, Step: 10 min, Pred: 10 min
R²:        0.9873
RMSE:      45201.34
MAE:       1516.18
MAPE(%):   11.97
```

**TN_VU**
```
Window size: 30 day, Step: 10 min, Pred: 10 min
R²:        -3.35
RMSE:      4.78
MAE:       0.90
MAPE(%):   7082133.0
```

**TP_VU**
```
Window size: 30 day, Step: 10 min, Pred: 10 min
R²:        -78.27
RMSE:      0.22
MAE:       0.18
MAPE(%):   7445272.5
```

#### 결론
- ✅ Q_in: 어느 정도 성능이 나옴
- ❌ TMS 지표: 기상 데이터만으로 6개의 종속변수를 예측하다보니 성능이 너무 낮음

### ✅ 다음 할 일 (2026/01/27)
- [X] 시간 특성(feature): 1시간, 2시간, 24시간
- [X] 이동 통계
- [X] 시간 변수 추가

---
