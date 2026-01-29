### 2026 / 01 / 29

```
notebook/preprocess/feature/modelA.ipynb
notebook/preprocess/feature/modelB.ipynb
notebook/preprocess/feature/modelC.ipynb
notebook/preprocess/feature/modelFLOW.ipynb
notebook/DL/flow_lstm_model.py
scripts/*.py
src/*.py
```

#### 1. 어제 진행하던 수질 모델 특성 엔지니어링 계속하는 중

#### 공통 문제 정의

- 예측 단위: 원본 1분 시계열 -> 5분 리샘플링
- 예측 목표: 시점 t에서 t+h(5분 * h) 이후의 종속 변수 예측
- 핵심 제약: 데이터 누수 방지(종속 변수의 현재값은 입력 특성에 사용하지 않으며, 과거 정보만 사용)

- 모델 A: 유기물/입자 계열(TOC/SS) 특성 엔지니어링 결과 => data/processed/modelA_dataset.csv
- 모델 B: 영양염 계열(TN/TP) 특성 엔지니어링 결과 => data/processed/modelB_dataset.csv
- 모델 C: 공정 사태 계열(FLUX/PH) 특성 엔지니어링 결과 => data/processed/modelC_dataset.csv
- 모델 FLOW: 유입유량(Q_in) 특성 엔지니어링 결과 => data/processed/modelFLOW_dataset.csv

#### 2. 유입유량 모델의 경우 .ipynb -> .py로 수정해야 함(코드 리뷰를 위해)
- 파이프라인을 활용하여 변경
- main에 무식하게 다 넣지 않기

#### 3. 유입유량 모델 딥러닝 LSTM 시도 - modelFLOW_dataset.csv 사용

| 실험 | 모델 구조 | Test R² | Test RMSE | 과적합 정도 | 학습 시간 |
|-----|----------|---------|-----------|-----------|----------|
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

#### 2026 / 01 / 30 해야 할 일
LSTM 개선

---

### 2026 / 01 / 28

```
notebook/preprocess/preprocess.ipynb
notebook/preprocess/show.ipynb
notebook/preprocess/feature/modelA.ipynb
```

#### TMS 관련

- 하나의 모델로 TMS 지표 6개를 예측하기에는 성능이 낮은 현상이 계속 나타남

-> 이를 해결하기 위해 TMS 지표 6개를 한 모델로 예측하는 것이 아닌 3-모델을 사용

    - 모델 A: 유기물/입자 계열(TOC + SS, 유입/침전/생물 반응에서 같이 움직이는 경우가 많고 강우/유량 이벤트에도 같은 영향을 받음) => notebook/feature/modelA.ipynb
    - 모델 B: 영양염 계열(TN + TP, 질소/인은 생물학적 영양염 제거 구간(BNR)에서 공정조건을 함께 공유해서 제거 성능이 같이 흔들리는 경우가 많음) => notebook/feature/modelB.ipynb
    - 모델 C: 공정 상태 계열(FLUX + pH, pH는 생물반응과 연동되고, FLUX는 공정 부하/활성의 대표지표라서 상태로 같이 해석이 쉬움) => notebook/feature/modelC.ipynb
    - pH는 변동 폭이 작고 센서 특성이 달라서 성능이 안 나오면 pH만 단독 모델로 빼도 될 것 같음

#### preprocess
1. 데이터 형태를 확인하기 위해서 show.ipynb에 데이터들의 기초 통계량, boxplot, distribution, 시계열 변화를 확인

-> 자세한 그래프는 results/boxplot/*.png, results/distribution/*.png, results/timeseries/*.png에서 확인 가능

2. 01 / 27 에 사용한 데이터는 가장 기본적인 결측치에서 선형 보간을 사용하였는데 구간 길이에 따라 나눠서 결측치 처리를 해야 한다고 판단하여 수정

-> 짧은 결측: 선형 보간(limit = 3) / 중간 결측: 스플라인 보간(limit = 12) / 남은 결측: forward/backward fill

-> 결측치 처리 전후 그래프는 results/preprocess/*.png에서 확인 가능

#### 2026 / 01 / 29 해야 할 일

feature engineering 하던 거 계속하기

딥러닝 돌려보기

---
### 2026 / 01 / 27
```
notebook/ML/primary/baseline.ipynb
notebook/ML/improved/v1/improved_baseline.py
notebook/ML/improved/v2/improved_v2_baseline.py
```
#### baseline
1. 가용한 데이터의 전체 기간 확인(flow, tms, asw)
2. 결측치 제거
3. 1시간 단위로 리샘플링
4. 시간 특성(시간, 요일, 월, 주말 여부, 시간대 구분, 주기성(sin/cos)), lagging, 1시간/2시간/24시간 이동평균 컬럼 추가
5. 연속된 데이터인지 확인
6. train, valid, test(0.6, 0.2, 0.2) 분리
7. 모델 선택(LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, HistGBR)
8. 성능 평가(MAE, RMSE, MAPE, R2)
9. 성능 시각화
```
flow: 2025/09/02 23:53:00 ~ 2025/12/03 10:39:00
tms: 2024/08/26 15:09:00 ~ 2025/09/29 05:23:00
aws: 2024/08/01 00:00:00 ~ 2026/01/25 05:09:00
```
#### 결과: notebook/ML/primary/baseline.ipynb 하단 확인

-> flow(Q_in), tms(FLUX_VU) 종속 변수만 성능이 좋음, 나머지 변수는 여전히 성능 낮음

#### improved_baseline
1. 결측치 제거
2. StandardScaler 적용
3. GridSearchCV로 하이퍼파라미터 튜닝
4. 피처 선택 (중요도 기반)
5. TimeSeriesSplit 교차 검증
6. XGBoost 추가, 모델 개수 축소(Ridge, Lasso, RandomForest, HistGBR)
- 자세한 내용은 README_v1.md 확인

#### 결과
- 보고서(results/ML/v1/analysis_report.md)
- 그래프(results/ML/v1/*.png)

-> 여전히 이전 문제가 존재
```
    - 데이터 손실 95.8%: 13,848개 → 576개만 사용
    - 심각한 과적합: Train R² 0.97 → Test R² -1.31
    - 피처 부족: 수질 관련 직접 피처 없음
```
#### improved_baseline_v2
1. 전처리 과정에서 1분 간격 확인 후 선형 보간으로 채우기
2. 도메인 피처 추가
3. 정규화 강화
- 자세한 내용은 README_v2.md 확인

#### 결과
```
Q_in 예측 극적 개선
V1: R² 0.01 (거의 예측 불가)
V2: R² 0.56 (+6,213% 개선!) ✅
```
#### 일부 수질 변수 개선
```
TN_VU: R² -0.79 → 0.30 (+138%)
SS_VU: R² -0.99 → 0.01 (+101%)
TP_VU: R² -6.73 → -0.27 (+96%)
FLUX_VU: R² 0.96 유지 ✅
```
#### 데이터 활용도
V1: 4.2% → V2: 94.3% (22배 증가)

- 보고서(results/ML/v2/analysis_report.md)
- 그래프(results/ML/v2/*.png)

-> 여전히 종속 변수가 여러 개인 경우 성능이 낮음, 대신 Q_in 혹은 FLUX_VU의 경우 머신러닝치고 꽤 괜찮은 성능을 보임

#### 2027 / 01 / 28 해야 할 일

데이터 EDA

딥러닝 모델 구상 및 구현

---

### 2026 / 01 / 26
```
notebook/ML/linear/flow_baseline.ipynb
notebook/ML/linear/tms_baseline.ipynb
```
1. 데이터가 모든 시간 구간이 1분 단위로 나뉘어져 있는지 확인하고 만약 빈 값이 있다면 보간으로 채워넣기 
2. sliding window의 크기를 한 달 간격으로 조절 및 step = 10분
3. 10분 간격으로 데이터를 예측 
4. 예측값은 실제값과 함께 시간 인덱스로 csv로 저장(pred_Q_in_10min.csv, pred_tms_10min.csv)
5. 모델 성능 평가(MAE, RMSE, MAPE, R2) 
6. 시각화
```
    Q_in(flow_TankA + flow_TankB) - window size(30 day), step(10 min), pred(10 min)
    R2: 0.9100146889686584 
    RMSE: 23.021405418042466 
    MAE: 11.438867568969727 
    MAPE(%): 4584243.0 

    TOC_VU - window size(30 day), step(10 min), pred(10 min)
    R2: -4.596914768218994 
    RMSE: 7.397290182125764 
    MAE: 6.130698204040527 
    MAPE(%): 2567921.5

    PH_VU - window size(30 day), step(10 min), pred(10 min)
    R2: -18.15496253967285 
    RMSE: 0.6590679384805752 
    MAE: 0.10965807735919952 
    MAPE(%): 21978.6796875
    
    SS_VU - window size(30 day), step(10 min), pred(10 min)
    R2: -5.308851718902588 
    RMSE: 2.6265916086339236 
    MAE: 0.9070245623588562 
    MAPE(%): 399477.40625
    
    FLUX_VU - window size(30 day), step(10 min), pred(10 min)
    R2: 0.9873223900794983 
    RMSE: 45201.33520151811 
    MAE: 1516.1761474609375 
    MAPE(%): 11.972585678100586
    
    TN_VU - window size(30 day), step(10 min), pred(10 min)
    R2: -3.3522963523864746 
    RMSE: 4.784890777074766 
    MAE: 0.8995549082756042 
    MAPE(%): 7082133.0
    
    TP_VU - window size(30 day), step(10 min), pred(10 min)
    R2: -78.27156066894531 
    RMSE: 0.22165860483351152 
    MAE: 0.1849478781223297 
    MAPE(%): 7445272.5
```
    -> Q_in의 경우 어느 정도 성능이 나오지만 TMS 지표는 기상 데이터만으로 6개의 종속변수를 예측하다보니 성능이 너무 낮게 나왔다.

#### 2026 / 01 / 27 해야 할 일

시간 특성(feature): 1시간, 2시간, 24시간

이동 통계

시간 변수 추가