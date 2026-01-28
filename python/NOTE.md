2026 / 01 / 28

```
notebook/preprocess/
notebook/DL/
```

2026 / 01 / 27
```
notebook/ML/primary/baseline.ipynb
notebook/ML/improved/v1/improved_baseline.py
notebook/ML/improved/v2/improved_v2_baseline.py
```
baseline
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
결과: notebook/ML/primary/baseline.ipynb 하단 확인

-> flow(Q_in), tms(FLUX_VU) 종속 변수만 성능이 좋음, 나머지 변수는 여전히 성능 낮음

improved_baseline
1. 결측치 제거
2. StandardScaler 적용
3. GridSearchCV로 하이퍼파라미터 튜닝
4. 피처 선택 (중요도 기반)
5. TimeSeriesSplit 교차 검증
6. XGBoost 추가, 모델 개수 축소(Ridge, Lasso, RandomForest, HistGBR)
- 자세한 내용은 README_v1.md 확인

결과
- 보고서(results/ML/v1/analysis_report.md)
- 그래프(results/ML/v1/*.png)

-> 여전히 이전 문제가 존재
```
    - 데이터 손실 95.8%: 13,848개 → 576개만 사용
    - 심각한 과적합: Train R² 0.97 → Test R² -1.31
    - 피처 부족: 수질 관련 직접 피처 없음
```
improved_baseline_v2
1. 전처리 과정에서 1분 간격 확인 후 선형 보간으로 채우기
2. 도메인 피처 추가
3. 정규화 강화
- 자세한 내용은 README_v2.md 확인

결과
```
Q_in 예측 극적 개선
V1: R² 0.01 (거의 예측 불가)
V2: R² 0.56 (+6,213% 개선!) ✅
```
일부 수질 변수 개선
```
TN_VU: R² -0.79 → 0.30 (+138%)
SS_VU: R² -0.99 → 0.01 (+101%)
TP_VU: R² -6.73 → -0.27 (+96%)
FLUX_VU: R² 0.96 유지 ✅
```
데이터 활용도
V1: 4.2% → V2: 94.3% (22배 증가)

- 보고서(results/ML/v2/analysis_report.md)
- 그래프(results/ML/v2/*.png)

-> 여전히 종속 변수가 여러 개인 경우 성능이 낮음, 대신 Q_in 혹은 FLUX_VU의 경우 머신러닝치고 꽤 괜찮은 성능을 보임

2027 / 01 / 28 해야 할 일

데이터 EDA
딥러닝 모델 구상 및 구현

2026 / 01 / 26
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

2026 / 01 / 27 해야 할 일

시간 특성(feature): 1시간, 2시간, 24시간

이동 통계

시간 변수 추가