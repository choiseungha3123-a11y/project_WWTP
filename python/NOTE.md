2026 / 01 / 26
notebook/linear_regression/flow_baseline.ipynb
notebook/linear_regression/tms_baseline.ipynb

1. 데이터가 모든 시간 구간이 1분 단위로 나뉘어져 있는지 확인하고 만약 빈 값이 있다면 보간으로 채워넣기 
2. sliding window의 크기를 한 달 간격으로 조절 및 step = 10분
3. 10분 간격으로 데이터를 예측 
4. 예측값은 실제값과 함께 시간 인덱스로 csv로 저장(pred_Q_in_10min.csv, pred_tms_10min.csv)
5. 모델 성능 평가(MAE, RMSE, MAPE, R2) 
6. 시각화

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

2026 / 01 / 27 해야 할 일
시간 특성(feature): 1시간, 2시간, 24시간
이동 통계
시간 변수 추가