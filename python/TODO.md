# 1. 데이터 분석
- [X] 데이터 수집
    - [X] 업체 데이터(TMS_Actual.csv, FLOW_Actual.csv)
    - [X] 기상청 데이터(기온, 습도, 강수량, 이슬점 온도)

- [X] 데이터 전처리
    - [X] 결측치 처리(보간) - V2에서 선형 보간 사용
    - [] 이상치 필터링
    - [X] Feature engineering(시차 변수, 시간 특성) - V1/V2에서 구현

- [] 탐색적 데이터 분석(EDA)
    - [] 데이터 시각화 - Learning curves, R² comparison
    - [] 다변량 상관 분석 - Feature importance 분석

# 2. 머신러닝
- [X] 베이스 라인 (V1 완료)
    - [X] 다중 회귀 분석 (Ridge, Lasso)
    - [X] 성능 평가 지표 설정(MAE, RMSE, R2, MAPE)

- [X] 앙상블 (V1 완료)
    - [X] 랜덤 포레스트
    - [X] XGBoost
    - [X] HistGradientBoosting

- [X] 개선 버전 (V2 완료)
    - [X] 결측치 보간 (Linear)
    - [X] 도메인 피처 추가 (강수량, 온도-습도, 탱크 수위, 수질 비율, 차분)
    - [X] 정규화 강화 (Ridge alpha 증가, RF max_depth 제한)
    - [X] 피처 수 감소 (과적합 방지)
    - [X] V2 실행 및 결과 분석

# 3. 딥러닝
- [] LSTM

# 4. 보고서