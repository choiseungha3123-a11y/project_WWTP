# 1. 데이터 분석
- [X] 데이터 수집
    - [X] 업체 데이터(TMS_Actual.csv, FLOW_Actual.csv)
    - [X] 기상청 데이터(기온, 습도, 강수량, 이슬점 온도)

- [X] 데이터 전처리
    - [X] 결측치 처리(보간) - 전략적 보간 (ffill/EWMA/NaN) 구현
    - [X] 이상치 필터링 - 도메인 지식 + 통계적 방법 (IQR/Z-score)
    - [X] Feature engineering(시차 변수, 시간 특성) - V1/V2에서 구현
    - [X] 시간축 정합 (정렬/중복 제거)
    - [X] 올바른 전처리 순서 적용 (정합→보간→이상치→리샘플링→피처→분할→스케일링→선택)

- [X] 탐색적 데이터 분석(EDA)
    - [X] 데이터 시각화
    - [X] 다변량 상관 분석

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

- [X] 최종 통합 버전 (2026-01-30 완료)
    - [X] 코드 통합 (train.py, pipeline.py, models.py)
    - [X] GridSearchCV → Optuna 전환
    - [X] TimeSeriesSplit 교차 검증
    - [X] XGBoost Early Stopping
    - [X] 피처 선택 (중요도 기반)
    - [X] 스케일링 (StandardScaler)
    - [X] TMS 모델 그룹화 (modelA, modelB, modelC)
    - [X] 데이터 누수 방지
    - [X] 도메인 특화 피처 (강수, 기상, TMS 상호작용)
    - [X] Learning Curve 시각화
    - [X] ModelA 특화 특성 추가 (100개 이상)
    - [X] ModelB 특화 특성 추가 (160개 이상)
    - [X] ModelC 특화 특성 추가 (170개 이상)
    - [X] ModelFLOW 특화 특성 추가 (165개 이상)

# 3. 딥러닝
- [X] LSTM (notebook/DL/flow_lstm_model.py 구현 완료)
    - [X] 시퀀스 데이터 처리
    - [X] 모델 학습 및 평가
    - [X] 결과 시각화

# 4. 프로젝트 관리
- [X] 의존성 관리
    - [X] requirements.txt 업데이트 (optuna, xgboost 추가)

- [X] 문서화
    - [X] QUICK_START.md 작성
    - [X] README.md 업데이트

# 5. 향후 개선 사항
- [] 실시간 예측 API 구축
- [] 모델 배포 (Docker, FastAPI)
- [] 모니터링 대시보드
- [] A/B 테스트 프레임워크
- [] 자동 재학습 파이프라인
- [] 앙상블 모델 (여러 모델 조합)
- [] 추가 도메인 피처 탐색
- [] 계절성 분석 및 반영

# 6. 보고서