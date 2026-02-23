# 1. 데이터 분석
- [X] 데이터 수집
    - [X] 업체 데이터 (TMS_Actual.csv, FLOW_Actual.csv)
    - [X] 기상청 데이터 (기온, 습도, 강수량, 이슬점 온도)

- [X] 데이터 전처리
    - [X] 결측치 처리 (ffill/중기 EWMA/장기 EWMA 전략)
    - [X] 이상치 필터링 (도메인 지식 + IQR/Z-score)
    - [X] Feature engineering (시차 변수, 시간 특성)
    - [X] 시간축 정합 (정렬/중복 제거)
    - [X] 전처리 순서 적용 (정합→보간→이상치→리샘플링→피처→분할→스케일링→선택)

- [X] EDA
    - [X] 데이터 시각화
    - [X] 다변량 상관 분석
    - [X] 시간대/요일별 주기성 분석 (flow, ph)

# 2. 머신러닝
- [X] 베이스라인 (Linear, Ridge, Lasso, Elastic Net)
    - [X] 성능 평가 지표 설정 (MAE, RMSE, R2, MAPE)

- [X] 앙상블 (RandomForest, XGBoost, HistGBR)

- [X] 최종 통합 버전
    - [X] Optuna 하이퍼파라미터 최적화
    - [X] TimeSeriesSplit 교차 검증
    - [X] 피처 선택 (중요도 기반)
    - [X] TMS 모델 그룹화 (modelA, modelB, modelC)
    - [X] 데이터 누수 방지
    - [X] 도메인 특화 피처 (강수, 기상, TMS 상호작용)
    - [X] Learning Curve 시각화

# 3. 딥러닝
- [X] LSTM_FLOW (R2: 0.7899)
    - [X] 30분 리샘플링
    - [X] Multi-head Attention (8 heads)
    - [X] Target lag 피처 추가
    - [X] 시간 특성 추가 (hour×weekday, weekday, iso_week)
    - [X] 모델 저장 및 평가

- [X] LSTM_TMS
    - [X] Target lag 피처 추가 (lag/rolling/diff/EWMA)
    - [X] Early stop 기능 구현
    - [X] 이상치 처리 수정 (배출허용기준 2배)
    - [X] FLUX 차분 처리 (누적값 → 차분)
    - [X] TN (R2: 0.8062)
    - [X] PH (R2: 0.7490)
    - [X] SS (R2: 0.3548)
    - [X] TOC (R2: 0.2759)
    - [X] FLUX (R2: 0.2251)
    - [ ] TP 성능 개선 (현재 R2: -0.0601)
    - [ ] TOC/SS/FLUX early stop 제거 재학습

# 4. 프로젝트 관리
- [X] 의존성 관리 (requirements.txt)
- [X] 문서화 (QUICK_START.md, README.md)
- [X] FastAPI 백엔드 연동 (main.py)
    - [X] 전처리 파이프라인 노트북과 통일
    - [X] 예측 시간해상도 30분 수정
    - [X] WebClient HTTP/1.1 호환

# 5. 향후 개선 사항
- [ ] TP 성능 향상
- [ ] TOC/SS/FLUX early stop 없이 재학습
- [ ] 모니터링 대시보드
- [ ] 자동 재학습 파이프라인
- [ ] 앙상블 모델
- [ ] 계절성 분석

# 6. 보고서
