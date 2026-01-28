# 요구사항 문서

## 소개

본 문서는 하수처리장(WWTP) 운영을 위한 딥러닝 기반 예측 시스템의 요구사항을 명시합니다. 이 시스템은 LSTM 신경망을 사용하여 유입유량(Q_in)과 수질 지표(TMS: TOC_VU, PH_VU, SS_VU, FLUX_VU, TN_VU, TP_VU)를 예측하며, 시계열 데이터의 시간적 의존성을 학습합니다.

## 용어 정의

- **WWTP_System**: 하수처리장 예측 시스템
- **LSTM_Model**: Long Short-Term Memory 신경망 모델
- **Q_in**: 하수처리장 유입유량 (m³/day 등)
- **TMS**: 원격 모니터링 시스템 수질 지표
- **TOC_VU**: 총유기탄소 값
- **PH_VU**: pH 값
- **SS_VU**: 부유물질 값
- **FLUX_VU**: 플럭스 값
- **TN_VU**: 총질소 값
- **TP_VU**: 총인 값
- **Sliding_Window**: 시계열 데이터를 이동하는 고정 크기 시간 윈도우
- **Preprocessed_Data**: 정제, 보간, 피처 엔지니어링이 완료된 데이터
- **Training_Pipeline**: 모델 학습, 검증, 평가의 전체 프로세스
- **Prediction_Service**: 학습된 모델로부터 예측을 생성하는 컴포넌트

## 요구사항

### 요구사항 1: 데이터 로딩 및 준비

**사용자 스토리:** 데이터 과학자로서, CSV 파일에서 전처리된 데이터를 로드하여 딥러닝 모델 학습을 위한 데이터셋을 준비하고 싶습니다.

#### 인수 기준

1. WHEN 전처리된 CSV 파일이 제공되면, THE WWTP_System SHALL 구조화된 데이터셋으로 메모리에 로드한다
2. WHEN 유량 데이터를 로드할 때, THE WWTP_System SHALL `python/data/processed/flow_proc.csv`에서 읽는다
3. WHEN TMS 데이터를 로드할 때, THE WWTP_System SHALL `python/data/processed/tms_proc.csv`에서 읽는다
4. WHEN 통합 데이터를 로드할 때, THE WWTP_System SHALL `python/data/processed/all_proc.csv`에서 읽는다
5. IF 필수 데이터 파일이 없으면, THEN THE WWTP_System SHALL 설명적인 오류 메시지를 반환한다
6. WHEN 데이터가 로드되면, THE WWTP_System SHALL 필수 컬럼이 존재하는지 검증한다

### 요구사항 2: 슬라이딩 윈도우 데이터셋 생성

**사용자 스토리:** 데이터 과학자로서, 시계열 데이터로부터 슬라이딩 윈도우 시퀀스를 생성하여 LSTM 모델이 시간적 패턴을 학습할 수 있도록 하고 싶습니다.

#### 인수 기준

1. WHEN 시퀀스를 생성할 때, THE WWTP_System SHALL 설정 가능한 윈도우 크기의 입력 시퀀스를 생성한다
2. WHEN 윈도우 크기가 지정되면, THE WWTP_System SHALL 각 입력이 해당 개수의 연속된 시간 단계를 포함하는 시퀀스를 생성한다
3. WHEN 시퀀스를 생성할 때, THE WWTP_System SHALL 각 입력 시퀀스를 해당하는 타겟 값과 쌍으로 만든다
4. WHEN 데이터셋에 N개 샘플이 있고 윈도우 크기가 W일 때, THE WWTP_System SHALL (N - W + 1)개의 시퀀스를 생성한다
5. WHEN 시퀀스를 생성할 때, THE WWTP_System SHALL 시간적 순서를 보존한다
6. WHEN 시퀀스가 생성되면, THE WWTP_System SHALL 적합된 스케일러를 사용하여 입력 피처를 정규화한다

### 요구사항 3: 학습-검증-테스트 분할

**사용자 스토리:** 데이터 과학자로서, 데이터를 학습, 검증, 테스트 세트로 분할하여 모델 성능을 적절히 평가하고 싶습니다.

#### 인수 기준

1. WHEN 시계열 데이터를 분할할 때, THE WWTP_System SHALL 시간적 순서를 유지한다
2. WHEN 분할 비율이 지정되면, THE WWTP_System SHALL 비율에 따라 데이터를 할당한다
3. THE WWTP_System SHALL 시간적 누수 없이 학습, 검증, 테스트 세트를 생성한다
4. WHEN 데이터를 분할할 때, THE WWTP_System SHALL 검증 및 테스트 세트가 학습 세트 대비 미래 데이터만 포함하도록 보장한다
5. WHEN 분할이 완료되면, THE WWTP_System SHALL 겹치지 않는 세 개의 데이터셋을 반환한다

### 요구사항 4: LSTM 모델 아키텍처

**사용자 스토리:** 데이터 과학자로서, LSTM 신경망 아키텍처를 정의하여 하수 데이터의 시간적 의존성을 포착하고 싶습니다.

#### 인수 기준

1. THE LSTM_Model SHALL 다차원 시계열 입력 시퀀스를 받는다
2. THE LSTM_Model SHALL 설정 가능한 은닉 유닛을 가진 최소 하나의 LSTM 레이어를 포함한다
3. THE LSTM_Model SHALL 여러 개의 적층된 LSTM 레이어를 지원한다
4. THE LSTM_Model SHALL 정규화를 위한 드롭아웃 레이어를 포함한다
5. THE LSTM_Model SHALL 타겟 변수(Q_in 또는 TMS 지표)에 대한 예측을 출력한다
6. WHEN 초기화될 때, THE LSTM_Model SHALL 레이어 크기, 드롭아웃 비율, 레이어 수에 대한 하이퍼파라미터를 받는다
7. THE LSTM_Model SHALL 회귀 작업에 적합한 활성화 함수를 사용한다

### 요구사항 5: 모델 학습

**사용자 스토리:** 데이터 과학자로서, 과거 데이터로 LSTM 모델을 학습시켜 미래 값을 예측할 수 있도록 하고 싶습니다.

#### 인수 기준

1. WHEN 학습이 시작되면, THE Training_Pipeline SHALL 학습 데이터셋을 사용한다
2. WHEN 학습할 때, THE Training_Pipeline SHALL 역전파를 사용하여 모델 파라미터를 최적화한다
3. THE Training_Pipeline SHALL 손실 함수로 평균 제곱 오차(MSE) 또는 평균 절대 오차(MAE)를 사용한다
4. THE Training_Pipeline SHALL 적절한 옵티마이저(Adam, RMSprop, 또는 SGD)를 사용한다
5. WHEN 각 에폭이 완료되면, THE Training_Pipeline SHALL 검증 세트에서 성능을 평가한다
6. WHEN 학습할 때, THE Training_Pipeline SHALL 설정 가능한 배치 크기를 지원한다
7. WHEN 학습할 때, THE Training_Pipeline SHALL 설정 가능한 에폭 수를 지원한다
8. WHEN 학습할 때, THE Training_Pipeline SHALL 설정 가능한 학습률을 지원한다

### 요구사항 6: 조기 종료 및 모델 체크포인트

**사용자 스토리:** 데이터 과학자로서, 과적합을 방지하고 최적의 모델을 저장하여 예측에 사용하고 싶습니다.

#### 인수 기준

1. WHEN 검증 손실이 개선을 멈추면, THE Training_Pipeline SHALL 학습을 조기에 중단한다
2. WHEN 조기 종료가 활성화되면, THE Training_Pipeline SHALL 설정 가능한 인내 기간 동안 검증 손실을 모니터링한다
3. WHEN 새로운 최적 검증 손실이 달성되면, THE Training_Pipeline SHALL 모델 체크포인트를 저장한다
4. WHEN 학습이 완료되면, THE Training_Pipeline SHALL 최적 모델 가중치를 복원한다
5. WHEN 체크포인트를 저장할 때, THE Training_Pipeline SHALL `python/model/` 디렉토리에 모델을 저장한다
6. WHEN 체크포인트를 저장할 때, THE Training_Pipeline SHALL 모델 아키텍처, 가중치, 학습 메타데이터를 포함한다

### 요구사항 7: 모델 평가

**사용자 스토리:** 데이터 과학자로서, 테스트 데이터에서 모델 성능을 평가하여 예측 정확도를 평가하고 싶습니다.

#### 인수 기준

1. WHEN 평가가 시작되면, THE WWTP_System SHALL 테스트 데이터셋을 사용한다
2. WHEN 평가할 때, THE WWTP_System SHALL R²(결정계수) 점수를 계산한다
3. WHEN 평가할 때, THE WWTP_System SHALL 평균 절대 오차(MAE)를 계산한다
4. WHEN 평가할 때, THE WWTP_System SHALL 평균 제곱근 오차(RMSE)를 계산한다
5. WHEN 평가할 때, THE WWTP_System SHALL 평균 절대 백분율 오차(MAPE)를 계산한다
6. WHEN 예측이 생성되면, THE WWTP_System SHALL 정규화된 예측을 원래 스케일로 역변환한다
7. WHEN 평가가 완료되면, THE WWTP_System SHALL 계산된 모든 메트릭을 반환한다

### 요구사항 8: 예측 생성

**사용자 스토리:** 시스템 운영자로서, 미래 시간 단계에 대한 예측을 생성하여 유입량 및 수질 조건을 예측하고 싶습니다.

#### 인수 기준

1. WHEN 학습된 모델이 로드되면, THE Prediction_Service SHALL 입력 시퀀스를 받는다
2. WHEN 예측을 생성할 때, THE Prediction_Service SHALL 학습 데이터와 동일한 전처리를 적용한다
3. WHEN 예측이 만들어지면, THE Prediction_Service SHALL 원래 스케일의 값을 반환한다
4. WHEN Q_in을 예측할 때, THE Prediction_Service SHALL 테스트 데이터에서 R² ≥ 0.95를 달성한다
5. WHEN TMS 지표를 예측할 때, THE Prediction_Service SHALL 테스트 데이터에서 R² ≥ 0.90을 달성한다
6. WHEN 예측이 생성되면, THE Prediction_Service SHALL 배치 예측을 효율적으로 처리한다

### 요구사항 9: 다중 타겟 예측

**사용자 스토리:** 데이터 과학자로서, 각 타겟 변수에 대해 별도의 모델을 학습시켜 각 예측 작업의 성능을 최적화하고 싶습니다.

#### 인수 기준

1. THE WWTP_System SHALL Q_in 예측을 위한 개별 모델 학습을 지원한다
2. THE WWTP_System SHALL 각 TMS 지표(TOC_VU, PH_VU, SS_VU, FLUX_VU, TN_VU, TP_VU)에 대한 개별 모델 학습을 지원한다
3. WHEN 여러 모델을 학습할 때, THE WWTP_System SHALL 별도의 모델 인스턴스를 관리한다
4. WHEN 여러 모델을 학습할 때, THE WWTP_System SHALL 각 모델을 설명적인 식별자와 함께 저장한다
5. WHEN 모델을 로드할 때, THE WWTP_System SHALL 타겟 변수로 모델을 식별한다

### 요구사항 10: 결과 시각화 및 로깅

**사용자 스토리:** 데이터 과학자로서, 학습 진행 상황과 예측 결과를 시각화하여 모델 동작을 이해하고 싶습니다.

#### 인수 기준

1. WHEN 학습이 진행되면, THE WWTP_System SHALL 각 에폭의 학습 및 검증 손실을 로그한다
2. WHEN 학습이 완료되면, THE WWTP_System SHALL 학습 이력을 보여주는 손실 곡선을 생성한다
3. WHEN 예측이 만들어지면, THE WWTP_System SHALL 예측값과 실제값을 비교하는 플롯을 생성한다
4. WHEN 결과를 저장할 때, THE WWTP_System SHALL `python/results/DL/` 디렉토리에 플롯을 저장한다
5. WHEN 결과를 저장할 때, THE WWTP_System SHALL 평가 메트릭을 구조화된 형식(CSV 또는 JSON)으로 저장한다
6. WHEN 시각화를 생성할 때, THE WWTP_System SHALL 적절한 레이블, 제목, 범례를 포함한다

### 요구사항 11: 하이퍼파라미터 설정

**사용자 스토리:** 데이터 과학자로서, 모델 하이퍼파라미터를 설정하여 다양한 아키텍처와 학습 설정을 실험하고 싶습니다.

#### 인수 기준

1. THE WWTP_System SHALL 슬라이딩 윈도우의 윈도우 크기 설정을 지원한다
2. THE WWTP_System SHALL LSTM 은닉 레이어 크기 설정을 지원한다
3. THE WWTP_System SHALL LSTM 레이어 수 설정을 지원한다
4. THE WWTP_System SHALL 드롭아웃 비율 설정을 지원한다
5. THE WWTP_System SHALL 배치 크기 설정을 지원한다
6. THE WWTP_System SHALL 학습률 설정을 지원한다
7. THE WWTP_System SHALL 학습 에폭 수 설정을 지원한다
8. THE WWTP_System SHALL 조기 종료 인내 설정을 지원한다
9. WHEN 설정이 제공되면, THE WWTP_System SHALL 파라미터 범위를 검증한다

### 요구사항 12: 모델 저장 및 로딩

**사용자 스토리:** 시스템 운영자로서, 학습된 모델을 저장하고 로드하여 재학습 없이 모델을 재사용하고 싶습니다.

#### 인수 기준

1. WHEN 모델이 학습되면, THE WWTP_System SHALL 모델을 디스크에 저장한다
2. WHEN 모델을 저장할 때, THE WWTP_System SHALL PyTorch의 표준 직렬화 형식을 사용한다
3. WHEN 모델을 로드할 때, THE WWTP_System SHALL 정확한 아키텍처와 가중치를 복원한다
4. WHEN 모델을 로드할 때, THE WWTP_System SHALL 관련 스케일러와 전처리 파라미터도 로드한다
5. IF 모델 파일이 손상되었거나 없으면, THEN THE WWTP_System SHALL 설명적인 오류 메시지를 반환한다
6. WHEN 모델을 저장할 때, THE WWTP_System SHALL 메타데이터(학습 날짜, 하이퍼파라미터, 성능 메트릭)를 포함한다

### 요구사항 13: GPU 가속 지원

**사용자 스토리:** 데이터 과학자로서, 가능한 경우 GPU 가속을 활용하여 모델을 더 빠르게 학습시키고 싶습니다.

#### 인수 기준

1. WHEN 초기화할 때, THE WWTP_System SHALL 사용 가능한 GPU 장치를 감지한다
2. WHEN GPU가 사용 가능하면, THE WWTP_System SHALL 모델과 데이터를 GPU 메모리로 이동한다
3. WHEN GPU가 사용 불가능하면, THE WWTP_System SHALL CPU 연산으로 대체한다
4. WHEN GPU에서 학습할 때, THE WWTP_System SHALL 메모리 관리를 적절히 처리한다
5. WHEN 장치를 전환할 때, THE WWTP_System SHALL 데이터와 모델이 동일한 장치에 있도록 보장한다

### 요구사항 14: 오류 처리 및 검증

**사용자 스토리:** 개발자로서, 문제가 발생했을 때 시스템이 명확한 피드백을 제공하도록 강력한 오류 처리를 원합니다.

#### 인수 기준

1. WHEN 잘못된 입력 데이터가 제공되면, THE WWTP_System SHALL 설명적인 오류를 발생시킨다
2. WHEN 필수 컬럼이 데이터에서 누락되면, THE WWTP_System SHALL 어떤 컬럼이 누락되었는지 식별한다
3. WHEN 하이퍼파라미터가 유효 범위를 벗어나면, THE WWTP_System SHALL 오류 메시지와 함께 설정을 거부한다
4. WHEN 파일 I/O 작업이 실패하면, THE WWTP_System SHALL 명확한 오류 메시지를 제공한다
5. WHEN 데이터에서 NaN 또는 무한 값이 감지되면, THE WWTP_System SHALL 적절히 처리하거나 오류를 발생시킨다
6. WHEN 모델 학습이 실패하면, THE WWTP_System SHALL 오류를 로그하고 부분 결과를 보존한다
