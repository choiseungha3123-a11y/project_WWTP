# 유입 유량 LSTM 모델 (y_Q_in 예측)

## 개요
시계열 데이터로부터 총 유입량 `y_Q_in` (`y_flowA`와 `y_flowB`의 합)을 예측하는 PyTorch LSTM 모델입니다.

## 모델 구조

### 1. 패키지 Import
- PyTorch: 딥러닝
- NumPy, Pandas: 데이터 처리
- Scikit-learn: 전처리 및 평가 지표
- Matplotlib: 시각화

### 2. Configuration 설정
- **데이터 경로**: `data/processed/modelFLOW_dataset.csv`
- **시퀀스 길이**: 24 시간 스텝 (config.py에서 설정)
- **LSTM 은닉층 크기**: 64 유닛
- **LSTM 레이어 수**: 2개 (쌓인 구조)
- **드롭아웃**: 0.2
- **배치 크기**: 32
- **학습률**: 0.001
- **최대 에포크**: 100
- **조기 종료 Patience**: 10

### 3. 랜덤 시드
- 재현성을 위해 42로 설정
- PyTorch, NumPy, CUDA에 적용

### 4. 데이터 로드 (`load_data()`)
- CSV 데이터셋 로드
- 타겟 생성: `y_Q_in = y_flowA + y_flowB`
- 특성(X)과 타겟(y) 분리
- 특성에서 제외: SYS_TIME, y_flowA, y_flowB, y_Q_in

### 5. 데이터 분할
- **학습**: 60%
- **검증**: 20%
- **테스트**: 20%
- 시계열 분할 (셔플 없음)

### 6. FlowLSTM 클래스
```python
class FlowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout)
    def forward(self, x)
```
- 드롭아웃이 적용된 다층 LSTM
- 완전 연결 출력 레이어
- y_Q_in 예측값 반환

### 7. Evaluate 함수
- 계산 지표: Loss, MSE, RMSE, MAE, R²
- 예측값을 원래 스케일로 역변환
- 평가 지표 딕셔너리 반환

### 8. 메인 실행
- 데이터 로드 및 전처리
- LSTM용 시퀀스 생성
- 조기 종료를 적용한 모델 학습
- train/val/test 세트에서 평가
- 모델, 스케일러, 플롯 저장

## 사용법

### 학습 실행
```bash
cd notebook/DL
python flow_lstm_model.py
```

### 출력 파일
- `model/flow_lstm_model.pth` - 학습된 모델
- `model/X_scaler.pkl` - 특성 스케일러
- `model/y_scaler.pkl` - 타겟 스케일러
- `results/DL/training_history.png` - 손실 곡선
- `results/DL/test_predictions.png` - 예측 플롯

## 모델 성능 지표
모델이 보고하는 지표:
- **Loss**: MSE 손실 값
- **RMSE**: 평균 제곱근 오차
- **MAE**: 평균 절대 오차
- **R²**: 결정 계수

## 필요 라이브러리
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## 참고사항
- GPU가 사용 가능하면 GPU 사용, 그렇지 않으면 CPU 사용
- 조기 종료로 과적합 방지
- 슬라이딩 윈도우 방식으로 시퀀스 생성
- StandardScaler를 사용하여 데이터 정규화
