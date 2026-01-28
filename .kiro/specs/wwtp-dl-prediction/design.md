# 설계 문서: 하수처리장 딥러닝 예측 시스템

## 개요

본 설계는 LSTM(Long Short-Term Memory) 신경망을 사용하여 하수처리장의 유입유량(Q_in)과 수질 지표(TMS)를 예측하는 딥러닝 시스템을 정의합니다. 시스템은 PyTorch를 기반으로 구현되며, 슬라이딩 윈도우 방식으로 시계열 데이터를 처리하여 시간적 의존성을 학습합니다.

### 설계 목표

- Q_in 예측 정확도: R² ≥ 0.95
- TMS 지표 예측 정확도: R² ≥ 0.90
- 모듈화된 아키텍처로 유지보수성 확보
- GPU 가속 지원으로 학습 시간 단축
- 재현 가능한 실험 환경 제공

## 아키텍처

시스템은 다음 주요 컴포넌트로 구성됩니다:

```
┌─────────────────────────────────────────────────────────────┐
│                    WWTP DL Prediction System                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │ Data Loader  │─────▶│   Dataset    │─────▶│  Model    │ │
│  │              │      │   Builder    │      │  Trainer  │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                     │                     │        │
│         │                     │                     ▼        │
│         │                     │              ┌───────────┐  │
│         │                     │              │   LSTM    │  │
│         │                     │              │   Model   │  │
│         │                     │              └───────────┘  │
│         │                     │                     │        │
│         ▼                     ▼                     ▼        │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │ Preprocessor │      │   Scaler     │      │ Evaluator │ │
│  │              │      │   Manager    │      │           │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│                                                      │        │
│                                                      ▼        │
│                                               ┌───────────┐  │
│                                               │Visualizer │  │
│                                               │           │  │
│                                               └───────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 컴포넌트 설명

1. **Data Loader**: CSV 파일에서 전처리된 데이터를 로드
2. **Preprocessor**: 데이터 검증 및 정규화
3. **Dataset Builder**: 슬라이딩 윈도우 시퀀스 생성 및 train/val/test 분할
4. **Scaler Manager**: 정규화 스케일러 관리 및 역변환
5. **LSTM Model**: PyTorch 기반 LSTM 신경망
6. **Model Trainer**: 학습 루프, 조기 종료, 체크포인트 관리
7. **Evaluator**: 성능 메트릭 계산
8. **Visualizer**: 학습 곡선 및 예측 결과 시각화

## 컴포넌트 및 인터페이스

### 1. Data Loader

**책임**: 전처리된 CSV 파일을 pandas DataFrame으로 로드

**인터페이스**:
```python
class DataLoader:
    def load_flow_data(self) -> pd.DataFrame:
        """유입유량 데이터 로드"""
        pass
    
    def load_tms_data(self) -> pd.DataFrame:
        """TMS 수질 데이터 로드"""
        pass
    
    def load_all_data(self) -> pd.DataFrame:
        """통합 데이터 로드"""
        pass
    
    def validate_columns(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """필수 컬럼 존재 여부 검증"""
        pass
```

**구현 세부사항**:
- `pandas.read_csv()`를 사용하여 데이터 로드
- 타임스탬프 컬럼을 datetime 타입으로 파싱
- 누락된 파일에 대해 `FileNotFoundError` 발생
- 누락된 컬럼에 대해 `ValueError` 발생

### 2. Preprocessor

**책임**: 데이터 정규화 및 검증

**인터페이스**:
```python
class Preprocessor:
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
    
    def fit_scaler(self, data: np.ndarray, name: str) -> None:
        """스케일러 학습"""
        pass
    
    def transform(self, data: np.ndarray, name: str) -> np.ndarray:
        """데이터 정규화"""
        pass
    
    def inverse_transform(self, data: np.ndarray, name: str) -> np.ndarray:
        """정규화 역변환"""
        pass
    
    def check_nan_inf(self, data: np.ndarray) -> bool:
        """NaN 또는 무한값 검사"""
        pass
    
    def save_scalers(self, path: str) -> None:
        """스케일러 저장"""
        pass
    
    def load_scalers(self, path: str) -> None:
        """스케일러 로드"""
        pass
```

**구현 세부사항**:
- `sklearn.preprocessing.StandardScaler` 사용
- 각 피처 세트(입력, 타겟)에 대해 별도 스케일러 유지
- 스케일러는 학습 데이터에만 fit
- `pickle`을 사용하여 스케일러 직렬화

### 3. Dataset Builder

**책임**: 슬라이딩 윈도우 시퀀스 생성 및 데이터 분할

**인터페이스**:
```python
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int):
        """
        Args:
            X: 입력 피처 (samples, features)
            y: 타겟 값 (samples,)
            window_size: 슬라이딩 윈도우 크기
        """
        pass
    
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """idx 위치의 (시퀀스, 타겟) 반환"""
        pass

class DatasetBuilder:
    def create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        window_size: int
    ) -> TimeSeriesDataset:
        """슬라이딩 윈도우 시퀀스 생성"""
        pass
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """시계열 데이터를 train/val/test로 분할"""
        pass
```

**구현 세부사항**:
- 슬라이딩 윈도우: `X[i:i+window_size]` → `y[i+window_size]`
- 시간 순서 보존: 분할 시 셔플링 없음
- train/val/test 순차 분할 (시간적 누수 방지)
- PyTorch Dataset 인터페이스 구현

### 4. LSTM Model

**책임**: LSTM 기반 시계열 예측 모델

**인터페이스**:
```python
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Args:
            input_size: 입력 피처 수
            hidden_size: LSTM 은닉 유닛 수
            num_layers: LSTM 레이어 수
            dropout: 드롭아웃 비율
            output_size: 출력 크기 (기본 1: 단일 타겟)
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            (batch_size, output_size)
        """
        pass
```

**아키텍처 세부사항**:
```
Input: (batch, seq_len, input_size)
  ↓
LSTM Layer 1: (batch, seq_len, hidden_size)
  ↓
Dropout
  ↓
LSTM Layer 2: (batch, seq_len, hidden_size)
  ↓
Dropout
  ↓
...
  ↓
LSTM Layer N: (batch, seq_len, hidden_size)
  ↓
Take last time step: (batch, hidden_size)
  ↓
Fully Connected: (batch, output_size)
  ↓
Output: (batch, output_size)
```

**하이퍼파라미터 권장값**:
- `hidden_size`: 64-128
- `num_layers`: 2-3
- `dropout`: 0.2-0.3
- `window_size`: 24-48 (시간 단위에 따라)

### 5. Model Trainer

**책임**: 모델 학습, 검증, 조기 종료, 체크포인트 관리

**인터페이스**:
```python
class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        patience: int = 10
    ):
        pass
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """한 에폭 학습"""
        pass
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """검증 세트 평가"""
        pass
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        criterion: nn.Module,
        save_path: str
    ) -> Dict[str, List[float]]:
        """전체 학습 루프 (조기 종료 포함)"""
        pass
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        path: str,
        metadata: Dict
    ) -> None:
        """체크포인트 저장"""
        pass
    
    def load_checkpoint(self, path: str) -> Dict:
        """체크포인트 로드"""
        pass
```

**구현 세부사항**:
- 손실 함수: `nn.MSELoss()` 또는 `nn.L1Loss()`
- 옵티마이저: `torch.optim.Adam()`
- 조기 종료: 검증 손실이 `patience` 에폭 동안 개선되지 않으면 중단
- 최적 모델: 검증 손실이 최소일 때 저장
- 체크포인트 내용: 모델 state_dict, 옵티마이저 state_dict, 에폭, 손실, 메타데이터

### 6. Evaluator

**책임**: 모델 성능 평가 메트릭 계산

**인터페이스**:
```python
class Evaluator:
    @staticmethod
    def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² 점수 계산"""
        pass
    
    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """평균 절대 오차 계산"""
        pass
    
    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """평균 제곱근 오차 계산"""
        pass
    
    @staticmethod
    def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """평균 절대 백분율 오차 계산"""
        pass
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        scaler: StandardScaler
    ) -> Dict[str, float]:
        """모든 메트릭 계산"""
        pass
```

**구현 세부사항**:
- `sklearn.metrics` 함수 활용
- 예측값을 원래 스케일로 역변환 후 메트릭 계산
- MAPE 계산 시 0으로 나누기 방지

### 7. Visualizer

**책임**: 학습 곡선 및 예측 결과 시각화

**인터페이스**:
```python
class Visualizer:
    @staticmethod
    def plot_training_history(
        train_losses: List[float],
        val_losses: List[float],
        save_path: str
    ) -> None:
        """학습/검증 손실 곡선 플롯"""
        pass
    
    @staticmethod
    def plot_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str,
        save_path: str
    ) -> None:
        """실제값 vs 예측값 플롯"""
        pass
    
    @staticmethod
    def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str
    ) -> None:
        """잔차 플롯"""
        pass
```

**구현 세부사항**:
- `matplotlib` 사용
- 한글 폰트 설정 (맑은 고딕 등)
- 고해상도 이미지 저장 (dpi=300)
- 그리드, 레이블, 범례 포함

## 데이터 모델

### 입력 데이터 구조

```python
# 원본 데이터 (pandas DataFrame)
flow_data: pd.DataFrame
    - columns: ['timestamp', 'Q_in', 'temperature', 'humidity', 'precipitation', 'tank_level', ...]
    - shape: (N, num_features)

tms_data: pd.DataFrame
    - columns: ['timestamp', 'TOC_VU', 'PH_VU', 'SS_VU', 'FLUX_VU', 'TN_VU', 'TP_VU', ...]
    - shape: (M, num_features)

# 전처리 후 데이터 (numpy array)
X: np.ndarray  # 입력 피처
    - shape: (N, num_input_features)
    
y: np.ndarray  # 타겟 변수
    - shape: (N,)

# 슬라이딩 윈도우 시퀀스 (torch.Tensor)
X_seq: torch.Tensor
    - shape: (N - window_size + 1, window_size, num_input_features)
    
y_seq: torch.Tensor
    - shape: (N - window_size + 1,)
```

### 모델 체크포인트 구조

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': int,
    'train_loss': float,
    'val_loss': float,
    'metadata': {
        'target_variable': str,  # 'Q_in', 'TOC_VU', etc.
        'window_size': int,
        'hidden_size': int,
        'num_layers': int,
        'dropout': float,
        'learning_rate': float,
        'batch_size': int,
        'train_date': str,
        'num_input_features': int
    }
}
```

### 평가 결과 구조

```python
evaluation_results = {
    'target': str,
    'r2_score': float,
    'mae': float,
    'rmse': float,
    'mape': float,
    'num_test_samples': int
}
```


## 정확성 속성 (Correctness Properties)

속성(Property)은 시스템의 모든 유효한 실행에서 참이어야 하는 특성 또는 동작입니다. 본질적으로 시스템이 무엇을 해야 하는지에 대한 형식적 진술입니다. 속성은 사람이 읽을 수 있는 명세와 기계가 검증 가능한 정확성 보장 사이의 다리 역할을 합니다.

### Property 1: 데이터 로딩 일관성

*모든* 유효한 CSV 파일에 대해, 파일을 로드하면 원본 데이터와 동일한 행 수와 컬럼을 가진 DataFrame이 반환되어야 한다.

**검증: 요구사항 1.1**

### Property 2: 파일 누락 오류 처리

*모든* 존재하지 않는 파일 경로에 대해, 데이터 로드 시도는 설명적인 오류 메시지와 함께 실패해야 한다.

**검증: 요구사항 1.5**

### Property 3: 컬럼 검증 정확성

*모든* DataFrame과 필수 컬럼 리스트에 대해, 검증 함수는 모든 필수 컬럼이 존재할 때만 True를 반환해야 한다.

**검증: 요구사항 1.6**

### Property 4: 슬라이딩 윈도우 시퀀스 형태

*모든* 데이터셋 (N개 샘플, F개 피처)과 윈도우 크기 W에 대해, 생성된 시퀀스는 형태 (N - W + 1, W, F)를 가져야 하며, 각 시퀀스는 W개의 연속된 시간 단계를 포함해야 한다.

**검증: 요구사항 2.1, 2.2, 2.4**

### Property 5: 시퀀스-타겟 쌍 일관성

*모든* 생성된 시퀀스에 대해, i번째 시퀀스의 타겟 값은 원본 데이터의 (i + window_size)번째 값과 일치해야 한다.

**검증: 요구사항 2.3**

### Property 6: 시간적 순서 보존

*모든* 슬라이딩 윈도우 시퀀스에 대해, 시퀀스 내 시간 단계의 순서는 원본 데이터의 순서와 동일해야 한다.

**검증: 요구사항 2.5**

### Property 7: 정규화 속성

*모든* 정규화된 데이터에 대해, StandardScaler로 변환된 데이터는 평균이 0에 가깝고 표준편차가 1에 가까워야 한다 (수치 오차 범위 내).

**검증: 요구사항 2.6**

### Property 8: 시간적 분할 무결성

*모든* 시계열 데이터 분할에 대해, 학습 세트의 모든 타임스탬프는 검증 세트의 모든 타임스탬프보다 이전이어야 하고, 검증 세트의 모든 타임스탬프는 테스트 세트의 모든 타임스탬프보다 이전이어야 하며, 세 세트 간에 겹치는 샘플이 없어야 한다.

**검증: 요구사항 3.1, 3.3, 3.4, 3.5**

### Property 9: 분할 비율 정확성

*모든* 분할 비율 (train_ratio, val_ratio, test_ratio)에 대해, 실제 분할된 데이터의 크기는 지정된 비율의 ±1 샘플 이내여야 한다 (반올림 오차 고려).

**검증: 요구사항 3.2**

### Property 10: 모델 입력 차원 처리

*모든* 입력 차원 수에 대해, LSTM 모델은 해당 차원의 입력을 받아 올바른 형태의 출력을 생성해야 한다.

**검증: 요구사항 4.1, 4.5**

### Property 11: 다층 LSTM 지원

*모든* 레이어 수 (1 ≤ num_layers ≤ 5)에 대해, 모델은 지정된 수의 LSTM 레이어로 초기화되고 정상적으로 forward pass를 수행해야 한다.

**검증: 요구사항 4.3**

### Property 12: 스케일 역변환 라운드트립

*모든* 원본 데이터에 대해, 정규화 후 역변환하면 원래 값과 동일해야 한다 (수치 오차 범위 내).

**검증: 요구사항 7.6**

### Property 13: 모델 저장-로드 라운드트립

*모든* 학습된 모델에 대해, 모델을 저장하고 로드한 후 동일한 입력에 대해 동일한 예측을 생성해야 한다 (수치 오차 범위 내).

**검증: 요구사항 6.6, 12.3**

## 오류 처리

### 데이터 로딩 오류

- **FileNotFoundError**: 지정된 CSV 파일이 존재하지 않을 때
  - 메시지: "Data file not found: {file_path}"
  
- **ValueError**: 필수 컬럼이 누락되었을 때
  - 메시지: "Missing required columns: {missing_columns}"

- **ValueError**: 데이터에 NaN 또는 무한값이 포함되었을 때
  - 메시지: "Data contains NaN or infinite values in columns: {columns}"

### 모델 학습 오류

- **ValueError**: 하이퍼파라미터가 유효 범위를 벗어났을 때
  - 예: window_size < 1, hidden_size < 1, dropout < 0 or > 1
  - 메시지: "Invalid hyperparameter: {param_name} = {value}"

- **RuntimeError**: GPU 메모리 부족
  - 메시지: "CUDA out of memory. Try reducing batch_size or model size."
  - 대응: CPU로 자동 전환 또는 배치 크기 감소 제안

- **RuntimeError**: 학습 중 손실이 NaN이 되었을 때
  - 메시지: "Training loss became NaN. Try reducing learning_rate."
  - 대응: 학습 중단 및 마지막 유효 체크포인트 보존

### 모델 로딩 오류

- **FileNotFoundError**: 모델 파일이 존재하지 않을 때
  - 메시지: "Model checkpoint not found: {checkpoint_path}"

- **RuntimeError**: 모델 파일이 손상되었을 때
  - 메시지: "Failed to load model checkpoint: {error_details}"

- **ValueError**: 모델 아키텍처가 불일치할 때
  - 메시지: "Model architecture mismatch. Expected input_size={expected}, got {actual}"

## 테스트 전략

### 이중 테스트 접근법

시스템의 정확성을 보장하기 위해 단위 테스트와 속성 기반 테스트를 모두 사용합니다:

- **단위 테스트**: 특정 예제, 엣지 케이스, 오류 조건 검증
- **속성 테스트**: 모든 입력에 대한 보편적 속성 검증

두 접근법은 상호 보완적이며 포괄적인 커버리지를 위해 모두 필요합니다.

### 단위 테스트 범위

단위 테스트는 다음에 집중합니다:

1. **특정 예제**:
   - 알려진 입력/출력 쌍으로 메트릭 계산 검증
   - 특정 파일 경로 로딩 테스트
   - 모델 아키텍처 구성 요소 검증

2. **엣지 케이스**:
   - 빈 데이터셋 처리
   - 단일 샘플 데이터셋
   - 윈도우 크기가 데이터 길이와 같을 때
   - 극단적인 하이퍼파라미터 값

3. **오류 조건**:
   - 누락된 파일 처리
   - 잘못된 데이터 형식
   - 유효하지 않은 하이퍼파라미터
   - GPU 메모리 부족 시나리오

4. **통합 지점**:
   - 컴포넌트 간 데이터 흐름
   - 학습 파이프라인 전체 실행
   - 모델 저장 및 로드 워크플로우

### 속성 기반 테스트 설정

**테스트 라이브러리**: Hypothesis (Python)

**설정**:
- 각 속성 테스트는 최소 100회 반복 실행
- 각 테스트는 설계 문서의 속성을 참조하는 태그 포함
- 태그 형식: `# Feature: wwtp-dl-prediction, Property {number}: {property_text}`

**예제**:
```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(
    data=npst.arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(50, 200), st.integers(5, 20))
    ),
    window_size=st.integers(10, 30)
)
def test_sliding_window_shape(data, window_size):
    """
    Feature: wwtp-dl-prediction, Property 4: 슬라이딩 윈도우 시퀀스 형태
    
    모든 데이터셋과 윈도우 크기에 대해, 생성된 시퀀스는 
    올바른 형태를 가져야 한다.
    """
    N, F = data.shape
    if window_size >= N:
        return  # Skip invalid combinations
    
    X_seq, y_seq = create_sequences(data[:, :-1], data[:, -1], window_size)
    
    expected_samples = N - window_size + 1
    assert X_seq.shape == (expected_samples, window_size, F - 1)
    assert y_seq.shape == (expected_samples,)
```

### 테스트 조직

```
python/tests/
├── unit/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_dataset_builder.py
│   ├── test_lstm_model.py
│   ├── test_trainer.py
│   ├── test_evaluator.py
│   └── test_visualizer.py
├── property/
│   ├── test_data_properties.py
│   ├── test_sequence_properties.py
│   ├── test_split_properties.py
│   ├── test_model_properties.py
│   └── test_roundtrip_properties.py
└── integration/
    ├── test_training_pipeline.py
    └── test_prediction_pipeline.py
```

### 성능 목표 검증

성능 목표 (R² ≥ 0.95 for Q_in, R² ≥ 0.90 for TMS)는 통합 테스트에서 검증합니다:

```python
def test_qin_prediction_accuracy():
    """Q_in 예측 정확도가 목표를 달성하는지 검증"""
    # 전체 파이프라인 실행
    model = train_model(target='Q_in', ...)
    metrics = evaluate_model(model, test_data)
    
    # 성능 목표 확인
    assert metrics['r2_score'] >= 0.95, \
        f"Q_in R² score {metrics['r2_score']} below target 0.95"
```

이러한 테스트는 속성 테스트가 아니라 전체 시스템 검증이므로 별도로 관리합니다.

### 지속적 통합

- 모든 테스트는 코드 변경 시 자동 실행
- 속성 테스트는 더 많은 반복으로 야간 빌드에서 실행 (예: 1000회)
- 성능 회귀 모니터링을 위한 벤치마크 테스트
