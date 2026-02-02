"""
딥러닝 모델 정의 모듈 (PyTorch)
LSTM 및 기타 시계열 딥러닝 모델 구현
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Tuple, Optional, List
import warnings


# ============================================================================
# LSTM 모델 클래스
# ============================================================================

class FlowLSTM(nn.Module):
    """
    시계열 예측을 위한 LSTM 모델
    
    구조:
        - 드롭아웃이 적용된 LSTM 레이어
        - 배치 정규화
        - 완전 연결 출력 레이어 (2층 구조)
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 output_size: int, 
                 dropout: float = 0.2):
        """
        FlowLSTM 모델을 초기화합니다.
        
        Parameters:
        -----------
        input_size : int
            입력 특성 개수
        hidden_size : int
            LSTM 은닉층 유닛 수
        num_layers : int
            쌓인 LSTM 레이어 수
        output_size : int
            출력 특성 개수
        dropout : float
            정규화를 위한 드롭아웃 비율
        """
        super(FlowLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 배치 정규화 (과적합 방지 및 학습 안정화)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout)
        
        # 완전 연결 레이어 (2층 구조)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        네트워크를 통한 순전파를 수행합니다.
        
        Parameters:
        -----------
        x : torch.Tensor
            입력 텐서, 크기 (batch_size, sequence_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            출력 텐서, 크기 (batch_size, output_size)
        """
        # 은닉 상태와 셀 상태 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 순전파
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 시간 스텝의 출력 사용
        out = out[:, -1, :]
        
        # 배치 정규화 적용
        out = self.batch_norm(out)
        
        # 드롭아웃 적용
        out = self.dropout(out)
        
        # 완전 연결 레이어 (2층 구조)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


# ============================================================================
# LSTM 래퍼 클래스 (sklearn 호환)
# ============================================================================

class LSTMWrapper:
    """
    sklearn 스타일 인터페이스를 제공하는 LSTM 래퍼
    
    ML 모델과 동일한 인터페이스로 사용 가능
    """
    
    def __init__(self,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 num_epochs: int = 100,
                 patience: int = 10,
                 weight_decay: float = 0.0001,
                 grad_clip: Optional[float] = 1.0,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Parameters:
        -----------
        hidden_size : int
            LSTM 은닉층 유닛 수
        num_layers : int
            LSTM 레이어 수
        dropout : float
            드롭아웃 비율
        batch_size : int
            배치 크기
        learning_rate : float
            학습률
        num_epochs : int
            최대 에포크 수
        patience : int
            조기 종료 patience
        weight_decay : float
            L2 정규화 계수
        grad_clip : float or None
            그래디언트 클리핑 값
        random_state : int
            랜덤 시드
        verbose : bool
            학습 과정 출력 여부
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.random_state = random_state
        self.verbose = verbose
        
        # 학습 후 설정되는 속성
        self.model_: Optional[FlowLSTM] = None
        self.X_scaler_: Optional[StandardScaler] = None
        self.y_scaler_: Optional[StandardScaler] = None
        self.device_: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history_: Dict[str, List[float]] = {}
        self.input_size_: Optional[int] = None
        self.output_size_: Optional[int] = None
        
        # 랜덤 시드 설정
        self._set_random_seed()
    
    def _set_random_seed(self):
        """재현성을 위한 랜덤 시드 설정"""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _prepare_data(self, 
                     X: np.ndarray, 
                     y: np.ndarray,
                     fit_scaler: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        데이터 정규화 및 텐서 변환
        
        Parameters:
        -----------
        X : np.ndarray
            입력 특성 (samples, sequence_length, features)
        y : np.ndarray
            타겟 변수 (samples, targets)
        fit_scaler : bool
            스케일러를 fit할지 여부
            
        Returns:
        --------
        X_tensor, y_tensor : torch.Tensor
        """
        # 3D 입력 확인
        if X.ndim != 3:
            raise ValueError(f"X는 3D 배열이어야 합니다 (samples, sequence_length, features). 현재: {X.shape}")
        
        # 2D로 reshape하여 스케일링
        n_samples, seq_len, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        
        if fit_scaler:
            self.X_scaler_ = StandardScaler()
            X_scaled_2d = self.X_scaler_.fit_transform(X_2d)
        else:
            if self.X_scaler_ is None:
                raise ValueError("스케일러가 fit되지 않았습니다.")
            X_scaled_2d = self.X_scaler_.transform(X_2d)
        
        # 다시 3D로 reshape
        X_scaled = X_scaled_2d.reshape(n_samples, seq_len, n_features)
        
        # y 스케일링
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if fit_scaler:
            self.y_scaler_ = StandardScaler()
            y_scaled = self.y_scaler_.fit_transform(y)
        else:
            if self.y_scaler_ is None:
                raise ValueError("스케일러가 fit되지 않았습니다.")
            y_scaled = self.y_scaler_.transform(y)
        
        # 텐서로 변환
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        return X_tensor, y_tensor
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'LSTMWrapper':
        """
        LSTM 모델 학습
        
        Parameters:
        -----------
        X : np.ndarray
            학습 입력 특성 (samples, sequence_length, features)
        y : np.ndarray
            학습 타겟 변수 (samples, targets)
        X_val : np.ndarray, optional
            검증 입력 특성
        y_val : np.ndarray, optional
            검증 타겟 변수
            
        Returns:
        --------
        self
        """
        # 입력/출력 크기 저장
        self.input_size_ = X.shape[2]
        self.output_size_ = y.shape[1] if y.ndim > 1 else 1
        
        # 데이터 준비
        X_train_tensor, y_train_tensor = self._prepare_data(X, y, fit_scaler=True)
        
        # DataLoader 생성
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 검증 데이터 준비
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val, fit_scaler=False)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 모델 초기화
        self.model_ = FlowLSTM(
            input_size=self.input_size_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size_,
            dropout=self.dropout
        ).to(self.device_)
        
        if self.verbose:
            print(f"\n모델 구조:")
            print(self.model_)
            print(f"전체 파라미터 수: {sum(p.numel() for p in self.model_.parameters()):,}")
            print(f"디바이스: {self.device_}")
        
        # 손실 함수 및 옵티마이저
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 학습
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        if self.verbose:
            print(f"\n{self.num_epochs} 에포크 동안 학습을 시작합니다...")
            print(f"조기 종료 patience: {self.patience}")
            if self.grad_clip:
                print(f"그래디언트 클리핑: {self.grad_clip}")
        
        for epoch in range(self.num_epochs):
            # 학습 단계
            self.model_.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)
                
                # 순전파
                outputs = self.model_(X_batch)
                loss = criterion(outputs, y_batch)
                
                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 검증 단계
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device_)
                        y_batch = y_batch.to(self.device_)
                        
                        outputs = self.model_(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # 진행 상황 출력
                if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                    print(f"Epoch [{epoch+1}/{self.num_epochs}] - "
                          f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                # 조기 종료 체크
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = self.model_.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"\n에포크 {epoch+1}에서 조기 종료 발동")
                        print(f"최고 검증 손실: {best_val_loss:.6f}")
                    break
            else:
                # 검증 데이터가 없으면 학습 손실만 출력
                if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                    print(f"Epoch [{epoch+1}/{self.num_epochs}] - Train Loss: {avg_train_loss:.6f}")
        
        # 최고 성능 모델 로드
        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)
        
        # 학습 히스토리 저장
        self.history_ = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss if val_loader is not None else None
        }
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측 수행
        
        Parameters:
        -----------
        X : np.ndarray
            입력 특성 (samples, sequence_length, features)
            
        Returns:
        --------
        np.ndarray
            예측값 (원래 스케일)
        """
        if self.model_ is None:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        
        # 데이터 준비
        X_tensor, _ = self._prepare_data(X, np.zeros((X.shape[0], self.output_size_)), fit_scaler=False)
        
        # 예측
        self.model_.eval()
        predictions = []
        
        with torch.no_grad():
            # 배치 단위로 예측
            for i in range(0, len(X_tensor), self.batch_size):
                X_batch = X_tensor[i:i+self.batch_size].to(self.device_)
                outputs = self.model_(X_batch)
                predictions.extend(outputs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # 원래 스케일로 역변환
        predictions_original = self.y_scaler_.inverse_transform(predictions)
        
        return predictions_original
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        R² 스코어 계산 (sklearn 호환)
        
        Parameters:
        -----------
        X : np.ndarray
            입력 특성
        y : np.ndarray
            실제 타겟 변수
            
        Returns:
        --------
        float
            R² 스코어
        """
        y_pred = self.predict(X)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        return r2_score(y, y_pred)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """파라미터 반환 (sklearn 호환)"""
        return {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'weight_decay': self.weight_decay,
            'grad_clip': self.grad_clip,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params: Any) -> 'LSTMWrapper':
        """파라미터 설정 (sklearn 호환)"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ============================================================================
# 딥러닝 모델 Zoo
# ============================================================================

def build_dl_model_zoo(random_state: int = 42, 
                       verbose: bool = True) -> Dict[str, Any]:
    """
    딥러닝 모델 Zoo 생성
    
    Parameters:
    -----------
    random_state : int
        랜덤 시드
    verbose : bool
        학습 과정 출력 여부
        
    Returns:
    --------
    dict
        모델 딕셔너리
    """
    zoo = {
        "LSTM": LSTMWrapper(
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_size=32,
            learning_rate=0.001,
            num_epochs=100,
            patience=10,
            weight_decay=0.0001,
            grad_clip=1.0,
            random_state=random_state,
            verbose=verbose
        ),
    }
    return zoo
