"""
하수처리장 딥러닝 예측 시스템

이 패키지는 하수처리장 유입량(Q_in)과 수질 지표(TMS)를 예측하기 위한
LSTM 기반 모델을 포함합니다.
"""

__version__ = "0.1.0"
__author__ = "WWTP DL Team"

# 쉬운 접근을 위한 설정 import
from .config import (
    # 하이퍼파라미터
    LSTM_CONFIG,
    TRAINING_CONFIG,
    WINDOW_SIZE,
    SPLIT_RATIOS,
    
    # 경로
    FLOW_DATA_PATH,
    TMS_DATA_PATH,
    ALL_DATA_PATH,
    MODEL_SAVE_DIR,
    RESULTS_SAVE_DIR,
    
    # 타겟 변수
    FLOW_TARGET,
    TMS_TARGETS,
    
    # 유틸리티 함수
    create_directories,
    get_model_path,
    get_scaler_path,
    get_plot_path,
    get_metrics_path,
    validate_config,
)

__all__ = [
    # 하이퍼파라미터
    "LSTM_CONFIG",
    "TRAINING_CONFIG",
    "WINDOW_SIZE",
    "SPLIT_RATIOS",
    
    # 경로
    "FLOW_DATA_PATH",
    "TMS_DATA_PATH",
    "ALL_DATA_PATH",
    "MODEL_SAVE_DIR",
    "RESULTS_SAVE_DIR",
    
    # 타겟 변수
    "FLOW_TARGET",
    "TMS_TARGETS",
    
    # 유틸리티 함수
    "create_directories",
    "get_model_path",
    "get_scaler_path",
    "get_plot_path",
    "get_metrics_path",
    "validate_config",
]
