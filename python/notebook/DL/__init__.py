"""
WWTP Deep Learning Prediction System

This package contains LSTM-based models for predicting wastewater treatment
plant inflow (Q_in) and water quality indicators (TMS).
"""

__version__ = "0.1.0"
__author__ = "WWTP DL Team"

# Import configuration for easy access
from .config import (
    # Hyperparameters
    LSTM_CONFIG,
    TRAINING_CONFIG,
    WINDOW_SIZE,
    SPLIT_RATIOS,
    
    # Paths
    FLOW_DATA_PATH,
    TMS_DATA_PATH,
    ALL_DATA_PATH,
    MODEL_SAVE_DIR,
    RESULTS_SAVE_DIR,
    
    # Target variables
    FLOW_TARGET,
    TMS_TARGETS,
    
    # Utility functions
    create_directories,
    get_model_path,
    get_scaler_path,
    get_plot_path,
    get_metrics_path,
    validate_config,
)

__all__ = [
    # Hyperparameters
    "LSTM_CONFIG",
    "TRAINING_CONFIG",
    "WINDOW_SIZE",
    "SPLIT_RATIOS",
    
    # Paths
    "FLOW_DATA_PATH",
    "TMS_DATA_PATH",
    "ALL_DATA_PATH",
    "MODEL_SAVE_DIR",
    "RESULTS_SAVE_DIR",
    
    # Target variables
    "FLOW_TARGET",
    "TMS_TARGETS",
    
    # Utility functions
    "create_directories",
    "get_model_path",
    "get_scaler_path",
    "get_plot_path",
    "get_metrics_path",
    "validate_config",
]
