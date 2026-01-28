"""
Configuration file for WWTP Deep Learning Prediction System

This module contains all hyperparameters, file paths, and system settings
for the LSTM-based prediction models.

Requirements: 1.2, 1.3, 1.4, 6.5, 10.4
"""

import os
from pathlib import Path

# ============================================================================
# Directory Paths
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results" / "DL"
NOTEBOOK_DIR = BASE_DIR / "notebook" / "DL"

# Data subdirectories (Requirements 1.2, 1.3, 1.4)
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FLOW_DATA_PATH = PROCESSED_DATA_DIR / "flow_proc.csv"
TMS_DATA_PATH = PROCESSED_DATA_DIR / "tms_proc.csv"
ALL_DATA_PATH = PROCESSED_DATA_DIR / "all_proc.csv"

# Model save directory (Requirement 6.5)
MODEL_SAVE_DIR = MODEL_DIR
SCALER_SAVE_DIR = MODEL_DIR

# Results save directory (Requirement 10.4)
RESULTS_SAVE_DIR = RESULTS_DIR
PLOTS_SAVE_DIR = RESULTS_DIR / "plots"
METRICS_SAVE_DIR = RESULTS_DIR / "metrics"

# ============================================================================
# Model Hyperparameters
# ============================================================================

# LSTM Architecture
LSTM_CONFIG = {
    "hidden_size": 64,          # Number of LSTM hidden units
    "num_layers": 2,            # Number of stacked LSTM layers
    "dropout": 0.2,             # Dropout rate for regularization
    "output_size": 1,           # Output dimension (single target)
}

# Sliding Window
WINDOW_SIZE = 24                # Number of time steps in input sequence

# ============================================================================
# Training Hyperparameters
# ============================================================================

TRAINING_CONFIG = {
    "batch_size": 32,           # Batch size for training
    "learning_rate": 0.001,     # Learning rate for optimizer
    "num_epochs": 100,          # Maximum number of training epochs
    "patience": 10,             # Early stopping patience
    "optimizer": "adam",        # Optimizer type: 'adam', 'rmsprop', 'sgd'
    "loss_function": "mse",     # Loss function: 'mse' or 'mae'
}

# ============================================================================
# Data Split Ratios
# ============================================================================

SPLIT_RATIOS = {
    "train": 0.7,               # Training set ratio
    "val": 0.15,                # Validation set ratio
    "test": 0.15,               # Test set ratio
}

# ============================================================================
# Target Variables
# ============================================================================

# Flow prediction target
FLOW_TARGET = "Q_in"

# TMS prediction targets
TMS_TARGETS = [
    "TOC_VU",
    "PH_VU",
    "SS_VU",
    "FLUX_VU",
    "TN_VU",
    "TP_VU",
]

# ============================================================================
# Device Configuration
# ============================================================================

# GPU/CPU device selection (auto-detect)
DEVICE_CONFIG = {
    "use_gpu": True,            # Try to use GPU if available
    "gpu_id": 0,                # GPU device ID
}

# ============================================================================
# Visualization Settings
# ============================================================================

VISUALIZATION_CONFIG = {
    "dpi": 300,                 # Plot resolution
    "figsize": (10, 6),         # Figure size (width, height)
    "font_family": "Malgun Gothic",  # Korean font support
    "grid": True,               # Show grid in plots
}

# ============================================================================
# Random Seed for Reproducibility
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# Performance Targets
# ============================================================================

PERFORMANCE_TARGETS = {
    "Q_in_r2": 0.95,            # R² target for flow prediction
    "TMS_r2": 0.90,             # R² target for TMS predictions
}

# ============================================================================
# Utility Functions
# ============================================================================

def create_directories():
    """
    Create all necessary directories if they don't exist.
    
    This function ensures that model save directories and results
    directories are available before training begins.
    """
    directories = [
        MODEL_SAVE_DIR,
        SCALER_SAVE_DIR,
        RESULTS_SAVE_DIR,
        PLOTS_SAVE_DIR,
        METRICS_SAVE_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {directory}")


def get_model_path(target_variable: str, suffix: str = "") -> Path:
    """
    Get the file path for saving/loading a model.
    
    Args:
        target_variable: Name of the target variable (e.g., 'Q_in', 'TOC_VU')
        suffix: Optional suffix to add to filename (e.g., 'best', 'final')
    
    Returns:
        Path object for the model file
    """
    filename = f"lstm_{target_variable}"
    if suffix:
        filename += f"_{suffix}"
    filename += ".pth"
    
    return MODEL_SAVE_DIR / filename


def get_scaler_path(scaler_name: str) -> Path:
    """
    Get the file path for saving/loading a scaler.
    
    Args:
        scaler_name: Name of the scaler (e.g., 'X_scaler', 'y_scaler_Q_in')
    
    Returns:
        Path object for the scaler file
    """
    filename = f"{scaler_name}.pkl"
    return SCALER_SAVE_DIR / filename


def get_plot_path(plot_name: str, target_variable: str = None) -> Path:
    """
    Get the file path for saving a plot.
    
    Args:
        plot_name: Name of the plot (e.g., 'training_history', 'predictions')
        target_variable: Optional target variable name
    
    Returns:
        Path object for the plot file
    """
    filename = plot_name
    if target_variable:
        filename += f"_{target_variable}"
    filename += ".png"
    
    return PLOTS_SAVE_DIR / filename


def get_metrics_path(target_variable: str) -> Path:
    """
    Get the file path for saving evaluation metrics.
    
    Args:
        target_variable: Name of the target variable
    
    Returns:
        Path object for the metrics file
    """
    filename = f"metrics_{target_variable}.json"
    return METRICS_SAVE_DIR / filename


def validate_config():
    """
    Validate configuration parameters.
    
    Raises:
        ValueError: If any configuration parameter is invalid
    """
    # Validate LSTM config
    if LSTM_CONFIG["hidden_size"] < 1:
        raise ValueError(f"Invalid hidden_size: {LSTM_CONFIG['hidden_size']}")
    if LSTM_CONFIG["num_layers"] < 1:
        raise ValueError(f"Invalid num_layers: {LSTM_CONFIG['num_layers']}")
    if not (0 <= LSTM_CONFIG["dropout"] <= 1):
        raise ValueError(f"Invalid dropout: {LSTM_CONFIG['dropout']}")
    
    # Validate window size
    if WINDOW_SIZE < 1:
        raise ValueError(f"Invalid window_size: {WINDOW_SIZE}")
    
    # Validate training config
    if TRAINING_CONFIG["batch_size"] < 1:
        raise ValueError(f"Invalid batch_size: {TRAINING_CONFIG['batch_size']}")
    if TRAINING_CONFIG["learning_rate"] <= 0:
        raise ValueError(f"Invalid learning_rate: {TRAINING_CONFIG['learning_rate']}")
    if TRAINING_CONFIG["num_epochs"] < 1:
        raise ValueError(f"Invalid num_epochs: {TRAINING_CONFIG['num_epochs']}")
    if TRAINING_CONFIG["patience"] < 1:
        raise ValueError(f"Invalid patience: {TRAINING_CONFIG['patience']}")
    
    # Validate split ratios
    total_ratio = sum(SPLIT_RATIOS.values())
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point error
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    print("✓ Configuration validated successfully")


if __name__ == "__main__":
    """
    Run this script to create directories and validate configuration.
    """
    print("=" * 60)
    print("WWTP Deep Learning Prediction System - Configuration")
    print("=" * 60)
    print()
    
    print("Creating directories...")
    create_directories()
    print()
    
    print("Validating configuration...")
    validate_config()
    print()
    
    print("Configuration Summary:")
    print(f"  Window Size: {WINDOW_SIZE}")
    print(f"  LSTM Hidden Size: {LSTM_CONFIG['hidden_size']}")
    print(f"  LSTM Layers: {LSTM_CONFIG['num_layers']}")
    print(f"  Dropout: {LSTM_CONFIG['dropout']}")
    print(f"  Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Max Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"  Early Stopping Patience: {TRAINING_CONFIG['patience']}")
    print()
    
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
