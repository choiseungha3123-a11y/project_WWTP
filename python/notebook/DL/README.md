# WWTP Deep Learning Prediction System

This directory contains the implementation of LSTM-based deep learning models for predicting wastewater treatment plant (WWTP) inflow and water quality indicators.

## Directory Structure

```
python/
├── notebook/DL/          # Deep learning notebooks and scripts
│   ├── config.py         # Configuration file (hyperparameters, paths)
│   └── README.md         # This file
├── model/                # Saved model checkpoints and scalers
├── results/DL/           # Training results and visualizations
│   ├── plots/            # Training curves and prediction plots
│   └── metrics/          # Evaluation metrics (JSON format)
└── data/processed/       # Preprocessed data files
    ├── flow_proc.csv     # Flow data
    ├── tms_proc.csv      # TMS water quality data
    └── all_proc.csv      # Combined data
```

## Configuration

All system settings are centralized in `config.py`:

- **Model Architecture**: LSTM layers, hidden size, dropout
- **Training Parameters**: Batch size, learning rate, epochs, early stopping
- **Data Paths**: Input data locations, model save paths, results paths
- **Target Variables**: Q_in (flow) and TMS indicators (TOC_VU, PH_VU, SS_VU, FLUX_VU, TN_VU, TP_VU)

### Key Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `WINDOW_SIZE` | 24 | Number of time steps in input sequence |
| `hidden_size` | 64 | LSTM hidden units |
| `num_layers` | 2 | Number of stacked LSTM layers |
| `dropout` | 0.2 | Dropout rate for regularization |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `num_epochs` | 100 | Maximum training epochs |
| `patience` | 10 | Early stopping patience |

### Data Split

- Training: 70%
- Validation: 15%
- Test: 15%

## Performance Targets

- **Q_in Prediction**: R² ≥ 0.95
- **TMS Indicators**: R² ≥ 0.90

## Setup

To initialize the directory structure and validate configuration:

```bash
cd python
python notebook/DL/config.py
```

This will:
1. Create all necessary directories
2. Validate configuration parameters
3. Display configuration summary

## Usage

### Import Configuration

```python
from config import (
    LSTM_CONFIG,
    TRAINING_CONFIG,
    WINDOW_SIZE,
    FLOW_DATA_PATH,
    TMS_DATA_PATH,
    get_model_path,
    get_scaler_path,
    get_plot_path,
    create_directories
)
```

### Get File Paths

```python
# Model paths
model_path = get_model_path("Q_in", suffix="best")
# Returns: python/model/lstm_Q_in_best.pth

# Scaler paths
scaler_path = get_scaler_path("X_scaler")
# Returns: python/model/X_scaler.pkl

# Plot paths
plot_path = get_plot_path("training_history", "Q_in")
# Returns: python/results/DL/plots/training_history_Q_in.png
```

## Requirements

This implementation addresses the following requirements:

- **1.2**: Load flow data from `python/data/processed/flow_proc.csv`
- **1.3**: Load TMS data from `python/data/processed/tms_proc.csv`
- **1.4**: Load combined data from `python/data/processed/all_proc.csv`
- **6.5**: Save models to `python/model/` directory
- **10.4**: Save results to `python/results/DL/` directory

## Next Steps

After configuration setup, the following components will be implemented:

1. Data Loader - Load and validate CSV files
2. Preprocessor - Data normalization and validation
3. Dataset Builder - Sliding window sequence generation
4. LSTM Model - Neural network architecture
5. Model Trainer - Training loop with early stopping
6. Evaluator - Performance metrics calculation
7. Visualizer - Training curves and prediction plots
8. Prediction Service - Inference pipeline

## Notes

- All paths are automatically resolved relative to the project root
- GPU acceleration is enabled by default (falls back to CPU if unavailable)
- Korean font support is configured for visualizations (Malgun Gothic)
- Random seed is set to 42 for reproducibility
