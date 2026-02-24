# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (PyTorch must be installed separately per platform)
pip install -r requirements.txt

# Run FastAPI backend (from python/)
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Run Streamlit demo dashboard (from python/)
streamlit run demo/app.py

# Run notebooks
jupyter notebook
```

## Architecture Overview

This is a WWTP (Wastewater Treatment Plant) inflow and water-quality prediction system.

### Layers

```
data/actual/      → notebook/preprocess/   → data/processed/
                  → notebook/feature/      (feature engineering)
                  → notebook/DL/           (LSTM training)
                  → model/save/            (artifacts)
                  → src/                   (FastAPI inference)
                  → demo/                  (Streamlit dashboard)
```

### `src/` — Production Inference

| File | Role |
|---|---|
| `config.py` | Model hyperparams for all 7 targets (source of truth) |
| `loader.py` | Loads `.pth` models + `.pkl` scalers + feature CSVs at startup |
| `models.py` | `LSTMRegressor` — single class for all targets |
| `schemas.py` | Pydantic I/O schemas (`FlowPredictIn`, `TMSPredictIn`, `PredictOut`) |
| `preprocess.py` | Input pipeline: merge 1-min records → 30-min resample → feature engineering → tensor |
| `predict.py` | `autoregressive_predict()` — sliding 48-step window, 24-step forecast |
| `main.py` | FastAPI routes: `/health`, `/ready`, `/predict/flow`, `/predict/tms` |

### API Input Requirements

Both `/predict/flow` and `/predict/tms` expect **1440 1-minute raw records** (24 hours) plus records from **3 AWS weather stations** (368, 541, 569). The pipeline resamples these to 30-minute intervals internally.

Request body shape (`TMSPredictIn` / `FlowPredictIn`):
```json
{
  "in": {
    "dataList": [...],          // 1440 TMS or Flow records
    "awsList": {
      "368": [...],             // AWS station records
      "541": [...],
      "569": [...]
    }
  },
  "request_id": "optional-uuid"
}
```

Response (`PredictOut`):
```json
{
  "request_id": "...",
  "ok": true,
  "output": {
    "predictions": { "toc": [...], "ss": [...], ... },  // hourly (12 values)
    "trajectories": { "toc": { "12h": [...] }, ... },   // 30-min (24 values)
    "metadata": { "window_size": 48, ... }
  },
  "latency_ms": 123,
  "error": null
}
```

### Inference Pipeline Detail

1. `merge_input_data()` — merges flow/TMS + 3 AWS lists, resamples 1-min → 30-min
2. `apply_tms_feature_engineering()` / `apply_flow_feature_engineering()` — applies the full feature pipeline per target (see MEMORY.md for order)
3. Scale with `X_scaler_{target}.pkl`, trim to recommended features from `data/recommand_features/save/`
4. `autoregressive_predict()` — uses the last 48 rows as initial window; predicts 1 step at a time for 24 steps, updating the target-lag feature in the window each iteration

### `notebook/` — Training & Research

- `notebook/DL/LSTM_TMS.ipynb` — trains all 6 TMS models; `MODE_CONFIGS` dict is the notebook-side source of model configs
- `notebook/DL/LSTM_FLOW.ipynb` — trains the flow model
- `notebook/feature/feature_engineering.py` — **shared module** imported by both notebooks and `src/preprocess.py`
- `notebook/DL/transformer_TMS.ipynb` — Transformer experiments (underperformed LSTM, not deployed)

### `demo/` — Streamlit Dashboard

Multi-page portfolio app with live inference, prediction analysis, performance comparison (ML baseline vs DL stages), and model architecture info. Utils in `demo/utils/`: `constants.py` (R² values), `data_loader.py`, `metrics.py`, `live_infer.py`.

## Key Conventions

- **`src/config.py` is the source of truth** for production model configs. MEMORY.md may have stale values — always check `config.py` when debugging model architecture mismatches.
- `FLUX_VU` is handled as a difference (diff) column during resampling in `merge_input_data()`, unlike other TMS columns which are averaged.
- Flow's autoregressive loop does **not** update target lags (uses `flow_TankA/B` lags instead of `Q_in` lags) — `flow_target_idx=None` is passed to `autoregressive_predict()`.
- Feature recommendation CSVs live at `data/recommand_features/save/` (note: "recommand" is the project's spelling).
