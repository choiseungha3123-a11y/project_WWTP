import json

# 노트북 로드
with open('notebook/DL/LSTM.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 새로운 import 셀 추가 (pickle)
import_cell_code = '''import pickle'''

# main 함수 수정
main_code = '''def main():

    dfs = load_data(DATA_DIR)
    
    X, y = preprocess_data(dfs)

    X_train, y_train, X_val, y_val, X_test, y_test = split_timewise(X, y)

    X_tr_s, X_va_s, X_te_s, y_tr_s, y_va_s, y_te_s, x_scaler, y_scaler = scale_data(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 스케일러 저장
    SCALER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCALER_SAVE_DIR / "X_scaler.pkl", "wb") as f:
        pickle.dump(x_scaler, f)
    with open(SCALER_SAVE_DIR / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)
    print(f"\\n✓ Scalers saved to {SCALER_SAVE_DIR}")

    y_tr_s = ensure_2d_y(y_tr_s)
    y_va_s = ensure_2d_y(y_va_s)
    y_te_s = ensure_2d_y(y_te_s)

    train_ds = TimeSeriesWindowDataset(X_tr_s, y_tr_s, SEQ_LEN, HORIZON)
    val_ds = TimeSeriesWindowDataset(X_va_s, y_va_s, SEQ_LEN, HORIZON)
    test_ds = TimeSeriesWindowDataset(X_te_s, y_te_s, SEQ_LEN, HORIZON)

    train_dl = DataLoader(train_ds,
                          batch_size=TRAINING_CONFIG["batch_size"],
                          shuffle=False,
                          drop_last=False)
    val_dl = DataLoader(val_ds,
                        batch_size=TRAINING_CONFIG["batch_size"],
                        shuffle=False,
                        drop_last=False)
    test_dl = DataLoader(test_ds,
                         batch_size=TRAINING_CONFIG["batch_size"],
                         shuffle=False,
                         drop_last=False)
    
    n_features = X_tr_s.shape[1]
    out_size = y_tr_s.shape[1]
    model = LSTMRegressor(n_features=n_features,
                          hidden_size=LSTM_CONFIG["hidden_size"],
                          num_layers=LSTM_CONFIG["num_layers"],
                          dropout=LSTM_CONFIG["dropout"],
                          out_size=out_size).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = TRAINING_CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 모델 저장 경로 설정
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODEL_SAVE_DIR / f"{MODE}_lstm_model.pth"

    hist = train_model(
        model, 
        train_dl, 
        val_dl, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=TRAINING_CONFIG["num_epochs"],
        patience=TRAINING_CONFIG["patience"],
        device=DEVICE,
        save_path=model_save_path
    )
    
    plot_learning_curve(
        train_loss=hist["train_loss"],
        train_mae=hist["train_mae"],
        train_rmse=hist["train_rmse"],
        train_mape=hist["train_mape"],
        val_loss=hist["val_loss"],
        val_mae=hist["val_mae"],
        val_rmse=hist["val_rmse"],
        val_mape=hist["val_mape"],
    )
    
    # 테스트 평가
    predictions, actuals, test_metrics = evaluate_model(model, test_dl, criterion, device=DEVICE)
    
    # 예측 결과 저장
    RESULTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        'actual': actuals.flatten(),
        'predicted': predictions.flatten()
    })
    results_df.to_csv(RESULTS_SAVE_DIR / f"{MODE}_predictions.csv", index=False)
    print(f"\\n✓ Predictions saved to {RESULTS_SAVE_DIR / f'{MODE}_predictions.csv'}")
    
    return model, hist, test_metrics
'''

# 첫 번째 import 셀 찾기
first_import_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'import' in ''.join(cell['source']):
        first_import_idx = i
        break

# pickle import 추가 (첫 번째 import 셀에)
if first_import_idx is not None:
    source_lines = nb['cells'][first_import_idx]['source']
    if 'import pickle' not in ''.join(source_lines):
        # 마지막 줄에 추가
        if source_lines and not source_lines[-1].endswith('\n'):
            source_lines[-1] += '\n'
        source_lines.append('import pickle\n')
        nb['cells'][first_import_idx]['source'] = source_lines
        print(f"Added pickle import to cell {first_import_idx}")

# main 함수 업데이트
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        
        if 'def main():' in source_text and 'dfs = load_data(DATA_DIR)' in source_text:
            cell['source'] = main_code.split('\n')
            print(f"Updated cell {i}: main function")
            break

# 노트북 저장
with open('notebook/DL/LSTM.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\n✓ Main function updated successfully!")
