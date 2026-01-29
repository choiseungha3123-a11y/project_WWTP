"""
개선된 파이프라인 모듈
StandardScaler, GridSearchCV, 피처 선택, Early Stopping 포함
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from .io import set_datetime_index, merge_sources_on_time
from .preprocess import resample_hourly
from .features import build_features, make_supervised_dataset, FeatureConfig
from .split import time_split, SplitConfig
from .feature_selection import select_top_features
from .models_improved import build_model_zoo_with_gridsearch
from .scaling import scale_data
from .metrics import compute_metrics, plot_metric_table
from .visualization import plot_learning_curve, plot_r2_comparison


# 타겟 컬럼 정의
TARGETS_FLOW = ["Q_in"]
TARGETS_TMS = ["TOC_VU", "PH_VU", "SS_VU", "FLUX_VU", "TN_VU", "TP_VU"]
TARGETS_ALL = TARGETS_FLOW + TARGETS_TMS


def get_target_cols(mode):
    """모드에 따른 타겟 컬럼 반환"""
    mode = mode.lower().strip()
    if mode == "flow":
        return TARGETS_FLOW
    if mode == "tms":
        return TARGETS_TMS
    if mode == "all":
        return TARGETS_ALL
    raise ValueError("mode는 'flow', 'tms', 'all' 중 하나여야 합니다.")


def fit_and_evaluate_improved(model, splits_scaled, model_name="", mode="", target_cols=None):
    """
    개선된 모델 학습 및 평가
    - 다중 타겟인 경우 각 타겟별 개별 학습
    - XGBoost Early Stopping 지원
    """
    X_train_scaled, y_train = splits_scaled["train"]
    X_valid_scaled, y_valid = splits_scaled["valid"]
    X_test_scaled, y_test = splits_scaled["test"]
    
    n_targets = y_train.shape[1] if len(y_train.shape) > 1 else 1
    
    if n_targets > 1:
        print(f"  다중 타겟 학습: {n_targets}개 타겟별 개별 모델 학습...")
        fitted_models = {}
        all_results = {"train": [], "valid": [], "test": []}
        
        for i, target_name in enumerate(target_cols):
            print(f"\n  [{i+1}/{n_targets}] {target_name} 학습 중...")
            
            y_train_single = y_train.iloc[:, i] if hasattr(y_train, 'iloc') else y_train[:, i]
            y_valid_single = y_valid.iloc[:, i] if hasattr(y_valid, 'iloc') else y_valid[:, i]
            y_test_single = y_test.iloc[:, i] if hasattr(y_test, 'iloc') else y_test[:, i]
            
            # 모델 복사
            if isinstance(model, GridSearchCV):
                single_model = type(model)(
                    estimator=type(model.estimator)(**model.estimator.get_params()),
                    param_grid=model.param_grid,
                    cv=model.cv,
                    scoring=model.scoring,
                    n_jobs=model.n_jobs,
                    verbose=0
                )
            else:
                single_model = type(model)(**model.get_params())
            
            # 학습
            if isinstance(single_model, GridSearchCV):
                single_model.fit(X_train_scaled, y_train_single)
                print(f"    최적 파라미터: {single_model.best_params_}")
                
                # XGBoost Early Stopping
                if 'XGB' in model_name:
                    best_params = single_model.best_params_.copy()
                    best_params['n_estimators'] = 500
                    
                    final_model = xgb.XGBRegressor(
                        **best_params,
                        early_stopping_rounds=20,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    final_model.fit(
                        X_train_scaled, y_train_single,
                        eval_set=[(X_train_scaled, y_train_single), (X_valid_scaled, y_valid_single)],
                        verbose=False
                    )
                    
                    if hasattr(final_model, 'best_iteration'):
                        print(f"    Early stopping: {final_model.best_iteration}번째 반복")
                    single_model = final_model
                
                elif 'HistGBR' in model_name:
                    if hasattr(single_model.best_estimator_, 'n_iter_'):
                        print(f"    반복 횟수: {single_model.best_estimator_.n_iter_}")
            else:
                single_model.fit(X_train_scaled, y_train_single)
            
            fitted_models[target_name] = single_model
            
            # 예측
            pred_train = single_model.predict(X_train_scaled)
            pred_valid = single_model.predict(X_valid_scaled)
            pred_test = single_model.predict(X_test_scaled)
            
            # 메트릭 계산
            all_results["train"].append(compute_metrics(y_train_single, pred_train))
            all_results["valid"].append(compute_metrics(y_valid_single, pred_valid))
            all_results["test"].append(compute_metrics(y_test_single, pred_test))
        
        # 평균 메트릭
        out = {}
        for split in ["train", "valid", "test"]:
            out[split] = {
                "R2_mean": float(np.mean([r["R2_mean"] for r in all_results[split]])),
                "RMSE_mean": float(np.mean([r["RMSE_mean"] for r in all_results[split]])),
                "MAPE_mean(%)": float(np.mean([r["MAPE_mean(%)"] for r in all_results[split]])),
                "R2_by_target": [r["R2_mean"] for r in all_results[split]],
                "RMSE_by_target": [r["RMSE_mean"] for r in all_results[split]],
                "MAPE_by_target(%)": [r["MAPE_mean(%)"] for r in all_results[split]],
            }
        
        # Learning curve (첫 번째 타겟 모델)
        first_model = fitted_models[target_cols[0]]
        plot_learning_curve(first_model, model_name, mode)
        
        return out, fitted_models
    
    else:
        # 단일 타겟
        print(f"  단일 타겟 학습...")
        
        if isinstance(model, GridSearchCV):
            model.fit(X_train_scaled, y_train)
            print(f"  최적 파라미터: {model.best_params_}")
            
            # XGBoost Early Stopping
            if 'XGB' in model_name:
                best_params = model.best_params_.copy()
                best_params['n_estimators'] = 500
                
                final_model = xgb.XGBRegressor(
                    **best_params,
                    early_stopping_rounds=20,
                    random_state=42,
                    n_jobs=-1
                )
                
                final_model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_train_scaled, y_train), (X_valid_scaled, y_valid)],
                    verbose=False
                )
                
                if hasattr(final_model, 'best_iteration'):
                    print(f"  Early stopping: {final_model.best_iteration}번째 반복")
                model = final_model
            
            elif 'HistGBR' in model_name:
                if hasattr(model.best_estimator_, 'n_iter_'):
                    print(f"  반복 횟수: {model.best_estimator_.n_iter_}")
        else:
            model.fit(X_train_scaled, y_train)
        
        # 평가
        out = {}
        pred_train = model.predict(X_train_scaled)
        pred_valid = model.predict(X_valid_scaled)
        pred_test = model.predict(X_test_scaled)
        
        out["train"] = compute_metrics(y_train.to_numpy(), pred_train)
        out["valid"] = compute_metrics(y_valid.to_numpy(), pred_valid)
        out["test"] = compute_metrics(y_test.to_numpy(), pred_test)
        
        # Learning curve
        plot_learning_curve(model, model_name, mode)
        
        return out, model


def run_improved_pipeline(
    dfs,
    mode,
    time_col_map=None,
    tz=None,
    resample_rule="1h",
    resample_agg="mean",
    feature_cfg=FeatureConfig(),
    split_cfg=SplitConfig(),
    n_top_features=50,
    cv_splits=3,
    random_state=42,
    save_dir="results/ML"
):
    """
    개선된 파이프라인 실행
    
    개선사항:
    - StandardScaler 적용
    - GridSearchCV 하이퍼파라미터 튜닝
    - 피처 선택 (중요도 기반)
    - TimeSeriesSplit 교차 검증
    - XGBoost + Early Stopping
    - Learning Curve 시각화
    """
    
    print(f"\n{'='*60}")
    print(f"개선된 파이프라인 - 모드: {mode.upper()}")
    print(f"{'='*60}")
    
    target_cols = get_target_cols(mode)
    
    # 1) 데이터 인덱싱
    dfs_indexed = {}
    for name, df in dfs.items():
        if df is None or len(df) == 0:
            dfs_indexed[name] = df
            continue
        
        if isinstance(df.index, pd.DatetimeIndex):
            dfs_indexed[name] = df.sort_index()
        else:
            if time_col_map is None or name not in time_col_map:
                raise ValueError(f"{name}에 DatetimeIndex가 없고 time_col_map도 제공되지 않았습니다.")
            dfs_indexed[name] = set_datetime_index(df, time_col=time_col_map[name], tz=tz)
    
    # 2) 병합
    df_all = merge_sources_on_time(dfs_indexed, how="outer")
    
    # 3) 리샘플링
    df_hourly = resample_hourly(df_all, rule=resample_rule, agg=resample_agg)
    
    # 4) 피처 엔지니어링
    df_feat = build_features(
        df_hourly=df_hourly,
        target_cols=target_cols,
        feature_base_cols=None,
        cfg=feature_cfg
    )
    
    # 5) 지도학습 데이터셋 생성
    X, y = make_supervised_dataset(df_feat, target_cols=target_cols, dropna=True)
    print(f"데이터셋 크기: {len(X)} 샘플, {X.shape[1]} 피처")
    
    # 6) 분할
    splits = time_split(X, y, cfg=split_cfg)
    X_train, y_train = splits["train"]
    
    # 7) 피처 선택
    top_features = select_top_features(X_train, y_train, n_features=n_top_features, random_state=random_state)
    X_train_selected = X_train[top_features]
    X_valid_selected = splits["valid"][0][top_features]
    X_test_selected = splits["test"][0][top_features]
    
    # 8) 스케일링
    print("\n데이터 스케일링 중...")
    X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_data(
        X_train_selected, X_valid_selected, X_test_selected
    )
    
    splits_scaled = {
        "train": (X_train_scaled, y_train),
        "valid": (X_valid_scaled, splits["valid"][1]),
        "test": (X_test_scaled, splits["test"][1])
    }
    
    # 9) 모델 학습 및 평가
    zoo = build_model_zoo_with_gridsearch(cv=cv_splits, random_state=random_state)
    
    results = {}
    fitted_models = {}
    
    for model_name, base_model in zoo.items():
        print(f"\n{'='*60}")
        print(f"모델 학습: {model_name}")
        print(f"{'='*60}")
        
        res, fitted_model = fit_and_evaluate_improved(
            base_model, splits_scaled,
            model_name=model_name,
            mode=mode,
            target_cols=target_cols
        )
        
        results[model_name] = res
        fitted_models[model_name] = fitted_model
        
        # 결과 출력
        print(f"\n  Train - R²: {res['train']['R2_mean']:.4f}, RMSE: {res['train']['RMSE_mean']:.2f}")
        print(f"  Valid - R²: {res['valid']['R2_mean']:.4f}, RMSE: {res['valid']['RMSE_mean']:.2f}")
        print(f"  Test  - R²: {res['test']['R2_mean']:.4f}, RMSE: {res['test']['RMSE_mean']:.2f}")
        
        # 과적합 체크
        train_r2 = res['train']['R2_mean']
        valid_r2 = res['valid']['R2_mean']
        if train_r2 - valid_r2 > 0.1:
            print(f"  ⚠️  과적합 가능성: Train R² ({train_r2:.4f}) >> Valid R² ({valid_r2:.4f})")
    
    metric_table = plot_metric_table(results, split="test")
    
    # R² 비교 시각화
    plot_r2_comparison(results, mode, save_dir=save_dir)
    
    print(f"\n{'='*60}")
    print("최종 결과 (Test Set)")
    print(f"{'='*60}")
    print(metric_table.to_string(index=False))
    
    return {
        "mode": mode,
        "target_cols": target_cols,
        "df_hourly": df_hourly,
        "df_features": df_feat,
        "X": X, "y": y,
        "splits": splits_scaled,
        "top_features": top_features,
        "scaler": scaler,
        "results": results,
        "metric_table": metric_table,
        "fitted_models": fitted_models
    }
