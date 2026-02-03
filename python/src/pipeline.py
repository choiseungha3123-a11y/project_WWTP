"""
파이프라인 오케스트레이션 모듈
전체 학습 파이프라인 실행

두 가지 파이프라인 제공:
1. run_pipeline: 기본 파이프라인
2. run_improved_pipeline: 개선된 파이프라인 (Optuna, 피처 선택, Scaling, Early Stopping)
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from .io import set_datetime_index, summarize_available_period, merge_sources_on_time
from .preprocess import (drop_missing_rows, resample_hourly, check_continuity, 
                         impute_missing_with_strategy, ImputationConfig, summarize_imputation,
                         detect_and_handle_outliers, OutlierConfig, summarize_outliers)
from .features import build_features, make_supervised_dataset, FeatureConfig
from .split import time_split, SplitConfig
from .models import build_model_zoo, build_model_zoo_with_optuna, wrap_multioutput_if_needed, OptunaModelWrapper
from .metrics import fit_and_evaluate, compute_metrics, plot_metric_table
from .feature_selection import select_top_features
from .scaling import scale_data
from .visualization import plot_learning_curve, plot_r2_comparison


# 타겟 컬럼 정의
TARGETS_FLOW = ["Q_in"]
TARGETS_TMS_ALL = ["TOC_VU", "PH_VU", "SS_VU", "FLUX_VU", "TN_VU", "TP_VU"]

# TMS 모델 그룹 정의
TARGETS_MODEL_A = ["TOC_VU", "SS_VU"]      # 유기물/입자 계열
TARGETS_MODEL_B = ["TN_VU", "TP_VU"]       # 영양염 계열
TARGETS_MODEL_C = ["FLUX_VU", "PH_VU"]     # 공정 상태 계열


def get_target_cols(mode):
    """
    모드에 따른 타겟 컬럼 반환
    
    Parameters:
    -----------
    mode : str
        예측 모드
        - 'flow': 유량 예측 (Q_in)
        - 'tms': 전체 TMS 지표 (6개)
        - 'modelA': 유기물/입자 계열 (TOC_VU, SS_VU)
        - 'modelB': 영양염 계열 (TN_VU, TP_VU)
        - 'modelC': 공정 상태 계열 (FLUX_VU, PH_VU)
        
    Returns:
    --------
    list : 타겟 컬럼 리스트
    """
    mode = mode.lower().strip()
    if mode == "flow":
        return TARGETS_FLOW
    elif mode == "tms":
        return TARGETS_TMS_ALL
    elif mode == "modela":
        return TARGETS_MODEL_A
    elif mode == "modelb":
        return TARGETS_MODEL_B
    elif mode == "modelc":
        return TARGETS_MODEL_C
    else:
        raise ValueError("mode는 'flow', 'tms', 'modelA', 'modelB', 'modelC' 중 하나여야 합니다.")


def get_exclude_features(mode, target_cols):
    """
    모드에 따라 입력 피처에서 제외할 컬럼 반환
    
    노트북 설계 기반:
    - ModelA (TOC+SS 예측): PH, FLUX, TN, TP를 입력으로 사용 → TOC, SS만 제외
    - ModelB (TN+TP 예측): PH, FLUX, SS, TOC를 입력으로 사용 → TN, TP만 제외
    - ModelC (FLUX+PH 예측): TOC, SS, TN, TP를 입력으로 사용 → FLUX, PH만 제외
    - ModelFLOW (Q_in 예측): 
        * TMS 데이터 전혀 사용 안 함 → 모든 TMS 제외
        * flow_TankA, flow_TankB 제외 (Q_in의 구성 요소이므로 데이터 누수)
        * level_TankA, level_TankB는 입력으로 사용 (수위 정보)
    - TMS 전체: FLOW 데이터 제외
    
    Parameters:
    -----------
    mode : str
        예측 모드
    target_cols : list
        타겟 컬럼 리스트
        
    Returns:
    --------
    list : 제외할 컬럼 리스트
    """
    mode = mode.lower().strip()
    exclude = target_cols.copy()  # 예측 대상은 항상 제외
    
    # FLOW 모드: TMS 지표 전체 제외 (미래 정보) + flow_TankA/B 제외 (데이터 누수)
    if mode == "flow":
        exclude.extend(TARGETS_TMS_ALL)
        # flow_TankA, flow_TankB는 Q_in의 구성 요소이므로 제외
        # level_TankA, level_TankB는 수위 정보로 입력 사용 가능
        exclude.extend(["flow_TankA", "flow_TankB"])
    
    # ModelA (TOC+SS): 나머지 TMS 지표(PH, FLUX, TN, TP)는 입력으로 사용
    # → TOC_VU, SS_VU만 제외 (이미 target_cols에 포함됨)
    elif mode == "modela":
        # FLOW 데이터는 제외 (미래 정보)
        exclude.extend(TARGETS_FLOW)
    
    # ModelB (TN+TP): 나머지 TMS 지표(PH, FLUX, SS, TOC)는 입력으로 사용
    # → TN_VU, TP_VU만 제외 (이미 target_cols에 포함됨)
    elif mode == "modelb":
        # FLOW 데이터는 제외 (미래 정보)
        exclude.extend(TARGETS_FLOW)
    
    # ModelC (FLUX+PH): 나머지 TMS 지표(TOC, SS, TN, TP)는 입력으로 사용
    # → FLUX_VU, PH_VU만 제외 (이미 target_cols에 포함됨)
    elif mode == "modelc":
        # FLOW 데이터는 제외 (미래 정보)
        exclude.extend(TARGETS_FLOW)
    
    # TMS 전체 모드: FLOW 데이터 제외 (미래 정보)
    elif mode == "tms":
        exclude.extend(TARGETS_FLOW)
    
    # 중복 제거
    return list(set(exclude))


def run_pipeline(
    dfs,
    mode,
    time_col_map=None,
    tz=None,
    dropna_cols_before_resample=None,
    resample_rule="1h",
    resample_agg="mean",
    feature_base_cols=None,
    feature_cfg=FeatureConfig(),
    split_cfg=SplitConfig(),
    imputation_cfg=ImputationConfig(),
    outlier_cfg=OutlierConfig(),
    use_outlier_detection=True,
    use_imputation=True,
    random_state=42
):
    """
    전체 학습 파이프라인 실행
    
    전처리 순서:
    1. 시간축 정합 (정렬/중복 처리)
    2. 결측치 보간 (ffill/EWMA)
    3. 이상치 처리 (NaN 변환 후 재보간)
    4. 리샘플링
    5. 파생 특성 (rolling/lag/시간특성 - 데이터 누수 방지)
    6. Train/Valid/Test 분리
    7. 모델 학습 및 평가
    
    Parameters:
    -----------
    dfs : dict
        데이터프레임 딕셔너리 {name: df}
    mode : str
        'flow', 'tms', 'all' 중 하나
    time_col_map : dict
        각 데이터프레임의 시간 컬럼명 매핑
    tz : str
        타임존 (선택사항)
    dropna_cols_before_resample : list
        리샘플링 전 결측치 제거할 컬럼 (레거시, use_imputation=False일 때만)
    resample_rule : str
        리샘플링 규칙 (예: '1h', '5min')
    resample_agg : str or dict
        집계 방법
    feature_base_cols : list
        피처 생성에 사용할 기본 컬럼
    feature_cfg : FeatureConfig
        피처 생성 설정
    split_cfg : SplitConfig
        데이터 분할 설정
    imputation_cfg : ImputationConfig
        결측치 보간 설정
    outlier_cfg : OutlierConfig
        이상치 탐지 설정
    use_outlier_detection : bool
        이상치 탐지 사용 여부 (기본: True)
    use_imputation : bool
        전략적 결측치 보간 사용 여부 (기본: True)
    random_state : int
        랜덤 시드
        
    Returns:
    --------
    dict : 파이프라인 실행 결과
    """
    
    print(f"\n{'='*60}")
    print(f"기본 파이프라인 - 모드: {mode.upper()}")
    print(f"{'='*60}")
    
    target_cols = get_target_cols(mode)

    # ========================================
    # 1) 시간축 정합 (정렬/중복 처리)
    # ========================================
    print("\n[1/7] 시간축 정합 중...")
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
        
        # 중복 제거 (같은 시간에 여러 값이 있으면 첫 번째 값 유지)
        dfs_indexed[name] = dfs_indexed[name][~dfs_indexed[name].index.duplicated(keep='first')]

    period_summary = summarize_available_period(dfs_indexed)

    # 병합
    df_all = merge_sources_on_time(dfs_indexed, how="outer")
    
    # 불필요한 메타데이터 컬럼 제거
    metadata_cols = ['data_save_dt', 'YYMMDDHHMI_368', 'YYMMDDHHMI_541', 'YYMMDDHHMI_569', 
                     'STN_368', 'STN_541', 'STN_569']
    cols_to_drop = [col for col in metadata_cols if col in df_all.columns]
    if cols_to_drop:
        print(f"\n메타데이터 컬럼 제거: {cols_to_drop}")
        df_all = df_all.drop(columns=cols_to_drop)

    # ========================================
    # 2) 결측치 보간 (ffill/EWMA)
    # ========================================
    if use_imputation:
        print("\n[2/7] 결측치 보간 중 (1차)...")
        df_all = impute_missing_with_strategy(
            df_all, 
            freq=resample_rule, 
            config=imputation_cfg,
            add_mask=True
        )
        imputation_summary_1 = summarize_imputation(df_all)
        if imputation_summary_1 is not None:
            print("\n결측치 처리 요약 (1차):")
            print(imputation_summary_1.to_string(index=False))
    else:
        imputation_summary_1 = None

    # ========================================
    # 3) 이상치 처리 (NaN 변환 후 재보간)
    # ========================================
    if use_outlier_detection:
        print("\n[3/7] 이상치 탐지 및 처리 중...")
        df_all = detect_and_handle_outliers(df_all, config=outlier_cfg, add_mask=True)
        outlier_summary = summarize_outliers(df_all)
        
        # 이상치를 NaN으로 변환했으므로 재보간
        if use_imputation:
            print("\n결측치 재보간 중 (이상치 처리 후)...")
            df_all = impute_missing_with_strategy(
                df_all, 
                freq=resample_rule, 
                config=imputation_cfg,
                add_mask=False  # mask는 이미 추가됨
            )
    else:
        outlier_summary = None
    
    # 레거시 지원: use_imputation=False일 때 결측치 행 제거
    if not use_imputation:
        df_all = drop_missing_rows(df_all, cols=dropna_cols_before_resample)
        imputation_summary = None
    else:
        imputation_summary = imputation_summary_1

    # ========================================
    # 4) 리샘플링
    # ========================================
    print(f"\n[4/7] 리샘플링 중 ({resample_rule})...")
    df_hourly = resample_hourly(df_all, rule=resample_rule, agg=resample_agg)

    # ========================================
    # 5) 파생 특성 (rolling/lag/시간특성)
    # ========================================
    print("\n[5/7] 파생 특성 생성 중...")
    
    # 제외할 컬럼 결정 (예측 대상만 제외, 나머지 TMS 지표는 입력으로 사용)
    exclude_cols = get_exclude_features(mode, target_cols)
    
    df_feat = build_features(
        df_hourly=df_hourly,
        target_cols=target_cols,
        exclude_cols=exclude_cols,
        feature_base_cols=feature_base_cols,
        mode=mode,  # 모델 모드 전달
        cfg=feature_cfg
    )

    # 연속성 확인
    continuity = check_continuity(df_hourly.dropna(how="all"), freq=resample_rule)

    # 지도학습 데이터셋 X, y 생성
    max_lag = max(feature_cfg.lag_hours) if feature_cfg.lag_hours else 24
    max_roll = max(feature_cfg.roll_hours) if feature_cfg.roll_hours else 24
    max_window = max(max_lag, max_roll)
    
    X, y = make_supervised_dataset(
        df_feat, 
        target_cols=target_cols, 
        exclude_cols=exclude_cols, 
        dropna=True,
        drop_initial_nan_only=True,
        max_lag_window=max_window
    )
    print(f"데이터셋 크기: {len(X)} 샘플, {X.shape[1]} 피처")
    
    # 입력 피처 정보 출력
    if mode == "flow":
        print(f"입력 데이터: AWS 기상 데이터 + level_TankA/B (TMS 지표, flow_TankA/B 제외)")
    elif mode in ["modela", "modelb", "modelc"]:
        tms_input_cols = [c for c in TARGETS_TMS_ALL if c not in target_cols]
        print(f"입력 데이터: AWS 기상 데이터 + 나머지 TMS 지표 {tms_input_cols} (FLOW 데이터 제외)")
    elif mode == "tms":
        print(f"입력 데이터: AWS 기상 데이터 (FLOW 데이터 제외)")

    # ========================================
    # 6) 데이터 분할
    # ========================================
    print("\n[6/7] 데이터 분할 중...")
    splits = time_split(X, y, cfg=split_cfg)
    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]
    X_test, y_test = splits["test"]
    
    print(f"  Train: {len(X_train)} 샘플")
    print(f"  Valid: {len(X_valid)} 샘플")
    print(f"  Test:  {len(X_test)} 샘플")

    # ========================================
    # 7) 모델 학습 및 평가
    # ========================================
    print("\n[7/7] 모델 학습 및 평가 중...")
    zoo = build_model_zoo(random_state=random_state)

    results = {}
    fitted_models = {}
    for model_name, base_model in zoo.items():
        print(f"\n모델 학습: {model_name}")
        model = wrap_multioutput_if_needed(base_model, y)
        res = fit_and_evaluate(model, splits)
        results[model_name] = res
        fitted_models[model_name] = model
        
        # 결과 출력
        print(f"  Train - R²: {res['train']['R2_mean']:.4f}, RMSE: {res['train']['RMSE_mean']:.2f}")
        print(f"  Valid - R²: {res['valid']['R2_mean']:.4f}, RMSE: {res['valid']['RMSE_mean']:.2f}")
        print(f"  Test  - R²: {res['test']['R2_mean']:.4f}, RMSE: {res['test']['RMSE_mean']:.2f}")

    metric_table = plot_metric_table(results, split="test")
    
    print(f"\n{'='*60}")
    print("최종 결과 (Test Set)")
    print(f"{'='*60}")
    print(metric_table.to_string(index=False))

    return {
        "mode": mode,
        "target_cols": target_cols,
        "period_summary": period_summary,
        "df_merged": df_all,
        "df_hourly": df_hourly,
        "df_features": df_feat,
        "continuity": continuity,
        "outlier_summary": outlier_summary,
        "imputation_summary": imputation_summary,
        "X": X, "y": y,
        "splits": splits,
        "results": results,
        "metric_table": metric_table,
        "fitted_models": fitted_models
    }



def fit_and_evaluate_improved(model, splits_scaled, model_name="", mode="", target_cols=None):
    """
    개선된 모델 학습 및 평가
    - 다중 타겟인 경우 각 타겟별 개별 학습
    - Optuna 하이퍼파라미터 최적화 지원
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
            if isinstance(model, OptunaModelWrapper):
                single_model = OptunaModelWrapper(
                    model.model_name,
                    model.cv_splits,
                    model.n_trials,
                    model.random_state
                )
            else:
                single_model = type(model)(**model.get_params())
            
            # 학습
            if isinstance(single_model, OptunaModelWrapper):
                single_model.fit(X_train_scaled, y_train_single)
                print(f"    최적 파라미터: {single_model.best_params_}")
                print(f"    최적 MSE: {single_model.study_.best_value:.4f}")
                
                # XGBoost Early Stopping (Optuna 최적화 후)
                if single_model.model_name == 'XGBoost':
                    best_params = single_model.best_params_.copy()
                    best_params['n_estimators'] = 500
                    best_params['early_stopping_rounds'] = 20
                    
                    # random_state와 n_jobs는 이미 best_params에 포함되어 있음
                    if 'random_state' not in best_params:
                        best_params['random_state'] = 42
                    if 'n_jobs' not in best_params:
                        best_params['n_jobs'] = -1
                    
                    final_model = xgb.XGBRegressor(**best_params)
                    
                    final_model.fit(
                        X_train_scaled, y_train_single,
                        eval_set=[(X_train_scaled, y_train_single), (X_valid_scaled, y_valid_single)],
                        verbose=False
                    )
                    
                    if hasattr(final_model, 'best_iteration'):
                        print(f"    Early stopping: {final_model.best_iteration}번째 반복")
                    single_model.best_model_ = final_model
                
                elif single_model.model_name == 'HistGBR':
                    if hasattr(single_model.best_model_, 'n_iter_'):
                        print(f"    반복 횟수: {single_model.best_model_.n_iter_}")
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
        if isinstance(first_model, OptunaModelWrapper):
            plot_learning_curve(first_model.best_model_, model_name, mode)
        else:
            plot_learning_curve(first_model, model_name, mode)
        
        return out, fitted_models
    
    else:
        # 단일 타겟
        print(f"  단일 타겟 학습...")
        
        if isinstance(model, OptunaModelWrapper):
            model.fit(X_train_scaled, y_train)
            print(f"  최적 파라미터: {model.best_params_}")
            print(f"  최적 MSE: {model.study_.best_value:.4f}")
            
            # XGBoost Early Stopping
            if model.model_name == 'XGBoost':
                best_params = model.best_params_.copy()
                best_params['n_estimators'] = 500
                best_params['early_stopping_rounds'] = 20
                
                # random_state와 n_jobs는 이미 best_params에 포함되어 있음
                if 'random_state' not in best_params:
                    best_params['random_state'] = 42
                if 'n_jobs' not in best_params:
                    best_params['n_jobs'] = -1
                
                final_model = xgb.XGBRegressor(**best_params)
                
                final_model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_train_scaled, y_train), (X_valid_scaled, y_valid)],
                    verbose=False
                )
                
                if hasattr(final_model, 'best_iteration'):
                    print(f"  Early stopping: {final_model.best_iteration}번째 반복")
                model.best_model_ = final_model
            
            elif model.model_name == 'HistGBR':
                if hasattr(model.best_model_, 'n_iter_'):
                    print(f"  반복 횟수: {model.best_model_.n_iter_}")
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
        if isinstance(model, OptunaModelWrapper):
            plot_learning_curve(model.best_model_, model_name, mode)
        else:
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
    imputation_cfg=ImputationConfig(),
    outlier_cfg=OutlierConfig(),
    use_outlier_detection=True,
    use_imputation=True,
    n_top_features=50,
    cv_splits=3,
    n_trials=50,
    random_state=42,
    save_dir="results/ML"
):
    """
    개선된 파이프라인 실행
    
    전처리 순서:
    1. 시간축 정합 (정렬/중복 처리)
    2. 결측치 보간 (ffill/EWMA)
    3. 이상치 처리 (NaN 변환 후 재보간)
    4. 리샘플링 (10분/1시간)
    5. 파생 특성 (rolling/lag/시간특성 - 데이터 누수 방지)
    6. Train/Valid/Test 분리
    7. 스케일링 (Train 기준)
    8. 특성 선택 (Train 기준)
    
    개선사항:
    - 올바른 전처리 순서 적용
    - 데이터 누수 방지 (분리 후 스케일링/선택)
    - Optuna 하이퍼파라미터 최적화
    - TimeSeriesSplit 교차 검증
    - XGBoost + Early Stopping
    - Learning Curve 시각화
    
    Parameters:
    -----------
    dfs : dict
        데이터프레임 딕셔너리 {name: df}
    mode : str
        'flow', 'tms', 'all' 중 하나
    time_col_map : dict
        각 데이터프레임의 시간 컬럼명 매핑
    tz : str
        타임존 (선택사항)
    resample_rule : str
        리샘플링 규칙 (예: '1h', '10min')
    resample_agg : str or dict
        집계 방법
    feature_cfg : FeatureConfig
        피처 생성 설정
    split_cfg : SplitConfig
        데이터 분할 설정
    imputation_cfg : ImputationConfig
        결측치 보간 설정
    outlier_cfg : OutlierConfig
        이상치 탐지 설정
    use_outlier_detection : bool
        이상치 탐지 사용 여부 (기본: True)
    use_imputation : bool
        전략적 결측치 보간 사용 여부 (기본: True)
    n_top_features : int
        선택할 상위 피처 개수
    cv_splits : int
        TimeSeriesSplit 분할 수
    n_trials : int
        Optuna 시도 횟수 (기본: 50)
    random_state : int
        랜덤 시드
    save_dir : str
        결과 저장 디렉토리
        
    Returns:
    --------
    dict : 파이프라인 실행 결과
    """
    
    print(f"\n{'='*60}")
    print(f"개선된 파이프라인 (Optuna) - 모드: {mode.upper()}")
    print(f"{'='*60}")
    
    target_cols = get_target_cols(mode)
    
    # ========================================
    # 1) 시간축 정합 (정렬/중복 처리)
    # ========================================
    print("\n[1/9] 시간축 정합 중...")
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
        
        # 중복 제거 (같은 시간에 여러 값이 있으면 평균)
        dfs_indexed[name] = dfs_indexed[name][~dfs_indexed[name].index.duplicated(keep='first')]
    
    # 병합
    df_all = merge_sources_on_time(dfs_indexed, how="outer")
    
    # 불필요한 메타데이터 컬럼 제거
    metadata_cols = ['data_save_dt', 'YYMMDDHHMI_368', 'YYMMDDHHMI_541', 'YYMMDDHHMI_569', 
                     'STN_368', 'STN_541', 'STN_569']
    cols_to_drop = [col for col in metadata_cols if col in df_all.columns]
    if cols_to_drop:
        print(f"\n메타데이터 컬럼 제거: {cols_to_drop}")
        df_all = df_all.drop(columns=cols_to_drop)
    
    # ========================================
    # 2) 결측치 보간 (ffill/EWMA)
    # ========================================
    if use_imputation:
        print("\n[2/9] 결측치 보간 중 (1차)...")
        df_all = impute_missing_with_strategy(
            df_all, 
            freq=resample_rule, 
            config=imputation_cfg,
            add_mask=True
        )
        imputation_summary_1 = summarize_imputation(df_all)
        if imputation_summary_1 is not None:
            print("\n결측치 처리 요약 (1차):")
            print(imputation_summary_1.to_string(index=False))
    else:
        imputation_summary_1 = None
    
    # ========================================
    # 3) 이상치 처리 (NaN 변환 후 재보간)
    # ========================================
    if use_outlier_detection:
        print("\n[3/9] 이상치 탐지 및 처리 중...")
        df_all = detect_and_handle_outliers(df_all, config=outlier_cfg, add_mask=True)
        outlier_summary = summarize_outliers(df_all)
        
        # 이상치를 NaN으로 변환했으므로 재보간
        if use_imputation:
            print("\n결측치 재보간 중 (이상치 처리 후)...")
            df_all = impute_missing_with_strategy(
                df_all, 
                freq=resample_rule, 
                config=imputation_cfg,
                add_mask=False  # mask는 이미 추가됨
            )
    else:
        outlier_summary = None
    
    # ========================================
    # 4) 리샘플링 (10분/1시간)
    # ========================================
    print(f"\n[4/9] 리샘플링 중 ({resample_rule})...")
    df_resampled = resample_hourly(df_all, rule=resample_rule, agg=resample_agg)
    
    # ========================================
    # 5) 파생 특성 (rolling/lag/시간특성)
    # ========================================
    print("\n[5/9] 파생 특성 생성 중...")
    
    # 제외할 컬럼 결정 (예측 대상만 제외, 나머지 TMS 지표는 입력으로 사용)
    exclude_cols = get_exclude_features(mode, target_cols)
    
    df_feat = build_features(
        df_hourly=df_resampled,
        target_cols=target_cols,
        exclude_cols=exclude_cols,
        feature_base_cols=None,
        mode=mode,  # 모델 모드 전달
        cfg=feature_cfg
    )
    
    # 지도학습 데이터셋 생성 (dropna로 완전한 행만 사용)
    max_lag = max(feature_cfg.lag_hours) if feature_cfg.lag_hours else 24
    max_roll = max(feature_cfg.roll_hours) if feature_cfg.roll_hours else 24
    max_window = max(max_lag, max_roll)
    
    X, y = make_supervised_dataset(
        df_feat, 
        target_cols=target_cols, 
        exclude_cols=exclude_cols, 
        dropna=True,
        drop_initial_nan_only=True,
        max_lag_window=max_window
    )
    print(f"데이터셋 크기: {len(X)} 샘플, {X.shape[1]} 피처")
    
    # 입력 피처 정보 출력
    if mode == "flow":
        print(f"입력 데이터: AWS 기상 데이터 + level_TankA/B (TMS 지표, flow_TankA/B 제외)")
    elif mode in ["modela", "modelb", "modelc"]:
        tms_input_cols = [c for c in TARGETS_TMS_ALL if c not in target_cols]
        print(f"입력 데이터: AWS 기상 데이터 + 나머지 TMS 지표 {tms_input_cols} (FLOW 데이터 제외)")
    elif mode == "tms":
        print(f"입력 데이터: AWS 기상 데이터 (FLOW 데이터 제외)")
    
    # ========================================
    # 6) Train/Valid/Test 분리
    # ========================================
    print("\n[6/9] 데이터 분할 중...")
    splits = time_split(X, y, cfg=split_cfg)
    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]
    X_test, y_test = splits["test"]
    
    print(f"  Train: {len(X_train)} 샘플")
    print(f"  Valid: {len(X_valid)} 샘플")
    print(f"  Test:  {len(X_test)} 샘플")
    
    # ========================================
    # 7) 스케일링 (Train 기준으로 fit)
    # ========================================
    print("\n[7/9] 스케일링 중 (Train 기준)...")
    X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_data(
        X_train, X_valid, X_test
    )
    
    # ========================================
    # 8) 특성 선택 (Train 기준으로 선택)
    # ========================================
    print(f"\n[8/9] 특성 선택 중 (상위 {n_top_features}개, Train 기준)...")
    top_features = select_top_features(X_train_scaled, y_train, n_features=n_top_features, random_state=random_state)
    
    X_train_selected = X_train_scaled[top_features]
    X_valid_selected = X_valid_scaled[top_features]
    X_test_selected = X_test_scaled[top_features]
    
    print(f"  선택된 피처: {len(top_features)}개")
    
    splits_scaled = {
        "train": (X_train_selected, y_train),
        "valid": (X_valid_selected, y_valid),
        "test": (X_test_selected, y_test)
    }
    
    # ========================================
    # 9) 모델 학습 및 평가
    # ========================================
    print(f"\n[9/9] 모델 학습 및 평가 중...")
    zoo = build_model_zoo_with_optuna(cv_splits=cv_splits, n_trials=n_trials, random_state=random_state)
    
    results = {}
    fitted_models = {}
    
    for model_name, base_model in zoo.items():
        print(f"\n{'='*60}")
        print(f"모델 학습: {model_name}")
        if isinstance(base_model, OptunaModelWrapper):
            print(f"Optuna 최적화 중 ({n_trials} trials)...")
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
        "df_resampled": df_resampled,
        "df_features": df_feat,
        "outlier_summary": outlier_summary,
        "imputation_summary": imputation_summary_1,
        "X": X, "y": y,
        "splits": splits_scaled,
        "top_features": top_features,
        "scaler": scaler,
        "results": results,
        "metric_table": metric_table,
        "fitted_models": fitted_models
    }


# ============================================================================
# Sliding Window 파이프라인
# ============================================================================

from .sliding_window import (
    create_sliding_windows, 
    flatten_windows_for_ml, 
    create_feature_names_for_flattened_windows,
    print_window_info
)
from .save_results import save_all_results


def run_sliding_window_pipeline(
    dfs,
    mode,
    window_size=24,
    horizon=1,
    stride=1,
    time_col_map=None,
    tz=None,
    resample_rule="1h",
    resample_agg="mean",
    feature_cfg=FeatureConfig(),
    split_cfg=SplitConfig(),
    imputation_cfg=ImputationConfig(),
    outlier_cfg=OutlierConfig(),
    use_outlier_detection=True,
    use_imputation=True,
    n_top_features=50,
    cv_splits=3,
    n_trials=50,
    random_state=42,
    save_dir="results/ML",
    use_3d_models=False,
    save_results=True,
    save_predictions=True,
    save_sequences=True,
    save_model=True,
    sequence_format="npz"
):
    """
    Sliding Window 기반 파이프라인 실행
    
    과거 N개의 시간 스텝을 입력으로 사용해서 미래를 예측하는 방식입니다.
    
    전처리 순서:
    1. 시간축 정합
    2. 결측치 보간
    3. 이상치 처리
    4. 리샘플링
    5. 파생 특성 생성
    6. Sliding Window 생성 (과거 N시간 → 미래 예측)
    7. 데이터 분할
    8. 스케일링
    9. 피처 선택
    10. 모델 학습 및 평가
    
    Parameters:
    -----------
    dfs : dict
        데이터프레임 딕셔너리
    mode : str
        예측 모드 (flow/tms/modelA/modelB/modelC)
    window_size : int
        과거 몇 개의 시간 스텝을 볼 것인지 (기본: 24 = 과거 24시간)
    horizon : int
        미래 몇 스텝 후를 예측할 것인지 (기본: 1 = 다음 시간)
    stride : int
        윈도우 이동 간격 (기본: 1 = 매 시간마다)
    use_3d_models : bool
        3D 입력을 지원하는 모델 사용 여부 (현재 미지원)
        False: 평탄화해서 일반 ML 모델 사용
    save_results : bool
        결과 저장 여부 (기본: True)
    save_predictions : bool
        예측값 저장 여부 (기본: True)
    save_sequences : bool
        시퀀스 데이터 저장 여부 (기본: True)
    save_model : bool
        모델 저장 여부 (기본: True)
    sequence_format : str
        시퀀스 저장 형식 ('npz', 'pickle', 'csv', 기본: 'npz')
    ... (나머지 파라미터는 run_improved_pipeline과 동일)
        
    Returns:
    --------
    dict : 파이프라인 실행 결과
    """
    
    print(f"\n{'='*60}")
    print(f"Sliding Window 파이프라인 - 모드: {mode.upper()}")
    print(f"윈도우 크기: {window_size} 시간 스텝")
    print(f"예측 horizon: {horizon} 스텝")
    print(f"{'='*60}")
    
    target_cols = get_target_cols(mode)
    
    # ========================================
    # 1-5단계: 기본 전처리 (개선 파이프라인과 동일)
    # ========================================
    print("\n[1/10] 시간축 정합 중...")
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
        
        dfs_indexed[name] = dfs_indexed[name][~dfs_indexed[name].index.duplicated(keep='first')]
    
    df_all = merge_sources_on_time(dfs_indexed, how="outer")
    
    # 메타데이터 컬럼 제거
    metadata_cols = ['data_save_dt', 'YYMMDDHHMI_368', 'YYMMDDHHMI_541', 'YYMMDDHHMI_569', 
                     'STN_368', 'STN_541', 'STN_569']
    cols_to_drop = [col for col in metadata_cols if col in df_all.columns]
    if cols_to_drop:
        df_all = df_all.drop(columns=cols_to_drop)
    
    # 결측치 보간
    if use_imputation:
        print("\n[2/10] 결측치 보간 중...")
        df_all = impute_missing_with_strategy(df_all, freq=resample_rule, config=imputation_cfg, add_mask=True)
    
    # 이상치 처리
    if use_outlier_detection:
        print("\n[3/10] 이상치 탐지 및 처리 중...")
        df_all = detect_and_handle_outliers(df_all, config=outlier_cfg, add_mask=True)
        if use_imputation:
            df_all = impute_missing_with_strategy(df_all, freq=resample_rule, config=imputation_cfg, add_mask=False)
    
    # 리샘플링
    print(f"\n[4/10] 리샘플링 중 ({resample_rule})...")
    df_resampled = resample_hourly(df_all, rule=resample_rule, agg=resample_agg)
    
    # 파생 특성 생성
    print("\n[5/10] 파생 특성 생성 중...")
    exclude_cols = get_exclude_features(mode, target_cols)
    df_feat = build_features(
        df_hourly=df_resampled,
        target_cols=target_cols,
        exclude_cols=exclude_cols,
        feature_base_cols=None,
        mode=mode,
        cfg=feature_cfg
    )
    
    # 지도학습 데이터셋 생성
    print(f"\n[5.5/10] 지도학습 데이터셋 생성 중...")
    print(f"  전처리 완료 데이터: {len(df_feat)} 샘플, {len(df_feat.columns)} 컬럼")
    
    # 최대 윈도우 크기 계산 (lag + rolling)
    max_lag = max(feature_cfg.lag_hours) if feature_cfg.lag_hours else 24
    max_roll = max(feature_cfg.roll_hours) if feature_cfg.roll_hours else 24
    max_window = max(max_lag, max_roll)
    
    print(f"  최대 윈도우 크기: {max_window}시간 (lag={max_lag}, roll={max_roll})")
    
    X, y = make_supervised_dataset(
        df_feat, 
        target_cols=target_cols, 
        exclude_cols=exclude_cols, 
        dropna=True,
        drop_initial_nan_only=True,  # 초반 NaN만 제거
        max_lag_window=max_window
    )
    
    if len(X) == 0:
        print("\n❌ 에러: 모든 데이터가 제거되었습니다!")
        print("\n가능한 원인:")
        print("  1. 원본 데이터가 너무 적음")
        print("  2. 결측치 보간이 제대로 되지 않음")
        print("  3. 이상치 처리로 너무 많은 값이 NaN으로 변환됨")
        raise ValueError("데이터셋이 비어있습니다. 전처리 과정을 확인하세요.")
    
    print(f"\n전처리 완료: {len(X)} 샘플, {X.shape[1]} 피처")
    
    # ========================================
    # 6단계: Sliding Window 생성
    # ========================================
    print(f"\n[6/10] Sliding Window 생성 중...")
    print(f"  윈도우 크기: {window_size} 시간 스텝")
    print(f"  예측 horizon: {horizon} 스텝 후")
    print(f"  윈도우 이동 간격: {stride} 스텝")
    print(f"  입력 데이터: {X.shape}")
    print(f"  타겟 데이터: {y.shape}")
    
    # 윈도우 생성 가능 여부 확인
    min_required_samples = window_size + horizon
    if len(X) < min_required_samples:
        raise ValueError(
            f"데이터가 부족합니다. "
            f"필요: {min_required_samples}개 (window_size={window_size} + horizon={horizon}), "
            f"실제: {len(X)}개"
        )
    
    X_seq, y_seq = create_sliding_windows(
        X, y, 
        window_size=window_size, 
        horizon=horizon, 
        stride=stride
    )
    
    print(f"  생성된 윈도우: {len(X_seq)}개")
    
    print_window_info(X_seq, y_seq, window_size)
    
    # ========================================
    # 7단계: 데이터 분할 (윈도우 단위)
    # ========================================
    print("\n[7/10] 데이터 분할 중...")
    n_samples = len(X_seq)
    train_size = int(n_samples * split_cfg.train_ratio)
    valid_size = int(n_samples * split_cfg.valid_ratio)
    
    X_train_seq = X_seq[:train_size]
    y_train = y_seq[:train_size]
    
    X_valid_seq = X_seq[train_size:train_size + valid_size]
    y_valid = y_seq[train_size:train_size + valid_size]
    
    X_test_seq = X_seq[train_size + valid_size:]
    y_test = y_seq[train_size + valid_size:]
    
    print(f"  Train: {len(X_train_seq)} 윈도우")
    print(f"  Valid: {len(X_valid_seq)} 윈도우")
    print(f"  Test:  {len(X_test_seq)} 윈도우")
    
    # ========================================
    # 7.5단계: 피처 선택 (평탄화 전) - 메모리 절약
    # ========================================
    print(f"\n[7.5/10] 피처 선택 (평탄화 전)...")
    print(f"  현재 특성 수: {X_seq.shape[2]}개")
    print(f"  평탄화 시 예상 특성 수: {X_seq.shape[2] * window_size:,}개")
    
    # 메모리 체크
    estimated_size_gb = (X_train_seq.shape[0] * X_train_seq.shape[1] * X_train_seq.shape[2] * 8) / (1024**3)
    print(f"  예상 메모리 사용량: {estimated_size_gb:.2f} GB")
    
    # 메모리가 너무 크면 피처 선택 먼저 수행
    if estimated_size_gb > 5 or (X_seq.shape[2] * window_size) > 50000:
        print(f"\n⚠️  메모리 절약을 위해 피처 선택을 먼저 수행합니다")
        
        # 각 시간 스텝의 마지막 값만 사용하여 피처 중요도 계산
        X_last_timestep = X_seq[:, -1, :]  # (n_samples, n_features)
        y_for_selection = y_seq
        
        # DataFrame으로 변환
        X_last_df = pd.DataFrame(X_last_timestep, columns=X.columns)
        y_df = pd.DataFrame(y_for_selection, columns=target_cols)
        
        # 피처 선택 (Train 데이터만 사용)
        X_train_last = X_last_df[:train_size]
        y_train_for_sel = y_df[:train_size]
        
        # 최대 선택 특성 수 계산 (평탄화 후 메모리 고려)
        max_features_for_memory = min(n_top_features, 100)  # 최대 100개
        
        print(f"  선택할 특성 수: {max_features_for_memory}개")
        
        from src.feature_selection import select_top_features
        selected_features = select_top_features(
            X_train_last, 
            y_train_for_sel,
            n_features=max_features_for_memory,
            random_state=random_state
        )
        
        print(f"  선택된 특성: {len(selected_features)}개")
        
        # 선택된 특성의 인덱스 찾기
        selected_indices = [X.columns.get_loc(feat) for feat in selected_features]
        
        # 3D 배열에서 선택된 특성만 추출
        X_train_seq = X_train_seq[:, :, selected_indices]
        X_valid_seq = X_valid_seq[:, :, selected_indices]
        X_test_seq = X_test_seq[:, :, selected_indices]
        
        # 선택된 특성 이름 업데이트
        X_columns_selected = [X.columns[i] for i in selected_indices]
        
        print(f"  축소 후 shape: {X_train_seq.shape}")
        print(f"  평탄화 후 예상 특성 수: {len(selected_features) * window_size:,}개")
        
        # 메모리 재계산
        estimated_size_gb = (X_train_seq.shape[0] * X_train_seq.shape[1] * X_train_seq.shape[2] * 8) / (1024**3)
        print(f"  축소 후 예상 메모리: {estimated_size_gb:.2f} GB")
    else:
        X_columns_selected = X.columns.tolist()
    
    # ========================================
    # 8단계: 평탄화 (일반 ML 모델용)
    # ========================================
    if not use_3d_models:
        print("\n[8/10] 윈도우 평탄화 중 (ML 모델용)...")
        
        # 메모리 체크 (최종)
        if estimated_size_gb > 10:
            print(f"\n❌ 에러: 여전히 메모리가 부족합니다 ({estimated_size_gb:.2f} GB)")
            print("\n추가 해결 방법:")
            print("  1. window_size 줄이기: --window-size 6")
            print("  2. stride 늘리기: --stride 2")
            print("  3. 피처 수 더 줄이기: --n-features 30")
            raise MemoryError(
                f"메모리 부족: {estimated_size_gb:.2f} GB 필요. "
                f"window_size를 줄이거나 stride를 늘리세요."
            )
        
        X_train_flat = flatten_windows_for_ml(X_train_seq)
        X_valid_flat = flatten_windows_for_ml(X_valid_seq)
        X_test_flat = flatten_windows_for_ml(X_test_seq)
        
        print(f"  평탄화 전: {X_train_seq.shape}")
        print(f"  평탄화 후: {X_train_flat.shape}")
        
        # 특성 이름 생성 (선택된 특성만)
        feature_names = create_feature_names_for_flattened_windows(
            X_columns_selected, 
            window_size
        )
        
        # DataFrame으로 변환
        X_train_df = pd.DataFrame(X_train_flat, columns=feature_names)
        X_valid_df = pd.DataFrame(X_valid_flat, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_flat, columns=feature_names)
        
        # y도 DataFrame으로 변환
        y_train_df = pd.DataFrame(y_train, columns=target_cols)
        y_valid_df = pd.DataFrame(y_valid, columns=target_cols)
        y_test_df = pd.DataFrame(y_test, columns=target_cols)
    else:
        print("\n[8/10] 3D 모델은 현재 지원되지 않습니다...")
        raise NotImplementedError("3D 모델 지원이 제거되었습니다. use_3d_models=False로 설정하세요.")
    
    # ========================================
    # 9단계: 스케일링 (평탄화된 데이터)
    # ========================================
    if not use_3d_models:
        print("\n[9/10] 스케일링 중 (Train 기준)...")
        X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_data(
            X_train_df, X_valid_df, X_test_df
        )
        
        # ========================================
        # 10단계: 추가 피처 선택 (평탄화 후)
        # ========================================
        # 이미 피처 선택을 했으면 스킵
        if 'selected_features' in locals():
            print(f"\n[10/10] 피처 선택 이미 완료 (평탄화 전에 수행됨)")
            splits_final = {
                "train": (X_train_scaled, y_train_df),
                "valid": (X_valid_scaled, y_valid_df),
                "test": (X_test_scaled, y_test_df)
            }
            top_features = X_train_scaled.columns.tolist()
        else:
            print(f"\n[10/10] 특성 선택 중 (상위 {n_top_features}개)...")
            top_features = select_top_features(
                X_train_scaled, y_train_df, 
                n_features=min(n_top_features, X_train_scaled.shape[1]), 
                random_state=random_state
            )
            
            X_train_selected = X_train_scaled[top_features]
            X_valid_selected = X_valid_scaled[top_features]
            X_test_selected = X_test_scaled[top_features]
            
            splits_final = {
                "train": (X_train_selected, y_train_df),
                "valid": (X_valid_selected, y_valid_df),
                "test": (X_test_selected, y_test_df)
            }
    else:
        # 3D 모델은 스케일링/피처 선택 생략 (별도 처리 필요)
        print("\n[9-10/10] 3D 모델은 스케일링/피처 선택 생략...")
        splits_final = {
            "train": (X_train_df, y_train_df),
            "valid": (X_valid_df, y_valid_df),
            "test": (X_test_df, y_test_df)
        }
        scaler = None
        top_features = None
    
    # ========================================
    # 11단계: 모델 학습 및 평가
    # ========================================
    if not use_3d_models:
        print(f"\n[11/10] 모델 학습 및 평가 중...")
        zoo = build_model_zoo_with_optuna(cv_splits=cv_splits, n_trials=n_trials, random_state=random_state)
        
        results = {}
        fitted_models = {}
        
        for model_name, base_model in zoo.items():
            print(f"\n{'='*60}")
            print(f"모델 학습: {model_name}")
            print(f"{'='*60}")
            
            res, fitted_model = fit_and_evaluate_improved(
                base_model, splits_final,
                model_name=model_name,
                mode=mode,
                target_cols=target_cols
            )
            
            results[model_name] = res
            fitted_models[model_name] = fitted_model
            
            print(f"\n  Train - R²: {res['train']['R2_mean']:.4f}, RMSE: {res['train']['RMSE_mean']:.2f}")
            print(f"  Valid - R²: {res['valid']['R2_mean']:.4f}, RMSE: {res['valid']['RMSE_mean']:.2f}")
            print(f"  Test  - R²: {res['test']['R2_mean']:.4f}, RMSE: {res['test']['RMSE_mean']:.2f}")
        
        metric_table = plot_metric_table(results, split="test")
        plot_r2_comparison(results, mode, save_dir=save_dir)
        
        print(f"\n{'='*60}")
        print("최종 결과 (Test Set)")
        print(f"{'='*60}")
        print(metric_table.to_string(index=False))
    else:
        print("\n3D 모델은 현재 지원되지 않습니다.")
        results = None
        fitted_models = None
        metric_table = None
    
    return {
        "mode": mode,
        "target_cols": target_cols,
        "window_size": window_size,
        "horizon": horizon,
        "stride": stride,
        "df_resampled": df_resampled,
        "df_features": df_feat,
        "X_original": X,
        "y_original": y,
        "X_seq": X_seq,
        "y_seq": y_seq,
        "splits": splits_final,
        "top_features": top_features,
        "scaler": scaler,
        "results": results,
        "metric_table": metric_table,
        "fitted_models": fitted_models,
        "saved_files": save_all_results(
            {
                "mode": mode,
                "target_cols": target_cols,
                "window_size": window_size,
                "horizon": horizon,
                "stride": stride,
                "X_original": X,
                "y_original": y,
                "X_seq": X_seq,
                "y_seq": y_seq,
                "splits": splits_final,
                "top_features": top_features,
                "scaler": scaler,
                "results": results,
                "metric_table": metric_table,
                "fitted_models": fitted_models
            },
            save_dir=save_dir,
            save_predictions_flag=save_predictions,
            save_sequences_flag=save_sequences,
            save_model_flag=save_model,
            sequence_format=sequence_format
        ) if save_results else None
    }



# ============================================================================
# 딥러닝 파이프라인은 제거되었습니다
# ============================================================================

