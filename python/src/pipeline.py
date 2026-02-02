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
    X, y = make_supervised_dataset(df_feat, target_cols=target_cols, exclude_cols=exclude_cols, dropna=True)
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
    X, y = make_supervised_dataset(df_feat, target_cols=target_cols, exclude_cols=exclude_cols, dropna=True)
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
        3D 입력을 지원하는 모델 사용 여부 (LSTM 등)
        False: 평탄화해서 일반 ML 모델 사용
        True: 3D 그대로 LSTM/RNN 사용
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
    X, y = make_supervised_dataset(df_feat, target_cols=target_cols, exclude_cols=exclude_cols, dropna=True)
    print(f"전처리 완료: {len(X)} 샘플, {X.shape[1]} 피처")
    
    # ========================================
    # 6단계: Sliding Window 생성
    # ========================================
    print(f"\n[6/10] Sliding Window 생성 중...")
    print(f"  윈도우 크기: {window_size} 시간 스텝")
    print(f"  예측 horizon: {horizon} 스텝 후")
    print(f"  윈도우 이동 간격: {stride} 스텝")
    
    X_seq, y_seq = create_sliding_windows(
        X, y, 
        window_size=window_size, 
        horizon=horizon, 
        stride=stride
    )
    
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
    # 8단계: 평탄화 (일반 ML 모델용)
    # ========================================
    if not use_3d_models:
        print("\n[8/10] 윈도우 평탄화 중 (ML 모델용)...")
        X_train_flat = flatten_windows_for_ml(X_train_seq)
        X_valid_flat = flatten_windows_for_ml(X_valid_seq)
        X_test_flat = flatten_windows_for_ml(X_test_seq)
        
        print(f"  평탄화 전: {X_train_seq.shape}")
        print(f"  평탄화 후: {X_train_flat.shape}")
        
        # 특성 이름 생성
        feature_names = create_feature_names_for_flattened_windows(
            X.columns.tolist(), 
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
        print("\n[8/10] 3D 형태 유지 (LSTM/RNN용)...")
        # 3D 모델용은 평탄화하지 않음
        X_train_df = X_train_seq
        X_valid_df = X_valid_seq
        X_test_df = X_test_seq
        y_train_df = pd.DataFrame(y_train, columns=target_cols)
        y_valid_df = pd.DataFrame(y_valid, columns=target_cols)
        y_test_df = pd.DataFrame(y_test, columns=target_cols)
    
    # ========================================
    # 9단계: 스케일링 (평탄화된 데이터)
    # ========================================
    if not use_3d_models:
        print("\n[9/10] 스케일링 중 (Train 기준)...")
        X_train_scaled, X_valid_scaled, X_test_scaled, scaler = scale_data(
            X_train_df, X_valid_df, X_test_df
        )
        
        # ========================================
        # 10단계: 피처 선택
        # ========================================
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
        print("\n3D 모델 학습은 별도 LSTM 파이프라인을 사용하세요.")
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
# LSTM 파이프라인
# ============================================================================

def run_lstm_pipeline(
    dfs,
    mode,
    window_size=24,
    horizon=1,
    stride=1,
    time_col_map=None,
    resample_rule="1h",
    n_top_features=50,
    cv_splits=3,
    n_trials=50,
    random_state=42,
    save_dir="results/DL",
    save_results=True,
    save_predictions=True,
    save_sequences=True,
    save_model=True,
    sequence_format="npz",
    # LSTM 하이퍼파라미터
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    batch_size=32,
    learning_rate=0.001,
    num_epochs=100,
    patience=10,
    weight_decay=0.0001,
    grad_clip=1.0,
    verbose=True
):
    """
    LSTM 딥러닝 파이프라인 실행
    
    Sliding Window 생성 후 LSTM 모델로 학습
    
    Parameters:
    -----------
    dfs : dict
        데이터프레임 딕셔너리 (flow, tms, aws 등)
    mode : str
        예측 모드 (flow, tms, modelA, modelB, modelC)
    window_size : int
        과거 몇 개의 시간 스텝을 볼 것인지
    horizon : int
        미래 몇 스텝 후를 예측할 것인지
    stride : int
        윈도우 이동 간격
    time_col_map : dict
        각 데이터프레임의 시간 컬럼 매핑
    resample_rule : str
        리샘플링 규칙 (예: '1h', '30min')
    n_top_features : int
        선택할 상위 특성 개수
    cv_splits : int
        TimeSeriesSplit 분할 수
    n_trials : int
        Optuna 시도 횟수 (피처 선택용)
    random_state : int
        랜덤 시드
    save_dir : str
        결과 저장 디렉토리
    save_results : bool
        결과 저장 여부
    save_predictions : bool
        예측값 저장 여부
    save_sequences : bool
        시퀀스 데이터 저장 여부
    save_model : bool
        모델 저장 여부
    sequence_format : str
        시퀀스 저장 형식 (npz, pickle, csv)
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
    grad_clip : float
        그래디언트 클리핑 값
    verbose : bool
        학습 과정 출력 여부
        
    Returns:
    --------
    dict
        파이프라인 결과
    """
    from src.models_dl import LSTMWrapper
    from src.metrics import compute_metrics
    import pandas as pd
    import numpy as np
    
    print("=" * 80)
    print("LSTM 딥러닝 파이프라인 시작")
    print("=" * 80)
    print(f"모드: {mode}")
    print(f"Window Size: {window_size}, Horizon: {horizon}, Stride: {stride}")
    print(f"LSTM 설정: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
    print("=" * 80)
    
    # ========================================
    # 1-6단계: Sliding Window 생성까지는 동일
    # ========================================
    print("\n[1-6/11] 데이터 전처리 및 Sliding Window 생성...")
    
    # 필요한 모듈 import
    from src.preprocess import (
        drop_missing_rows, 
        resample_hourly, 
        impute_missing_with_strategy, 
        detect_and_handle_outliers,
        ImputationConfig,
        OutlierConfig
    )
    from src.features import build_features, FeatureConfig
    from src.split import SplitConfig
    from src.io import set_datetime_index, merge_sources_on_time
    from src.sliding_window import create_sliding_windows, split_windowed_data, flatten_windows_for_ml, create_feature_names_for_flattened_windows
    
    # get_target_cols와 get_exclude_features는 이미 pipeline.py에 정의되어 있음
    target_cols = get_target_cols(mode)
    
    # 1단계: 시간축 정합
    dfs_indexed = {}
    for name, df in dfs.items():
        if df is None or len(df) == 0:
            dfs_indexed[name] = df
            continue
        
        if isinstance(df.index, pd.DatetimeIndex):
            dfs_indexed[name] = df.sort_index()
        else:
            time_col = time_col_map.get(name) if time_col_map else None
            if time_col and time_col in df.columns:
                dfs_indexed[name] = set_datetime_index(df, time_col)
            else:
                dfs_indexed[name] = df
    
    # 2단계: 데이터 병합
    df_merged = merge_sources_on_time(dfs_indexed, how="outer")
    
    # 3단계: 결측치 보간
    imputation_cfg = ImputationConfig()
    df_imputed = impute_missing_with_strategy(df_merged, freq=resample_rule, config=imputation_cfg)
    
    # 4단계: 이상치 처리
    outlier_cfg = OutlierConfig()
    df_outlier = detect_and_handle_outliers(df_imputed, config=outlier_cfg, add_mask=True)
    
    # 5단계: 리샘플링
    df_resampled = resample_hourly(df_outlier, rule=resample_rule, agg="mean")
    
    # 6단계: 파생 특성 생성
    feature_cfg = FeatureConfig()
    exclude_cols = get_exclude_features(mode, target_cols)
    df_feat = build_features(
        df_hourly=df_resampled,
        target_cols=target_cols,
        exclude_cols=exclude_cols,
        feature_base_cols=None,
        mode=mode,
        cfg=feature_cfg
    )
    
    # 7단계: 타겟 분리
    X = df_feat.drop(columns=target_cols, errors="ignore")
    y = df_feat[target_cols]
    
    # datetime 타입 컬럼 제거 (RandomForest가 학습할 수 없음)
    datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        print(f"  Datetime 컬럼 제거: {datetime_cols}")
        X = X.drop(columns=datetime_cols)
    
    # ⭐ NaN 제거 (Sliding Window 생성 전에 필수!)
    print(f"\n전처리 전 크기:")
    print(f"  특성 크기: {X.shape}")
    print(f"  타겟 크기: {y.shape}")
    
    # NaN이 있는 행 제거
    valid_mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    print(f"\nNaN 제거 후:")
    print(f"  특성 크기: {X_clean.shape} (제거: {X.shape[0] - X_clean.shape[0]}행)")
    print(f"  타겟 크기: {y_clean.shape}")
    print(f"  NaN 비율: {(1 - len(X_clean)/len(X))*100:.1f}%")
    
    # ========================================
    # 8단계: Sliding Window 생성
    # ========================================
    print(f"\n[7/11] Sliding Window 생성 중...")
    print(f"  Window Size: {window_size}, Horizon: {horizon}, Stride: {stride}")
    
    X_seq, y_seq = create_sliding_windows(
        X_clean.values, y_clean.values,
        window_size=window_size,
        horizon=horizon,
        stride=stride
    )
    
    print(f"  생성된 윈도우 수: {len(X_seq)}")
    print(f"  X_seq 크기: {X_seq.shape}")  # (n_windows, window_size, n_features)
    print(f"  y_seq 크기: {y_seq.shape}")  # (n_windows, n_targets)
    
    # ========================================
    # 9단계: 데이터 분할
    # ========================================
    print(f"\n[8/11] 데이터 분할 중...")
    
    split_cfg = SplitConfig(train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2)
    splits = split_windowed_data(
        X_seq, y_seq, 
        train_ratio=split_cfg.train_ratio,
        valid_ratio=split_cfg.valid_ratio,
        test_ratio=split_cfg.test_ratio
    )
    
    X_train_seq, y_train_seq = splits["train"]
    X_valid_seq, y_valid_seq = splits["valid"]
    X_test_seq, y_test_seq = splits["test"]
    
    print(f"  Train: {X_train_seq.shape[0]} 윈도우")
    print(f"  Valid: {X_valid_seq.shape[0]} 윈도우")
    print(f"  Test:  {X_test_seq.shape[0]} 윈도우")
    
    # ========================================
    # 10단계: 피처 선택 (평탄화 후)
    # ========================================
    skip_reshape = False  # reshape 건너뛰기 플래그
    
    if n_top_features > 0 and n_top_features < X_clean.shape[1]:
        print(f"\n[9/11] 피처 선택 중 (상위 {n_top_features}개)...")
        
        # 평탄화
        X_train_flat = flatten_windows_for_ml(X_train_seq)
        
        # 특성 이름 생성
        feature_names_flat = create_feature_names_for_flattened_windows(X_clean.columns.tolist(), window_size)
        
        # 피처 선택
        from src.feature_selection import select_top_features
        
        # DataFrame으로 변환 (select_top_features는 DataFrame을 받음)
        X_train_flat_df = pd.DataFrame(X_train_flat, columns=feature_names_flat)
        y_train_df = pd.DataFrame(y_train_seq, columns=target_cols)
        
        top_features = select_top_features(
            X_train_flat_df, 
            y_train_df,
            n_features=min(n_top_features, len(feature_names_flat)),
            random_state=random_state
        )
        
        print(f"  선택된 특성: {len(top_features)}개")
        
        # 전체 데이터를 평탄화
        X_train_flat_all = flatten_windows_for_ml(X_train_seq)
        X_valid_flat_all = flatten_windows_for_ml(X_valid_seq)
        X_test_flat_all = flatten_windows_for_ml(X_test_seq)
        
        # 선택된 특성만 사용
        X_train_flat_df_all = pd.DataFrame(X_train_flat_all, columns=feature_names_flat)
        X_valid_flat_df = pd.DataFrame(X_valid_flat_all, columns=feature_names_flat)
        X_test_flat_df = pd.DataFrame(X_test_flat_all, columns=feature_names_flat)
        
        X_train_selected = X_train_flat_df_all[top_features]
        X_valid_selected = X_valid_flat_df[top_features]
        X_test_selected = X_test_flat_df[top_features]
        
        # 다시 3D로 reshape (LSTM 입력용)
        # 선택된 특성 수가 window_size의 배수가 아닐 수 있으므로 조정
        n_features_selected = len(top_features)
        
        # 각 타임스텝당 특성 수 계산
        if n_features_selected >= window_size:
            # 특성이 충분히 많으면 window_size로 나눔
            n_features_per_timestep = n_features_selected // window_size
            n_features_to_use = n_features_per_timestep * window_size
            
            # 사용할 특성만 선택 (나머지는 버림)
            X_train_selected = X_train_selected.iloc[:, :n_features_to_use]
            X_valid_selected = X_valid_selected.iloc[:, :n_features_to_use]
            X_test_selected = X_test_selected.iloc[:, :n_features_to_use]
            
            print(f"  특성 조정: {n_features_selected} → {n_features_to_use}개 사용")
        else:
            # 특성이 적으면 각 타임스텝에 1개씩 배치하고 나머지는 0으로 패딩
            print(f"  ⚠️  경고: 선택된 특성({n_features_selected})이 윈도우 크기({window_size})보다 적습니다.")
            print(f"  피처 선택을 건너뛰고 전체 특성을 사용합니다.")
            
            # 피처 선택 건너뛰기
            X_train_3d = X_train_seq
            X_valid_3d = X_valid_seq
            X_test_3d = X_test_seq
            
            print(f"  3D 데이터 크기:")
            print(f"    Train: {X_train_3d.shape}")
            print(f"    Valid: {X_valid_3d.shape}")
            print(f"    Test:  {X_test_3d.shape}")
            
            # 다음 단계로 건너뛰기
            skip_reshape = True
        
        if not skip_reshape:
            X_train_3d = X_train_selected.values.reshape(X_train_seq.shape[0], window_size, n_features_per_timestep)
            X_valid_3d = X_valid_selected.values.reshape(X_valid_seq.shape[0], window_size, n_features_per_timestep)
            X_test_3d = X_test_selected.values.reshape(X_test_seq.shape[0], window_size, n_features_per_timestep)
            
            print(f"  3D reshape 완료:")
            print(f"    Train: {X_train_3d.shape}")
            print(f"    Valid: {X_valid_3d.shape}")
            print(f"    Test:  {X_test_3d.shape}")
    else:
        print(f"\n[9/11] 피처 선택 건너뛰기 (전체 특성 사용)")
        X_train_3d = X_train_seq
        X_valid_3d = X_valid_seq
        X_test_3d = X_test_seq
    
    # ========================================
    # 11단계: LSTM 모델 학습
    # ========================================
    print(f"\n[10/11] LSTM 모델 학습 중...")
    
    lstm_model = LSTMWrapper(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        patience=patience,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        random_state=random_state,
        verbose=verbose
    )
    
    # 학습 (검증 데이터 포함)
    lstm_model.fit(X_train_3d, y_train_seq, X_val=X_valid_3d, y_val=y_valid_seq)
    
    # ========================================
    # 12단계: 평가
    # ========================================
    print(f"\n[11/11] 모델 평가 중...")
    
    # 예측
    y_train_pred = lstm_model.predict(X_train_3d)
    y_valid_pred = lstm_model.predict(X_valid_3d)
    y_test_pred = lstm_model.predict(X_test_3d)
    
    # 평가 지표 계산
    train_metrics = compute_metrics(y_train_seq, y_train_pred)
    valid_metrics = compute_metrics(y_valid_seq, y_valid_pred)
    test_metrics = compute_metrics(y_test_seq, y_test_pred)
    
    print(f"\n{'='*60}")
    print("평가 결과")
    print(f"{'='*60}")
    print(f"\nTrain:")
    print(f"  R²:   {train_metrics['R2_mean']:.4f}")
    print(f"  RMSE: {train_metrics['RMSE_mean']:.2f}")
    print(f"  MAPE: {train_metrics['MAPE_mean(%)']:.2f}%")
    
    print(f"\nValid:")
    print(f"  R²:   {valid_metrics['R2_mean']:.4f}")
    print(f"  RMSE: {valid_metrics['RMSE_mean']:.2f}")
    print(f"  MAPE: {valid_metrics['MAPE_mean(%)']:.2f}%")
    
    print(f"\nTest:")
    print(f"  R²:   {test_metrics['R2_mean']:.4f}")
    print(f"  RMSE: {test_metrics['RMSE_mean']:.2f}")
    print(f"  MAPE: {test_metrics['MAPE_mean(%)']:.2f}%")
    
    # 결과 딕셔너리 생성
    results = {
        "LSTM": {
            "train": train_metrics,
            "valid": valid_metrics,
            "test": test_metrics
        }
    }
    
    # 메트릭 테이블 생성
    metric_table = pd.DataFrame({
        "model": ["LSTM"],
        "R2_mean": [test_metrics["R2_mean"]],
        "RMSE_mean": [test_metrics["RMSE_mean"]],
        "MAPE_mean(%)": [test_metrics["MAPE_mean(%)"]]
    })
    
    print(f"\n{'='*60}")
    print("최종 결과 (Test Set)")
    print(f"{'='*60}")
    print(metric_table.to_string(index=False))
    
    # ========================================
    # 결과 저장
    # ========================================
    # LSTM은 3D 데이터를 사용하므로 splits_final을 3D로 유지
    splits_final_3d = {
        "train": (X_train_3d, y_train_seq),
        "valid": (X_valid_3d, y_valid_seq),
        "test": (X_test_3d, y_test_seq)
    }
    
    fitted_models = {"LSTM": lstm_model}
    
    saved_files = None
    if save_results:
        from src.save_results import save_all_results
        
        # LSTM은 예측값 저장을 건너뛰고 시퀀스와 모델만 저장
        saved_files = save_all_results(
            {
                "mode": mode,
                "target_cols": target_cols,
                "window_size": window_size,
                "horizon": horizon,
                "stride": stride,
                "X_original": X_clean.values,
                "y_original": y_clean.values,
                "X_seq": X_seq,
                "y_seq": y_seq,
                "splits": splits_final_3d,
                "top_features": top_features if 'top_features' in locals() else None,
                "scaler": None,  # LSTM은 내부에서 스케일링
                "results": results,
                "metric_table": metric_table,
                "fitted_models": fitted_models
            },
            save_dir=save_dir,
            save_predictions_flag=False,  # LSTM은 예측값 저장 건너뛰기 (3D 데이터 문제)
            save_sequences_flag=save_sequences,
            save_model_flag=save_model,
            sequence_format=sequence_format
        )
    
    return {
        "mode": mode,
        "target_cols": target_cols,
        "window_size": window_size,
        "horizon": horizon,
        "stride": stride,
        "df_resampled": df_resampled,
        "df_features": df_feat,
        "X_original": X_clean.values,
        "y_original": y_clean.values,
        "X_seq": X_seq,
        "y_seq": y_seq,
        "X_train_3d": X_train_3d,
        "X_valid_3d": X_valid_3d,
        "X_test_3d": X_test_3d,
        "y_train": y_train_seq,
        "y_valid": y_valid_seq,
        "y_test": y_test_seq,
        "splits": splits_final_3d,
        "top_features": top_features if 'top_features' in locals() else None,
        "lstm_model": lstm_model,
        "results": results,
        "metric_table": metric_table,
        "fitted_models": fitted_models,
        "history": lstm_model.history_,
        "saved_files": saved_files
    }

