"""
파이프라인 오케스트레이션 모듈
전체 학습 파이프라인 실행
"""

import pandas as pd
from .io import set_datetime_index, summarize_available_period, merge_sources_on_time
from .preprocess import drop_missing_rows, resample_hourly, check_continuity
from .features import build_features, make_supervised_dataset, FeatureConfig
from .split import time_split, SplitConfig
from .models import build_model_zoo, wrap_multioutput_if_needed
from .metrics import fit_and_evaluate, plot_metric_table


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
    random_state=42
):
    """
    전체 학습 파이프라인 실행
    
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
        리샘플링 전 결측치 제거할 컬럼
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
    random_state : int
        랜덤 시드
        
    Returns:
    --------
    dict : 파이프라인 실행 결과
    """
    
    target_cols = get_target_cols(mode)

    # 단계 1) 시간 인덱스 설정
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

    period_summary = summarize_available_period(dfs_indexed)

    # 병합
    df_all = merge_sources_on_time(dfs_indexed, how="outer")

    # 단계 2) 결측치 행 제거 (원본)
    df_all_clean = drop_missing_rows(df_all, cols=dropna_cols_before_resample)

    # 단계 3) 1시간 단위로 리샘플링
    df_hourly = resample_hourly(df_all_clean, rule=resample_rule, agg=resample_agg)

    # 단계 4) 피처 생성
    df_feat = build_features(
        df_hourly=df_hourly,
        target_cols=target_cols,
        feature_base_cols=feature_base_cols,
        cfg=feature_cfg
    )

    # 단계 5) 연속성 확인
    continuity = check_continuity(df_hourly.dropna(how="all"), freq=resample_rule)

    # 지도학습 데이터셋 X, y 생성
    X, y = make_supervised_dataset(df_feat, target_cols=target_cols, dropna=True)

    # 단계 6) 데이터 분할
    splits = time_split(X, y, cfg=split_cfg)

    # 단계 7) 모델 생성
    zoo = build_model_zoo(random_state=random_state)

    # 단계 8) 평가
    results = {}
    fitted_models = {}
    for model_name, base_model in zoo.items():
        model = wrap_multioutput_if_needed(base_model, y)
        res = fit_and_evaluate(model, splits)
        results[model_name] = res
        fitted_models[model_name] = model

    metric_table = plot_metric_table(results, split="test")

    return {
        "mode": mode,
        "target_cols": target_cols,
        "period_summary": period_summary,
        "df_merged": df_all,
        "df_hourly": df_hourly,
        "df_features": df_feat,
        "continuity": continuity,
        "X": X, "y": y,
        "splits": splits,
        "results": results,
        "metric_table": metric_table,
        "fitted_models": fitted_models
    }
