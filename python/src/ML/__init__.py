"""
WWTP 예측 모델링 파이프라인
"""

__version__ = "1.0.0"

# 모듈 import
from src.ML.io import (
    load_csvs,
    set_datetime_index,
    summarize_available_period,
    merge_sources_on_time,
    prep_flow,
    prep_aws_station,
    prep_aws
)

from src.ML.preprocess import (
    drop_missing_rows,
    resample_hourly,
    check_continuity,
    impute_missing_with_strategy,
    ImputationConfig,
    summarize_imputation,
    detect_and_handle_outliers,
    OutlierConfig,
    summarize_outliers
)

from src.ML.features import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_target_history_features,
    add_rain_features,
    add_weather_features,
    add_tms_interaction_features,
    add_rain_tms_interaction_features,
    add_level_flow_features,
    add_rain_spatial_features,
    build_features,
    make_supervised_dataset,
    FeatureConfig
)

from src.ML.split import (
    time_split,
    SplitConfig
)

from src.ML.models import (
    build_model_zoo,
    build_model_zoo_with_optuna,
    wrap_multioutput_if_needed,
    OptunaModelWrapper
)

from src.ML.metrics import (
    compute_metrics,
    fit_and_evaluate,
    plot_predictions,
    plot_metric_table
)

from src.ML.feature_selection import (
    select_top_features
)

from src.ML.scaling import (
    scale_data
)

from src.ML.sliding_window import (
    create_sliding_windows,
    create_sliding_windows_with_index,
    flatten_windows_for_ml,
    create_feature_names_for_flattened_windows,
    split_windowed_data,
    print_window_info
)

from src.ML.visualization import (
    plot_learning_curve,
    plot_r2_comparison
)

from src.ML.save_results import (
    save_predictions,
    save_sequence_dataset,
    load_sequence_dataset,
    save_model_and_metadata,
    save_all_results
)

from src.ML.pipeline import (
    get_target_cols,
    get_exclude_features,
    run_pipeline,
    run_improved_pipeline,
    run_sliding_window_pipeline
)

__all__ = [
    # io
    'load_csvs',
    'set_datetime_index',
    'summarize_available_period',
    'merge_sources_on_time',
    'prep_flow',
    'prep_aws_station',
    'prep_aws',
    
    # preprocess
    'drop_missing_rows',
    'resample_hourly',
    'check_continuity',
    'impute_missing_with_strategy',
    'ImputationConfig',
    'summarize_imputation',
    'detect_and_handle_outliers',
    'OutlierConfig',
    'summarize_outliers',
    
    # features
    'add_time_features',
    'add_lag_features',
    'add_rolling_features',
    'add_target_history_features',
    'add_rain_features',
    'add_weather_features',
    'add_tms_interaction_features',
    'add_rain_tms_interaction_features',
    'add_level_flow_features',
    'add_rain_spatial_features',
    'build_features',
    'make_supervised_dataset',
    'FeatureConfig',
    
    # split
    'time_split',
    'SplitConfig',
    
    # models
    'build_model_zoo',
    'build_model_zoo_with_optuna',
    'wrap_multioutput_if_needed',
    'OptunaModelWrapper',
    
    # metrics
    'compute_metrics',
    'fit_and_evaluate',
    'plot_predictions',
    'plot_metric_table',
    
    # feature_selection
    'select_top_features',
    
    # scaling
    'scale_data',
    
    # sliding_window
    'create_sliding_windows',
    'create_sliding_windows_with_index',
    'flatten_windows_for_ml',
    'create_feature_names_for_flattened_windows',
    'split_windowed_data',
    'print_window_info',
    
    # visualization
    'plot_learning_curve',
    'plot_r2_comparison',
    
    # save_results
    'save_predictions',
    'save_sequence_dataset',
    'load_sequence_dataset',
    'save_model_and_metadata',
    'save_all_results',
    
    # pipeline
    'get_target_cols',
    'get_exclude_features',
    'run_pipeline',
    'run_improved_pipeline',
    'run_sliding_window_pipeline',
]
