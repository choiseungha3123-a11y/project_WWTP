"""
LSTM 파이프라인 사용 예제
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.DL.pipeline import DLPipeline
from src.DL.config import get_default_config


def example_1_basic_usage():
    """예제 1: 기본 사용법"""
    print("\n" + "="*60)
    print("예제 1: 기본 사용법")
    print("="*60)
    
    # 기본 설정 사용
    config = get_default_config()
    
    # 경로 설정
    data_dir = project_root / "data" / "actual"
    save_dir = project_root / "results" / "DL" / "example1"
    
    # 타겟 컬럼
    target_cols = ["Q_in"]
    
    # 파이프라인 실행
    pipeline = DLPipeline(config)
    pipeline.run(data_dir, target_cols, save_dir)


def example_2_custom_config():
    """예제 2: 커스텀 설정"""
    print("\n" + "="*60)
    print("예제 2: 커스텀 설정")
    print("="*60)
    
    from src.DL.config import PipelineConfig, WindowConfig, LSTMConfig, TrainingConfig
    
    # 커스텀 설정
    config = get_default_config()
    
    # 윈도우 크기 변경 (12시간)
    config.window.window_size = 12
    
    # LSTM 모델 크기 변경
    config.lstm.hidden_size = 128
    config.lstm.num_layers = 3
    
    # 학습 설정 변경
    config.training.batch_size = 64
    config.training.num_epochs = 50
    
    # 경로 설정
    data_dir = project_root / "data" / "actual"
    save_dir = project_root / "results" / "DL" / "example2"
    
    # 타겟 컬럼
    target_cols = ["Q_in"]
    
    # 파이프라인 실행
    pipeline = DLPipeline(config)
    pipeline.run(data_dir, target_cols, save_dir)


def example_3_step_by_step():
    """예제 3: 단계별 실행"""
    print("\n" + "="*60)
    print("예제 3: 단계별 실행")
    print("="*60)
    
    config = get_default_config()
    pipeline = DLPipeline(config)
    
    data_dir = project_root / "data" / "actual"
    save_dir = project_root / "results" / "DL" / "example3"
    target_cols = ["Q_in"]
    
    # 단계별 실행
    pipeline.step1_load_data(data_dir)
    pipeline.step2_align_time()
    pipeline.step3_impute_missing()
    pipeline.step4_handle_outliers()
    pipeline.step5_resample()
    pipeline.step6_create_features(target_cols)
    pipeline.step7_create_windows(target_cols)
    pipeline.step8_scale_data()
    
    # 중간 데이터 확인
    print(f"\n중간 데이터 확인:")
    print(f"  리샘플링 후: {pipeline.df_resampled.shape}")
    print(f"  특성 생성 후: {pipeline.df_features.shape}")
    print(f"  윈도우 생성 후: X={pipeline.X_seq.shape}, y={pipeline.y_seq.shape}")
    
    # 학습 및 평가
    pipeline.step9_train_model()
    pipeline.step10_evaluate()
    pipeline.step11_save_results(save_dir)


def example_4_multi_target():
    """예제 4: 다중 타겟 예측 (TMS)"""
    print("\n" + "="*60)
    print("예제 4: 다중 타겟 예측 (TMS)")
    print("="*60)
    
    config = get_default_config()
    
    # 경로 설정
    data_dir = project_root / "data" / "actual"
    save_dir = project_root / "results" / "DL" / "example4_tms"
    
    # 다중 타겟 (TMS 지표들)
    target_cols = ["TOC_VU", "PH_VU", "SS_VU", "TN_VU", "TP_VU", "FLUX_VU"]
    
    # 파이프라인 실행
    pipeline = DLPipeline(config)
    pipeline.run(data_dir, target_cols, save_dir)


def example_5_quick_test():
    """예제 5: 빠른 테스트 (작은 설정)"""
    print("\n" + "="*60)
    print("예제 5: 빠른 테스트")
    print("="*60)
    
    config = get_default_config()
    
    # 빠른 테스트를 위한 설정
    config.window.window_size = 6  # 6시간 윈도우
    config.window.stride = 2       # 2시간씩 이동 (샘플 수 감소)
    config.lstm.hidden_size = 32   # 작은 모델
    config.lstm.num_layers = 1
    config.training.batch_size = 16
    config.training.num_epochs = 10  # 10 에포크만
    config.training.patience = 3
    
    # 경로 설정
    data_dir = project_root / "data" / "actual"
    save_dir = project_root / "results" / "DL" / "example5_quick"
    
    # 타겟 컬럼
    target_cols = ["Q_in"]
    
    # 파이프라인 실행
    pipeline = DLPipeline(config)
    pipeline.run(data_dir, target_cols, save_dir)


if __name__ == "__main__":
    # 실행할 예제 선택
    import argparse
    
    parser = argparse.ArgumentParser(description="LSTM 파이프라인 사용 예제")
    parser.add_argument("--example", type=int, default=1, choices=[1, 2, 3, 4, 5],
                       help="실행할 예제 번호 (1-5)")
    args = parser.parse_args()
    
    examples = {
        1: example_1_basic_usage,
        2: example_2_custom_config,
        3: example_3_step_by_step,
        4: example_4_multi_target,
        5: example_5_quick_test
    }
    
    examples[args.example]()
    
    print("\n" + "="*60)
    print("예제 실행 완료!")
    print("="*60)
