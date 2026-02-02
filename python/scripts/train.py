"""
통합 학습 스크립트
CLI를 통한 모델 학습 실행

사용법:
# 기본 파이프라인
python scripts/train.py --mode flow --data-root data/actual --resample 1h

# 개선된 파이프라인 (Optuna, 피처 선택, XGBoost, Early Stopping)
python scripts/train.py --mode flow --improved --n-features 50 --cv-splits 3 --n-trials 50

# Sliding Window 파이프라인 (과거 N시간 → 미래 예측)
python scripts/train.py --mode flow --sliding-window --window-size 24 --horizon 1

# Sliding Window + 개선 파이프라인
python scripts/train.py --mode flow --sliding-window --improved --window-size 24 --n-features 50
"""

import sys
from pathlib import Path
import argparse
import warnings
import os

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.io import load_csvs, prep_flow, prep_aws, set_datetime_index, merge_sources_on_time
from src.pipeline import run_pipeline, run_improved_pipeline, run_sliding_window_pipeline
from src.features import FeatureConfig
from src.split import SplitConfig
from src.preprocess import ImputationConfig, OutlierConfig
from src.metrics import plot_predictions
import numpy as np

# 경고 필터 설정
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Windows 환경에서 joblib 멀티프로세싱 안정성 향상
os.environ['LOKY_MAX_CPU_COUNT'] = str(max(1, os.cpu_count() - 1))  # CPU 1개는 시스템용으로 남김


def build_argparser():
    """CLI 인자 파서 생성"""
    p = argparse.ArgumentParser(description="WWTP 예측 모델 학습 (통합)")
    
    # 공통 옵션
    p.add_argument("--mode", 
                   choices=["flow", "tms", "modelA", "modelB", "modelC"], 
                   default="flow",
                   help="예측 모드: flow(유량), tms(전체 수질), modelA(TOC+SS), modelB(TN+TP), modelC(FLUX+pH)")
    p.add_argument("--data-root", default="data/actual",
                   help="*_Actual.csv 및 AWS_*.csv 파일이 있는 디렉토리")
    p.add_argument("--resample", default="1h",
                   help="Pandas 리샘플링 규칙, 예: 5min, 1h")
    p.add_argument("--random-state", type=int, default=42,
                   help="랜덤 시드")
    p.add_argument("--train-ratio", type=float, default=0.6,
                   help="학습 데이터 비율")
    p.add_argument("--valid-ratio", type=float, default=0.2,
                   help="검증 데이터 비율")
    p.add_argument("--test-ratio", type=float, default=0.2,
                   help="테스트 데이터 비율")
    
    # 파이프라인 선택
    p.add_argument("--improved", action="store_true",
                   help="개선된 파이프라인 사용 (Optuna, 피처 선택, Scaling)")
    p.add_argument("--sliding-window", action="store_true",
                   help="Sliding Window 파이프라인 사용 (과거 N시간 → 미래 예측)")
    p.add_argument("--lstm", action="store_true",
                   help="LSTM 딥러닝 파이프라인 사용 (Sliding Window + LSTM)")
    
    # 기본 파이프라인 전용
    p.add_argument("--how", default="outer", choices=["inner", "outer", "left", "right"],
                   help="데이터 병합 방식 (기본 파이프라인)")
    p.add_argument("--agg", default="mean",
                   help="집계 방법: mean 또는 'auto' (기본 파이프라인)")
    p.add_argument("--plot", action="store_true",
                   help="최고 성능 모델의 예측 결과 시각화 (기본 파이프라인)")
    
    # 개선된 파이프라인 전용
    p.add_argument("--n-features", type=int, default=50,
                   help="선택할 피처 개수 (개선/Sliding Window 파이프라인)")
    p.add_argument("--cv-splits", type=int, default=3,
                   help="TimeSeriesSplit 분할 수 (개선/Sliding Window 파이프라인)")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Optuna 시도 횟수 (개선/Sliding Window 파이프라인)")
    p.add_argument("--save-dir", default="results/ML",
                   help="결과 저장 디렉토리 (개선/Sliding Window 파이프라인)")
    
    # Sliding Window 전용
    p.add_argument("--window-size", type=int, default=24,
                   help="과거 몇 개의 시간 스텝을 볼 것인지 (Sliding Window/LSTM, 기본: 24시간)")
    p.add_argument("--horizon", type=int, default=1,
                   help="미래 몇 스텝 후를 예측할 것인지 (Sliding Window/LSTM, 기본: 1 = 다음 시간)")
    p.add_argument("--stride", type=int, default=1,
                   help="윈도우 이동 간격 (Sliding Window/LSTM, 기본: 1 = 매 시간마다)")
    p.add_argument("--use-3d", action="store_true",
                   help="3D 입력 모델 사용 (LSTM 등, Sliding Window, 현재는 미지원)")
    
    # LSTM 전용
    p.add_argument("--hidden-size", type=int, default=64,
                   help="LSTM 은닉층 유닛 수 (LSTM, 기본: 64)")
    p.add_argument("--num-layers", type=int, default=2,
                   help="LSTM 레이어 수 (LSTM, 기본: 2)")
    p.add_argument("--dropout", type=float, default=0.2,
                   help="드롭아웃 비율 (LSTM, 기본: 0.2)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="배치 크기 (LSTM, 기본: 32)")
    p.add_argument("--learning-rate", type=float, default=0.001,
                   help="학습률 (LSTM, 기본: 0.001)")
    p.add_argument("--num-epochs", type=int, default=100,
                   help="최대 에포크 수 (LSTM, 기본: 100)")
    p.add_argument("--patience", type=int, default=10,
                   help="조기 종료 patience (LSTM, 기본: 10)")
    p.add_argument("--weight-decay", type=float, default=0.0001,
                   help="L2 정규화 계수 (LSTM, 기본: 0.0001)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="그래디언트 클리핑 값 (LSTM, 기본: 1.0)")
    
    # 결과 저장 옵션
    p.add_argument("--no-save", action="store_true",
                   help="결과 저장 안 함")
    p.add_argument("--no-save-predictions", action="store_true",
                   help="예측값 저장 안 함")
    p.add_argument("--no-save-sequences", action="store_true",
                   help="시퀀스 데이터 저장 안 함")
    p.add_argument("--no-save-model", action="store_true",
                   help="모델 저장 안 함")
    p.add_argument("--sequence-format", default="npz", choices=["npz", "pickle", "csv"],
                   help="시퀀스 저장 형식 (기본: npz)")
    
    return p


def main():
    """메인 실행 함수"""
    args = build_argparser().parse_args()

    # 파이프라인 타입 결정
    if args.lstm:
        pipeline_type = "LSTM 딥러닝 파이프라인"
    elif args.sliding_window:
        pipeline_type = "Sliding Window 파이프라인"
        if args.improved:
            pipeline_type += " (개선)"
    elif args.improved:
        pipeline_type = "개선 파이프라인"
    else:
        pipeline_type = "기본 파이프라인"
    
    print("=" * 60)
    print(f"WWTP 예측 모델 학습 ({pipeline_type})")
    print("=" * 60)
    print(f"모드: {args.mode}")
    print(f"데이터 경로: {args.data_root}")
    print(f"리샘플링: {args.resample}")
    
    if args.sliding_window or args.lstm:
        print(f"윈도우 크기: {args.window_size} 시간 스텝")
        print(f"예측 horizon: {args.horizon} 스텝 후")
        print(f"윈도우 이동 간격: {args.stride} 스텝")
    
    if args.lstm:
        print(f"LSTM 설정: hidden_size={args.hidden_size}, num_layers={args.num_layers}, dropout={args.dropout}")
    
    if args.improved or args.sliding_window or args.lstm:
        print(f"피처 선택: 상위 {args.n_features}개")
        print(f"교차 검증: {args.cv_splits} splits")
        print(f"Optuna 시도: {args.n_trials} trials")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/8] 데이터 로드 중...")
    df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569 = load_csvs(args.data_root)
    df_flow = prep_flow(df_flow)
    df_aws = prep_aws(df_aws_368, df_aws_541, df_aws_569)

    dfs = {"flow": df_flow, "tms": df_tms, "aws": df_aws}
    time_col_map = {"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"}

    # 2. 분할 설정
    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )

    # 3. 파이프라인 실행
    if args.lstm:
        # ========================================
        # LSTM 딥러닝 파이프라인
        # ========================================
        print("[3/8] LSTM 딥러닝 파이프라인 실행 중...")
        
        from src.pipeline import run_lstm_pipeline
        
        out = run_lstm_pipeline(
            dfs,
            mode=args.mode,
            window_size=args.window_size,
            horizon=args.horizon,
            stride=args.stride,
            time_col_map=time_col_map,
            resample_rule=args.resample,
            n_top_features=args.n_features,
            cv_splits=args.cv_splits,
            n_trials=args.n_trials,
            random_state=args.random_state,
            save_dir=args.save_dir if args.save_dir != "results/ML" else "results/DL",
            save_results=not args.no_save,
            save_predictions=not args.no_save_predictions,
            save_sequences=not args.no_save_sequences,
            save_model=not args.no_save_model,
            sequence_format=args.sequence_format,
            # LSTM 하이퍼파라미터
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            patience=args.patience,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            verbose=True
        )
        
        # LSTM 결과 출력
        print("\n" + "=" * 60)
        print("LSTM 학습 정보")
        print("=" * 60)
        print(f"원본 데이터: {len(out['X_original'])} 샘플")
        print(f"윈도우 생성 후: {len(out['X_seq'])} 윈도우")
        print(f"감소율: {(1 - len(out['X_seq'])/len(out['X_original']))*100:.1f}%")
        print(f"  (윈도우 크기 {args.window_size} + horizon {args.horizon} 때문)")
        
        if out.get("top_features"):
            print("\n" + "=" * 60)
            print("선택된 피처")
            print("=" * 60)
            print(f"총 {len(out['top_features'])}개 피처 선택됨")
            print(f"상위 10개: {out['top_features'][:10]}")
        
        if out.get("history"):
            print("\n" + "=" * 60)
            print("학습 히스토리")
            print("=" * 60)
            print(f"최종 에포크: {len(out['history']['train_losses'])}")
            print(f"최고 검증 손실: {out['history']['best_val_loss']:.6f}")
        
    elif args.sliding_window:
        # ========================================
        # Sliding Window 파이프라인
        # ========================================
        print("[3/8] Sliding Window 파이프라인 실행 중...")
        
        if args.use_3d:
            print("\n⚠️  3D 모델은 현재 지원하지 않습니다.")
            print("LSTM 모델을 사용하려면 python/notebook/DL/flow_lstm_model.py를 실행하세요.")
            return
        
        out = run_sliding_window_pipeline(
            dfs,
            mode=args.mode,
            window_size=args.window_size,
            horizon=args.horizon,
            stride=args.stride,
            time_col_map=time_col_map,
            tz=None,
            resample_rule=args.resample,
            resample_agg="mean",
            split_cfg=split_cfg,
            n_top_features=args.n_features,
            cv_splits=args.cv_splits,
            n_trials=args.n_trials,
            random_state=args.random_state,
            save_dir=args.save_dir,
            use_3d_models=args.use_3d,
            save_results=not args.no_save,
            save_predictions=not args.no_save_predictions,
            save_sequences=not args.no_save_sequences,
            save_model=not args.no_save_model,
            sequence_format=args.sequence_format
        )
        
        # Sliding Window 결과 출력
        print("\n" + "=" * 60)
        print("Sliding Window 정보")
        print("=" * 60)
        print(f"원본 데이터: {len(out['X_original'])} 샘플")
        print(f"윈도우 생성 후: {len(out['X_seq'])} 윈도우")
        print(f"감소율: {(1 - len(out['X_seq'])/len(out['X_original']))*100:.1f}%")
        print(f"  (윈도우 크기 {args.window_size} + horizon {args.horizon} 때문)")
        
        if out.get("top_features"):
            print("\n" + "=" * 60)
            print("선택된 피처")
            print("=" * 60)
            print(f"총 {len(out['top_features'])}개 피처 선택됨")
            print(f"상위 10개: {out['top_features'][-10:]}")
        
    elif args.improved:
        # ========================================
        # 개선된 파이프라인
        # ========================================
        print("[3/8] 개선된 파이프라인 실행 중...")
        out = run_improved_pipeline(
            dfs,
            mode=args.mode,
            time_col_map=time_col_map,
            tz=None,
            resample_rule=args.resample,
            resample_agg="mean",
            split_cfg=split_cfg,
            n_top_features=args.n_features,
            cv_splits=args.cv_splits,
            n_trials=args.n_trials,
            random_state=args.random_state,
            save_dir=args.save_dir
        )
        
        # 개선된 파이프라인 결과 출력
        print("\n" + "=" * 60)
        print("선택된 피처")
        print("=" * 60)
        print(f"총 {len(out['top_features'])}개 피처 선택됨")
        print(f"상위 10개: {out['top_features'][-10:]}")
        
    else:
        # ========================================
        # 기본 파이프라인
        # ========================================
        print("[2/8] 집계 방법 설정 중...")
        if args.agg == "auto":
            merged_tmp = merge_sources_on_time(
                {k: set_datetime_index(v, time_col_map[k]) for k, v in dfs.items()},
                how=args.how
            )
            num_cols = merged_tmp.select_dtypes(include=[np.number]).columns
            resample_agg = {c: ("sum" if str(c).startswith("RN") else "mean") for c in num_cols}
        else:
            resample_agg = args.agg

        print("[3/8] 기본 파이프라인 실행 중...")
        out = run_pipeline(
            dfs,
            mode=args.mode,
            time_col_map=time_col_map,
            tz=None,
            resample_rule=args.resample,
            resample_agg=resample_agg,
            split_cfg=split_cfg,
            random_state=args.random_state,
        )

        # 기본 파이프라인 결과 출력
        if "period_summary" in out:
            print("\n" + "=" * 60)
            print("데이터 기간 요약")
            print("=" * 60)
            print(out["period_summary"].to_string(index=False))

        if "continuity" in out:
            print("\n" + "=" * 60)
            print("연속성 확인")
            print("=" * 60)
            for key, val in out["continuity"].items():
                print(f"{key}: {val}")

        # 최고 성능 모델 시각화 (선택사항)
        if args.plot:
            print("\n[8/8] 최고 성능 모델 시각화 중...")
            best_model_name = out["metric_table"].iloc[0]["model"]
            best_model = out["fitted_models"][best_model_name]
            X_test, y_test = out["splits"]["test"]
            y_pred = best_model.predict(X_test)
            plot_predictions(y_test, y_pred, title=f"TEST | {best_model_name}")

    # 공통 결과 출력
    if out.get("metric_table") is not None:
        print("\n" + "=" * 60)
        print("최고 성능 모델")
        print("=" * 60)
        best_model_name = out["metric_table"].iloc[0]["model"]
        best_r2 = out["metric_table"].iloc[0]["R2_mean"]
        best_rmse = out["metric_table"].iloc[0]["RMSE_mean"]
        print(f"모델: {best_model_name}")
        print(f"Test R²: {best_r2:.4f}")
        print(f"Test RMSE: {best_rmse:.2f}")

        print(f"\n결과 저장 위치: {args.save_dir}")
        
        if out.get("saved_files"):
            print("\n저장된 파일:")
            saved = out["saved_files"]
            if saved.get("predictions"):
                print(f"  📊 예측값: {len(saved['predictions'])}개 파일")
            if saved.get("sequences"):
                print(f"  📦 시퀀스: {len(saved['sequences'])}개 파일")
            if saved.get("models"):
                print(f"  🤖 모델: {len(saved['models'])}개 파일")

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
