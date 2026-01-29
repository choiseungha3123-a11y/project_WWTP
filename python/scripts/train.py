"""
학습 스크립트
CLI를 통한 모델 학습 실행

사용법:
  python scripts/train.py --mode flow --data-root data/actual --resample 5min
"""

import sys
from pathlib import Path
import argparse
import warnings

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.io import load_csvs, prep_flow, prep_aws, set_datetime_index, merge_sources_on_time
from src.pipeline import run_pipeline
from src.features import FeatureConfig
from src.split import SplitConfig
from src.metrics import plot_predictions
import numpy as np

warnings.filterwarnings("ignore")


def build_argparser():
    """CLI 인자 파서 생성"""
    p = argparse.ArgumentParser(description="WWTP 예측 모델 학습")
    p.add_argument("--mode", choices=["flow", "tms", "all"], default="flow",
                   help="예측 모드: flow(유량), tms(수질), all(전체)")
    p.add_argument("--data-root", default="data/actual",
                   help="*_Actual.csv 및 AWS_*.csv 파일이 있는 디렉토리")
    p.add_argument("--resample", default="5min",
                   help="Pandas 리샘플링 규칙, 예: 5min, 1h")
    p.add_argument("--how", default="inner", choices=["inner", "outer", "left", "right"],
                   help="데이터 병합 방식")
    p.add_argument("--agg", default="mean",
                   help="집계 방법: mean 또는 'auto' (RN*는 sum, 나머지는 mean)")
    p.add_argument("--random-state", type=int, default=42,
                   help="랜덤 시드")
    p.add_argument("--train-ratio", type=float, default=0.6,
                   help="학습 데이터 비율")
    p.add_argument("--valid-ratio", type=float, default=0.2,
                   help="검증 데이터 비율")
    p.add_argument("--test-ratio", type=float, default=0.2,
                   help="테스트 데이터 비율")
    p.add_argument("--plot", action="store_true",
                   help="최고 성능 모델의 예측 결과 시각화")
    return p


def main():
    """메인 실행 함수"""
    args = build_argparser().parse_args()

    print("=" * 60)
    print("WWTP 예측 모델 학습 시작")
    print("=" * 60)
    print(f"모드: {args.mode}")
    print(f"데이터 경로: {args.data_root}")
    print(f"리샘플링: {args.resample}")
    print(f"병합 방식: {args.how}")
    print(f"집계 방법: {args.agg}")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/8] 데이터 로드 중...")
    df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569 = load_csvs(args.data_root)
    df_flow = prep_flow(df_flow)
    df_aws = prep_aws(df_aws_368, df_aws_541, df_aws_569)

    dfs = {"flow": df_flow, "tms": df_tms, "aws": df_aws}
    time_col_map = {"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"}

    # 2. 집계 방법 설정
    if args.agg == "auto":
        print("[2/8] 자동 집계 방법 설정 중...")
        merged_tmp = merge_sources_on_time(
            {k: set_datetime_index(v, time_col_map[k]) for k, v in dfs.items()},
            how=args.how
        )
        num_cols = merged_tmp.select_dtypes(include=[np.number]).columns
        resample_agg = {c: ("sum" if str(c).startswith("RN") else "mean") for c in num_cols}
    else:
        resample_agg = args.agg

    # 3. 분할 설정
    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )

    # 4. 파이프라인 실행
    print("[3/8] 파이프라인 실행 중...")
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

    # 5. 결과 출력
    print("\n" + "=" * 60)
    print("데이터 기간 요약")
    print("=" * 60)
    print(out["period_summary"].to_string(index=False))

    print("\n" + "=" * 60)
    print("연속성 확인")
    print("=" * 60)
    for key, val in out["continuity"].items():
        print(f"{key}: {val}")

    print("\n" + "=" * 60)
    print("데이터셋 크기")
    print("=" * 60)
    print(f"전체: {len(out['X'])} 샘플")
    print(f"학습: {len(out['splits']['train'][0])} 샘플")
    print(f"검증: {len(out['splits']['valid'][0])} 샘플")
    print(f"테스트: {len(out['splits']['test'][0])} 샘플")
    print(f"피처 수: {out['X'].shape[1]}")
    print(f"타겟: {out['target_cols']}")

    print("\n" + "=" * 60)
    print("모델 성능 (테스트 데이터)")
    print("=" * 60)
    print(out["metric_table"].to_string(index=False))

    # 6. 최고 성능 모델 시각화 (선택사항)
    if args.plot:
        print("\n[8/8] 최고 성능 모델 시각화 중...")
        best_model_name = out["metric_table"].iloc[0]["model"]
        best_model = out["fitted_models"][best_model_name]
        X_test, y_test = out["splits"]["test"]
        y_pred = best_model.predict(X_test)
        plot_predictions(y_test, y_pred, title=f"TEST | {best_model_name}")

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
