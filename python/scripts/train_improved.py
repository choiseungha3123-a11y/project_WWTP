"""
개선된 학습 스크립트
GridSearchCV, 피처 선택, XGBoost, Early Stopping 포함

사용법:
  python scripts/train_improved.py --mode flow --data-root data/actual
"""

import sys
from pathlib import Path
import argparse
import warnings

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.io import load_csvs, prep_flow, prep_aws
from src.pipeline_improved import run_improved_pipeline
from src.features import FeatureConfig
from src.split import SplitConfig

warnings.filterwarnings("ignore")


def build_argparser():
    """CLI 인자 파서 생성"""
    p = argparse.ArgumentParser(description="WWTP 예측 모델 학습 (개선 버전)")
    p.add_argument("--mode", choices=["flow", "tms", "all"], default="flow",
                   help="예측 모드: flow(유량), tms(수질), all(전체)")
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
    p.add_argument("--n-features", type=int, default=50,
                   help="선택할 피처 개수")
    p.add_argument("--cv-splits", type=int, default=3,
                   help="TimeSeriesSplit 분할 수")
    p.add_argument("--save-dir", default="results/ML/improved",
                   help="결과 저장 디렉토리")
    return p


def main():
    """메인 실행 함수"""
    args = build_argparser().parse_args()

    print("=" * 60)
    print("WWTP 예측 모델 학습 (개선 버전)")
    print("=" * 60)
    print(f"모드: {args.mode}")
    print(f"데이터 경로: {args.data_root}")
    print(f"리샘플링: {args.resample}")
    print(f"피처 선택: 상위 {args.n_features}개")
    print(f"교차 검증: {args.cv_splits} splits")
    print("=" * 60)

    # 데이터 로드
    print("\n데이터 로드 중...")
    df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569 = load_csvs(args.data_root)
    df_flow = prep_flow(df_flow)
    df_aws = prep_aws(df_aws_368, df_aws_541, df_aws_569)

    dfs = {"flow": df_flow, "tms": df_tms, "aws": df_aws}
    time_col_map = {"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"}

    # 분할 설정
    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )

    # 파이프라인 실행
    print("\n개선된 파이프라인 실행 중...")
    result = run_improved_pipeline(
        dfs,
        mode=args.mode,
        time_col_map=time_col_map,
        tz=None,
        resample_rule=args.resample,
        resample_agg="mean",
        split_cfg=split_cfg,
        n_top_features=args.n_features,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
        save_dir=args.save_dir
    )

    # 최종 결과
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"\n최고 성능 모델: {result['metric_table'].iloc[0]['model']}")
    print(f"Test R²: {result['metric_table'].iloc[0]['R2_mean']:.4f}")
    print(f"Test RMSE: {result['metric_table'].iloc[0]['RMSE_mean']:.2f}")
    print(f"\n결과 저장 위치: {args.save_dir}")


if __name__ == "__main__":
    main()
