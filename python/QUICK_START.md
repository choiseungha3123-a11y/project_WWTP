# 빠른 시작 가이드

## 🚀 빠른 시작

### 1단계: 의존성 설치

```bash
pip install -r requirements.txt
```

### 2단계: 학습 실행

```bash
python scripts/train.py --mode flow --data-root data/actual
```

## 📚 사용법

### CLI로 학습

```bash
# FLOW 모드 (유량 예측)
python scripts/train.py --mode flow --data-root data/actual

# TMS 모드 (수질 예측)
python scripts/train.py --mode tms --data-root data/actual

# 전체 모드 (유량 + 수질)
python scripts/train.py --mode all --data-root data/actual

# 시각화 포함
python scripts/train.py --mode flow --data-root data/actual --plot

# 커스텀 설정
python scripts/train.py \
  --mode flow \
  --data-root data/actual \
  --resample 5min \
  --train-ratio 0.7 \
  --valid-ratio 0.15 \
  --test-ratio 0.15 \
  --random-state 42
```

### Python 코드에서 사용

**기본 파이프라인:**

```python
from src.io import load_csvs, prep_flow, prep_aws
from src.pipeline import run_pipeline
from src.features import FeatureConfig
from src.split import SplitConfig

# 데이터 로드
df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569 = load_csvs("data/actual")
df_flow = prep_flow(df_flow)
df_aws = prep_aws(df_aws_368, df_aws_541, df_aws_569)

dfs = {"flow": df_flow, "tms": df_tms, "aws": df_aws}
time_col_map = {"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"}

# 커스텀 설정
feature_cfg = FeatureConfig(
    lag_hours=[1, 2, 3, 6, 12, 24],
    roll_hours=[3, 6, 12, 24]
)
split_cfg = SplitConfig(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)

# 파이프라인 실행
result = run_pipeline(
    dfs,
    mode="flow",
    time_col_map=time_col_map,
    resample_rule="1h",
    resample_agg="mean",
    feature_cfg=feature_cfg,
    split_cfg=split_cfg,
    random_state=42
)

# 결과 확인
print(result["metric_table"])
print(result["continuity"])
```

**개선된 파이프라인:**

```python
from src.io import load_csvs, prep_flow, prep_aws
from src.pipeline_improved import run_improved_pipeline
from src.features import FeatureConfig
from src.split import SplitConfig

# 데이터 로드
df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569 = load_csvs("data/actual")
df_flow = prep_flow(df_flow)
df_aws = prep_aws(df_aws_368, df_aws_541, df_aws_569)

dfs = {"flow": df_flow, "tms": df_tms, "aws": df_aws}
time_col_map = {"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"}

# 개선된 파이프라인 실행
result = run_improved_pipeline(
    dfs,
    mode="flow",
    time_col_map=time_col_map,
    resample_rule="1h",
    n_top_features=50,
    cv_splits=3,
    random_state=42,
    save_dir="results/ML/improved"
)

# 결과 확인
print(result["metric_table"])
print(f"선택된 피처: {len(result['top_features'])}개")
```

## 📁 프로젝트 구조

```
src/
├── __init__.py              # 패키지 초기화
├── io.py                   # 데이터 로드 및 전처리
├── preprocess.py           # 결측치 처리, 리샘플링
├── features.py             # 피처 엔지니어링
├── split.py                # 데이터 분할
├── models.py               # 기본 모델 정의
├── models_improved.py      # 개선된 모델 (GridSearchCV)
├── feature_selection.py    # 피처 선택
├── scaling.py              # StandardScaler
├── metrics.py              # 평가 지표
├── visualization.py        # Learning Curve 시각화
├── pipeline.py             # 기본 파이프라인
└── pipeline_improved.py    # 개선된 파이프라인

scripts/
├── train.py                # 기본 학습 스크립트
└── train_improved.py       # 개선된 학습 스크립트
```

## 🔄 파이프라인 비교

| 기능 | 기본 파이프라인 | 개선된 파이프라인 |
|------|----------------|------------------|
| 모델 | 6개 기본 모델 | 5개 모델 + GridSearchCV |
| 스케일링 | ❌ | ✅ StandardScaler |
| 피처 선택 | ❌ | ✅ 중요도 기반 |
| 하이퍼파라미터 튜닝 | ❌ | ✅ GridSearchCV |
| 교차 검증 | ❌ | ✅ TimeSeriesSplit |
| XGBoost | ❌ | ✅ Early Stopping |
| 시각화 | 기본 | Learning Curve 추가 |
| 속도 | 빠름 | 느림 (튜닝 포함) |
| 성능 | 기본 | 최적화됨 |

**추천:**
- 빠른 실험: `scripts/train.py` (기본 파이프라인)
- 최고 성능: `scripts/train_improved.py` (개선된 파이프라인)

## 💡 주요 옵션

- `--mode`: 예측 모드 (flow/tms/all)
- `--data-root`: 데이터 디렉토리 경로
- `--resample`: 리샘플링 규칙 (5min, 1h 등)
- `--train-ratio`: 학습 데이터 비율 (기본: 0.6)
- `--valid-ratio`: 검증 데이터 비율 (기본: 0.2)
- `--test-ratio`: 테스트 데이터 비율 (기본: 0.2)
- `--plot`: 최고 성능 모델 시각화
- `--random-state`: 랜덤 시드 (기본: 42)

## 📊 예상 출력

```
============================================================
WWTP 예측 모델 학습 시작
============================================================
모드: flow
데이터 경로: data/actual
리샘플링: 5min
병합 방식: inner
집계 방법: mean
============================================================

[1/8] 데이터 로드 중...
[3/8] 파이프라인 실행 중...

============================================================
데이터 기간 요약
============================================================
  source                start                  end  n_rows
    flow  2023-01-01 00:00:00  2023-12-31 23:55:00  105120
     tms  2023-01-01 00:00:00  2023-12-31 23:55:00  105120
     aws  2023-01-01 00:00:00  2023-12-31 23:55:00  105120

============================================================
모델 성능 (테스트 데이터)
============================================================
              model   R2_mean  RMSE_mean  MAPE_mean(%)
  HistGBR           0.950000   0.150000          5.000
  RandomForest      0.945000   0.155000          5.200
  Ridge             0.920000   0.180000          6.000
  LinearRegression  0.918000   0.182000          6.100
  ElasticNet        0.915000   0.185000          6.200
  Lasso             0.910000   0.190000          6.500

============================================================
학습 완료!
============================================================
```

## 🎯 다음 단계

1. 다른 모드 시도 (`tms`, `all`)
2. 하이퍼파라미터 조정
3. 피처 엔지니어링 실험
4. 결과 분석 및 시각화
