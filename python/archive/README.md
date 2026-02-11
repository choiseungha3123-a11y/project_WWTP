# Archive - 이전 버전 코드 보관소

이 폴더에는 프로젝트 개발 과정에서 생성되었지만 현재는 사용하지 않는 이전 버전의 코드들이 보관되어 있습니다.

## 📁 폴더 구조

```
archive/
├── old_ML_versions/          # 이전 ML 모델 버전들
│   ├── improved/            # 개선 버전 v1, v2
│   ├── linear/              # 초기 선형 모델 실험
│   ├── baseline.py          # 초기 베이스라인 (primary)
│   └── train_ml_models.ipynb
├── old_DL_versions/          # 이전 DL 코드
│   ├── config.py            # 구버전 설정
│   └── flow_lstm_model.py   # 초기 LSTM 모델
├── old_src/                  # 이전 src/ 모듈 구조
│   ├── DL/                  # LSTM 파이프라인 모듈 (현재 notebook/DL/로 통합)
│   └── ML/                  # ML 파이프라인 모듈 (현재 미사용)
├── old_scripts/              # 이전 실행 스크립트
│   ├── DL/train_lstm.py     # LSTM 학습 스크립트
│   └── ML/train.py          # ML 학습 스크립트
├── old_notebooks/            # 테스트/실험용 노트북
│   ├── collect/             # 데이터 수집 스크립트
│   ├── feature/             # 이전 피처 엔지니어링 노트북 (modelA/B/C/FLOW)
│   ├── dl_preprocessing_test.ipynb
│   ├── preprocess_causal_mask.ipynb
│   └── train_standalone.ipynb
├── old_model/                # 이전 모델 가중치
│   ├── modelA_lstm_model.pth
│   ├── modelB_lstm_model.pth
│   └── modelC_lstm_model.pth
├── old_data/                 # 이전 데이터 가공 결과
│   ├── output/              # 구버전 예측 결과 및 피처 목록
│   └── processed/           # 구버전 전처리 데이터
└── old_results/              # 이전 시각화 결과
    ├── flow_diagnosis.png
    ├── test_predictions.png
    └── training_history.png
```

## 📝 버전 히스토리

### ML 모델 발전 과정

1. **Primary Baseline** (2026-01-27)
   - 위치: `old_ML_versions/baseline.py`
   - 최초 베이스라인 모델
   - 결측치 제거 방식 사용
   - 성능: Q_in, FLUX_VU만 양호

2. **Improved V1** (2026-01-27)
   - 위치: `old_ML_versions/improved/v1/`
   - StandardScaler 추가
   - GridSearchCV 하이퍼파라미터 튜닝
   - 문제점: 데이터 손실 95.8%, 심각한 과적합

3. **Improved V2** (2026-01-27)
   - 위치: `old_ML_versions/improved/v2/`
   - 선형 보간으로 결측치 처리
   - 도메인 피처 추가
   - 성능: Q_in R² 0.01 → 0.56으로 개선

4. **Linear Model Experiment** (2026-01-26)
   - 위치: `old_ML_versions/linear/`
   - Sliding Window 기반 선형 모델
   - Window size: 30일, Step: 10분
   - 결론: 기상 데이터만으로는 TMS 예측 어려움

5. **ML Pipeline with src/ structure** (2026-01-30 ~ 2026-02-06) ❌ **아카이브됨**
   - 위치: `old_src/ML/`, `old_scripts/ML/train.py`
   - Optuna 최적화, TimeSeriesSplit 교차 검증
   - 피처 선택 (중요도 기반), TMS 모델 그룹화 (modelA, B, C)
   - 도메인 특화 피처 (강수, 기상, TMS 상호작용)

### DL 모델 발전 과정

1. **Initial LSTM** (2026-01-29)
   - 위치: `old_DL_versions/flow_lstm_model.py`, `config.py`
   - 유입유량 모델 초기 실험
   - 문제점: 잦은 오류 발생

2. **LSTM Pipeline with src/ structure** (2026-02-03 ~ 2026-02-06) ❌ **아카이브됨**
   - 위치: `old_src/DL/`, `old_scripts/DL/train_lstm.py`
   - ML과 DL 폴더 구분된 완전 자동화 전처리 파이프라인
   - 도메인 특화 피처 완성 (1,000+ 라인)
   - 모델 사양: modelA(TOC/SS), modelB(TN/TP), modelC(FLUX/PH)
   - 구버전 모델 가중치: `old_model/modelA/B/C_lstm_model.pth`

3. **Latest LSTM** (2026-02-05~) ✅ **현재 사용 중**
   - 위치: `notebook/DL/LSTM_FLOW.ipynb`, `notebook/DL/LSTM_TMS.ipynb`
   - 30분 단위 리샘플링
   - Attention 추가, Walk-Forward Validation
   - 타겟별 개별 모델: flow, toc, ss, tn, tp, flux, ph

## 🗂️ 현재 사용 중인 코드

프로젝트에서 실제로 사용하는 최신 코드는 다음 위치에 있습니다:

### 백엔드 API
- **파일**: `src/main.py` — FastAPI 서버 (예측 엔드포인트)

### 딥러닝 (LSTM)
- **노트북**: `notebook/DL/`
  - `LSTM_FLOW.ipynb` — 유입유량 LSTM 학습
  - `LSTM_TMS.ipynb` — TMS(TOC/SS/TN/TP/FLUX/PH) LSTM 학습
  - `analyze_predictions.py` — 예측 분석
  - `diagnosis.py` — 모델 진단
  - `ensemble_predict.py` — 앙상블 예측
  - `postprocess_correction.py` — 후처리 보정
- **모델/스케일러**: `model/save/{target}_lstm_model.pth`, `X_scaler_{target}.pkl`, `y_scaler_{target}.pkl`
- **문서**: `QUICK_START_DL.md`

### 피처 엔지니어링
- **모듈**: `notebook/feature/`
  - `feature_engineering.py` — 통합 피처 엔지니어링 파이프라인
  - `WF_feature_selection.py` — Walk-Forward 피처 선택
- **선택된 피처 목록**: `data/recommand_features/{target}_recommended_features.csv`

### EDA 및 전처리
- **노트북**: `notebook/EDA/`
  - `flow_tms_periodicity_eda.ipynb` — 유입량/TMS 주기성 분석
- **노트북**: `notebook/preprocess/`
  - `preprocess.ipynb` — 전처리 실험
  - `show.ipynb` — 데이터 시각화
  - `correlation.ipynb` — 상관관계 분석
  - `split_distribution.ipynb` — 분포 분석

### ML (참고용)
- **노트북**: `notebook/ML/primary/baseline.ipynb`

## ⚠️ 주의사항

- 이 폴더의 코드들은 **참고용**으로만 사용하세요
- 실제 프로덕션에서는 `src/` 및 `notebook/DL/`의 최신 코드를 사용하세요
- 아카이브된 코드는 더 이상 유지보수되지 않습니다
- 필요시 이전 버전의 아이디어나 구현 방법을 참고할 수 있습니다

## 📚 관련 문서

- `NOTE.md` - 개발 노트 및 변경 이력
- `TODO.md` - 할 일 목록
- `QUICK_START_DL.md` - DL 파이프라인 가이드
- `README.md` - 프로젝트 개요

---

**아카이브 날짜**: 2026-02-11
