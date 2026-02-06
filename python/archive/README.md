# Archive - 이전 버전 코드 보관소

이 폴더에는 프로젝트 개발 과정에서 생성되었지만 현재는 사용하지 않는 이전 버전의 코드들이 보관되어 있습니다.

## 📁 폴더 구조

```
archive/
├── old_ML_versions/          # 이전 ML 모델 버전들
│   ├── improved/            # 개선 버전 v1, v2
│   ├── linear/              # 초기 선형 모델 실험
│   └── baseline.py          # 초기 베이스라인 (primary)
├── old_DL_versions/          # 이전 DL 코드
│   ├── config.py            # 구버전 설정 (src/DL/로 통합됨)
│   └── flow_lstm_model.py   # 초기 LSTM 모델 (src/DL/로 통합됨)
└── old_notebooks/            # 테스트/실험용 노트북
    ├── dl_preprocessing_test.ipynb
    ├── preprocess_causal_mask.ipynb
    └── train_standalone.ipynb
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

5. **Final Integrated Version** (2026-01-30 완료) ✅ **현재 사용 중**
   - 위치: `src/ML/`, `scripts/ML/train.py`
   - Optuna 최적화
   - TimeSeriesSplit 교차 검증
   - 피처 선택 (중요도 기반)
   - TMS 모델 그룹화 (modelA, B, C)
   - 데이터 누수 방지
   - 도메인 특화 피처 (강수, 기상, TMS 상호작용)

### DL 모델 발전 과정

1. **Initial LSTM** (2026-01-29)
   - 위치: `old_DL_versions/flow_lstm_model.py`, `config.py`
   - 유입유량 모델 초기 실험
   - 문제점: 잦은 오류 발생

2. **LSTM Pipeline Refactoring** (2026-02-03)
   - 위치: `src/DL/` ✅ **현재 사용 중**
   - ML과 DL 폴더 구분
   - 완전 자동화된 전처리 파이프라인
   - 도메인 특화 피처 완성 (1,000+ 라인)
   - 모델 사양 정의 (flow, modelA, B, C)

3. **Latest LSTM** (2026-02-05)
   - 위치: `notebook/DL/LSTM.ipynb` ✅ **현재 사용 중**
   - 30분 단위 리샘플링
   - Attention 추가
   - Walk-Forward Validation

## 🗂️ 현재 사용 중인 코드

프로젝트에서 실제로 사용하는 최신 코드는 다음 위치에 있습니다:

### 머신러닝
- **모듈**: `src/ML/`
  - `pipeline.py` - 통합 파이프라인
  - `models.py` - 모델 정의 (Optuna 포함)
  - `features.py` - 도메인 특화 피처
  - `preprocess.py` - 전처리
  - 기타 모듈들...
- **실행**: `scripts/ML/train.py`
- **문서**: `QUICK_START.md`

### 딥러닝
- **모듈**: `src/DL/`
  - `pipeline.py` - LSTM 파이프라인
  - `model.py` - LSTM 모델
  - `features.py` - DL용 피처
  - `trainer.py` - 학습 관리
  - 기타 모듈들...
- **실행**: `scripts/DL/train_lstm.py`
- **노트북**: `notebook/DL/LSTM.ipynb`
- **문서**: `QUICK_START_DL.md`

### 피처 엔지니어링
- **노트북**: `notebook/feature/`
  - `modelFLOW.ipynb` - 유입량 모델 피처
  - `modelA.ipynb` - TOC/SS 피처
  - `modelB.ipynb` - TN/TP 피처
  - `modelC.ipynb` - FLUX/PH 피처

### EDA 및 전처리
- **노트북**: `notebook/preprocess/`
  - `preprocess.ipynb` - 전처리 실험
  - `show.ipynb` - 데이터 시각화
  - `correlation.ipynb` - 상관관계 분석

## ⚠️ 주의사항

- 이 폴더의 코드들은 **참고용**으로만 사용하세요
- 실제 프로덕션에서는 `src/` 및 `scripts/`의 최신 코드를 사용하세요
- 아카이브된 코드는 더 이상 유지보수되지 않습니다
- 필요시 이전 버전의 아이디어나 구현 방법을 참고할 수 있습니다

## 📚 관련 문서

- `NOTE.md` - 개발 노트 및 변경 이력
- `TODO.md` - 할 일 목록
- `QUICK_START.md` - ML 파이프라인 가이드
- `QUICK_START_DL.md` - DL 파이프라인 가이드
- `README.md` - 프로젝트 개요

---

**아카이브 날짜**: 2026-02-06
