# 이상치 처리 개선 전략

## 현재 문제
- FLUX: 극단적으로 왜곡된 분포 (최댓값이 중앙값의 1,622배)
- TP: 왜곡된 분포 (99%가 0.82인데 평균은 0.11)
- TOC: 비교적 정상 분포지만 높은 값이 간헐적으로 발생

## 추천 해결 방법 (우선순위 순)

### 1순위: 로그 변환 + Huber Loss (가장 효과적)
```python
# 타겟별 맞춤 처리
MODE_PREPROCESSING = {
    "flux": {"use_log": True, "loss": "huber"},
    "tp": {"use_log": True, "loss": "huber"},
    "toc": {"use_log": False, "loss": "huber"},
}

# 전처리
if MODE_PREPROCESSING[MODE]["use_log"]:
    y_transformed = np.log1p(y)  # log(1+y)

# 손실 함수
if MODE_PREPROCESSING[MODE]["loss"] == "huber":
    criterion = HuberLoss(delta=1.0)
else:
    criterion = nn.MSELoss()
```

### 2순위: Robust Scaler
```python
from sklearn.preprocessing import RobustScaler

# StandardScaler 대신 사용
y_scaler = RobustScaler(quantile_range=(10, 90))
```

### 3순위: IQR threshold 완화
```python
OutlierConfig(
    method="iqr",
    iqr_threshold=3.0,  # 1.5 → 3.0 (더 보수적)
    require_both=True   # 유지!
)
```

### 4순위: 도메인 범위 재검토
```python
domain_rules = {
    "TOC_VU": (0, 250),
    "TP_VU": (0, 20),  # 99.9%가 17.81이므로 적절
    "FLUX_VU": (0, 50000),  # 99%가 8,164이므로 상향 조정
}
```

## 실험 순서
1. Huber Loss로 변경 (가장 쉬움) → 테스트
2. FLUX/TP에 로그 변환 적용 → 테스트
3. Robust Scaler 적용 → 테스트
4. IQR threshold 완화 → 테스트

## require_both=False는 사용하지 마세요!
- TOC: 12,147개 정상 데이터 오판 (2.85%)
- TP: 26,715개 정상 데이터 오판 (6.26%)
- FLUX: 영향 없음
- 결론: 과도한 필터링으로 모델 성능 저하 가능성 높음
