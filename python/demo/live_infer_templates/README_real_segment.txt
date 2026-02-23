실제 구간 기반 라이브 추론 템플릿
chosen_date=2025-09-28
flow: rows=48, cols=10, file=flow_live_infer_template_real_20250928.csv
toc: rows=48, cols=6, file=toc_live_infer_template_real_20250928.csv
ss: rows=48, cols=8, file=ss_live_infer_template_real_20250928.csv
tn: rows=48, cols=2, file=tn_live_infer_template_real_20250928.csv
tp: rows=48, cols=10, file=tp_live_infer_template_real_20250928.csv
flux: rows=48, cols=17, file=flux_live_infer_template_real_20250928.csv
ph: rows=48, cols=7, file=ph_live_infer_template_real_20250928.csv

생성 방식:
- data/actual 실제 데이터(FLOW/TMS/AWS_368/541/569) 사용
- 30분 리샘플링 + 기존 feature_engineering 파이프라인 적용
- 각 모델의 recommended_features 컬럼만 추출
