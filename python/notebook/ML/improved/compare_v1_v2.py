"""
V1 vs V2 결과 비교 스크립트

V1과 V2 실행 후 이 스크립트를 실행하여 개선 효과를 확인하세요.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_results():
    """V1과 V2 결과 비교"""
    
    print("="*80)
    print("V1 vs V2 결과 비교")
    print("="*80)
    
    # 결과 디렉토리 확인
    v1_dir = "../../results/ML/v1"
    v2_dir = "../../results/ML/v2"
    
    v1_exists = os.path.exists(v1_dir)
    v2_exists = os.path.exists(v2_dir)
    
    print(f"\nV1 결과 존재: {'✓' if v1_exists else '✗'} ({v1_dir})")
    print(f"V2 결과 존재: {'✓' if v2_exists else '✗'} ({v2_dir})")
    
    if not v1_exists:
        print("\n⚠️  V1 결과가 없습니다. improved_baseline.py를 먼저 실행하세요.")
        return
    
    if not v2_exists:
        print("\n⚠️  V2 결과가 없습니다. improved_v2_baseline.py를 먼저 실행하세요.")
        return
    
    # 파일 목록 비교
    print("\n" + "="*80)
    print("생성된 파일 비교")
    print("="*80)
    
    v1_files = set(os.listdir(v1_dir)) if v1_exists else set()
    v2_files = set(os.listdir(v2_dir)) if v2_exists else set()
    
    print(f"\nV1 파일 수: {len(v1_files)}")
    print(f"V2 파일 수: {len(v2_files)}")
    
    # 이미지 파일만 필터링
    v1_images = {f for f in v1_files if f.endswith('.png')}
    v2_images = {f for f in v2_files if f.endswith('.png')}
    
    print(f"\nV1 이미지: {len(v1_images)}개")
    for img in sorted(v1_images):
        print(f"  - {img}")
    
    print(f"\nV2 이미지: {len(v2_images)}개")
    for img in sorted(v2_images):
        print(f"  - {img}")
    
    # 개선 효과 요약
    print("\n" + "="*80)
    print("주요 개선사항 체크리스트")
    print("="*80)
    
    print("\n✓ 구현 완료:")
    print("  1. 결측치 보간 (Hybrid: Linear + KNN + Forward Fill)")
    print("  2. 도메인 피처 추가 (5개 카테고리)")
    print("  3. 정규화 강화 (Ridge alpha 10배, RF max_depth 제한)")
    print("  4. 피처 수 감소 (50 → 30)")
    print("  5. Lag/Rolling 윈도우 축소")
    
    print("\n⏳ 실행 후 확인 필요:")
    print("  - 데이터 사용률 증가 (4.2% → 70%+)")
    print("  - FLOW R² 개선 (0.57 → 0.75+)")
    print("  - TMS R² 개선 (-1.82 → 0.50+)")
    print("  - 과적합 감소 (Train-Test 격차 축소)")
    
    print("\n" + "="*80)
    print("다음 단계")
    print("="*80)
    print("\n1. V1과 V2의 R² comparison 그래프를 비교하세요")
    print("2. Learning curve를 비교하여 과적합 감소를 확인하세요")
    print("3. 콘솔 출력에서 데이터 사용률을 비교하세요")
    print("4. 필요시 하이퍼파라미터를 추가 튜닝하세요")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    compare_results()
