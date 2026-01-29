"""
데이터 스케일링 모듈
StandardScaler 적용
"""

from sklearn.preprocessing import StandardScaler


def create_scaler():
    """StandardScaler 생성"""
    return StandardScaler()


def scale_data(X_train, X_valid, X_test, scaler=None):
    """
    학습/검증/테스트 데이터 스케일링
    
    Parameters:
    -----------
    X_train : DataFrame or array
        학습 데이터
    X_valid : DataFrame or array
        검증 데이터
    X_test : DataFrame or array
        테스트 데이터
    scaler : StandardScaler, optional
        기존 스케일러 (None이면 새로 생성)
        
    Returns:
    --------
    tuple : (X_train_scaled, X_valid_scaled, X_test_scaled, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler
