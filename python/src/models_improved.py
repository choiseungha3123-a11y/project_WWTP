"""
개선된 모델 모듈
GridSearchCV, XGBoost, Early Stopping 지원
"""

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb


def build_model_zoo_with_gridsearch(cv=3, random_state=42):
    """
    GridSearchCV를 포함한 개선된 모델 정의
    
    Parameters:
    -----------
    cv : int
        TimeSeriesSplit 분할 수
    random_state : int
        랜덤 시드
        
    Returns:
    --------
    dict : 모델 딕셔너리
    """
    
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Ridge
    ridge_params = {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    }
    ridge = GridSearchCV(
        Ridge(random_state=random_state),
        ridge_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Lasso
    lasso_params = {
        'alpha': [0.001, 0.01, 0.1, 1.0]
    }
    lasso = GridSearchCV(
        Lasso(random_state=random_state, max_iter=5000),
        lasso_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # RandomForest
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf = GridSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=-1),
        rf_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # HistGradientBoosting (Early Stopping 지원)
    hgb_params = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [500],
        'max_depth': [5, 10, 20],
        'early_stopping': [True],
        'n_iter_no_change': [20],
        'validation_fraction': [0.2]
    }
    hgb = GridSearchCV(
        HistGradientBoostingRegressor(random_state=random_state),
        hgb_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # XGBoost
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    xgb_model = GridSearchCV(
        xgb.XGBRegressor(random_state=random_state, n_jobs=-1),
        xgb_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    zoo = {
        "Ridge": ridge,
        "Lasso": lasso,
        "RandomForest": rf,
        "HistGBR": hgb,
        "XGBoost": xgb_model,
    }
    
    return zoo
