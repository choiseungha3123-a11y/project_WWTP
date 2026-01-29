"""
모델 정의 및 래퍼 모듈
다양한 회귀 모델 정의 및 멀티아웃풋 래퍼
"""

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def build_model_zoo(random_state=42):
    """사용 가능한 모델들의 딕셔너리 생성"""
    zoo = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "HistGBR": HistGradientBoostingRegressor(
            random_state=random_state, learning_rate=0.05, max_iter=500
        ),
    }
    return zoo


def wrap_multioutput_if_needed(model, y):
    """필요시 멀티아웃풋 래퍼 적용"""
    if y.shape[1] <= 1:
        return model

    # HistGBR은 래퍼가 필요함
    if isinstance(model, HistGradientBoostingRegressor):
        return MultiOutputRegressor(model)
    return model
