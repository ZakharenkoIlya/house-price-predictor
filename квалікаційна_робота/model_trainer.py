import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def train_models(X_train, y_train):
    """Навчання базових моделей та метамоделі (стекінгу) з покращеними параметрами"""

    # Словник базових моделей
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=500, depth=8, learning_rate=0.05, loss_function='RMSE', verbose=0),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)
    }

    trained_models = {}

    # Перевірка правильного формату вхідних даних
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train має бути pandas DataFrame з іменами колонок")

    # Навчання базових моделей
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✅ {name} навчена")

    # Побудова стекінг-регресора
    estimators = [(name, model) for name, model in trained_models.items()]
    meta_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        passthrough=True
    )
    meta_model.fit(X_train, y_train)
    trained_models["MetaModel"] = meta_model
    print("✅ MetaModel навчена")

    return trained_models
