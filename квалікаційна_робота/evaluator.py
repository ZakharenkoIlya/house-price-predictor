import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error


def evaluate_model(models, X_test, y_test):
    """ Оцінка моделей за допомогою MAE та R². """

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        medae = median_absolute_error(y_test, y_pred)
        results[name] = {"MAE": mae, "R²": r2, "RMSE": rmse, "MAPE": mape, "MedAE": medae}
        print(f"📊 {name}: MAE = {mae:.4f}, R² = {r2:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.4f}, MedAE = {medae:.4f}")

    return results