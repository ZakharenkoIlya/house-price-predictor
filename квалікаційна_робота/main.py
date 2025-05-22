import pandas as pd
from sklearn.model_selection import train_test_split
from model_trainer import train_models
from evaluator import evaluate_model
from model_manager import save_models, load_models, models_exist

'''
def get_important_features(model, X, threshold=0.1):
    """Повертає список важливих ознак, важливість яких вище порогу"""
    importances = model.feature_importances_
    important_features = [f for f, imp in zip(X.columns, importances) if imp >= threshold]
    ignored_features = [f for f, imp in zip(X.columns, importances) if imp < threshold]

    print("⚠️ Ігноруються ознаки (шум):", ignored_features)
    return important_features
'''

'''
    # Фільтрація ознак за важливістю (на базі GradientBoosting)
    print("📊 Аналіз важливості ознак...")
    selector_model = CatBoostRegressor()
    selector_model.fit(X_train, y_train)
    important_features = get_important_features(selector_model, X_train)

    # Обрізаємо набори за важливими ознаками
    X_train = X_train[important_features]
    X_test = X_test[important_features]
'''

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=["id", "date"])  # Видаляємо непотрібне

    X = df.drop(columns=["price"])
    y = df["price"]
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns.tolist()


def main():
    # Завантаження даних
    (X_train, X_test, y_train, y_test), feature_names = load_data("kc_house_data.csv")

    # Вибір дії користувача
    if models_exist():
        choice = input("🔧 Бажаєш навчити нові моделі? (y/n): ").strip().lower()
    else:
        print("⚠️ Моделі не знайдені. Потрібно навчити нові.")
        choice = 'y'

    if choice == 'y':
        models = train_models(X_train, y_train)
        save_models(models)
    else:
        model_names = ["RandomForest", "GradientBoosting", "XGBoost", "CatBoost", "HistGradientBoosting", "MetaModel"]
        models = load_models(model_names)
        if models is None:
            print("❌ Завантаження не вдалося. Завершення програми.")
            return

    # Оцінка
    evaluate_model(models, X_test, y_test)


if __name__ == "__main__":
    main()
