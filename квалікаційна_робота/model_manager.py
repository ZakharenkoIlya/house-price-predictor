import joblib
import os
from pathlib import Path

MODELS_DIR = "models/"


def save_models(models):
    """Очищає папку MODELS_DIR і зберігає нові моделі"""
    Path(MODELS_DIR).mkdir(exist_ok=True)

    # Видаляємо всі .pkl файли перед збереженням
    for file in Path(MODELS_DIR).glob("*.pkl"):
        file.unlink()

    for name, model in models.items():
        joblib.dump(model, f"{MODELS_DIR}{name}.pkl")

    print(f"✅ Моделі збережено у {MODELS_DIR}")

def load_models(model_names):
    """Завантажує збережені моделі"""
    models = {}
    for name in model_names:
        try:
            models[name] = joblib.load(f"{MODELS_DIR}{name}.pkl")
        except FileNotFoundError:
            print(f"⚠️ Модель {name} не знайдена!")
            return None
    print("✅ Моделі завантажено")
    return models

def models_exist():
    """Перевіряє, чи є збережені моделі"""
    return os.path.exists(MODELS_DIR) and any(Path(MODELS_DIR).glob("*.pkl"))