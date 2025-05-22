import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import os
from pathlib import Path

MODELS_DIR = "models/"

class HousePriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Predictor")

        self.models = self.load_models()
        self.df = pd.read_csv("kc_house_data.csv").drop(columns=["id", "date", "price"])
        self.feature_sliders = {}
        self.input_vars = {}
        self.value_labels = {}
        self.prediction_labels = {}  # Для виводу кожного прогнозу

        self.build_ui()

    def load_models(self):
        """Завантаження всіх моделей з папки"""
        models = {}
        if not os.path.exists(MODELS_DIR):
            print("❌ Папка models/ не знайдена.")
            return models

        for file in os.listdir(MODELS_DIR):
            if file.endswith(".pkl"):
                model_name = file.replace(".pkl", "")
                try:
                    models[model_name] = joblib.load(os.path.join(MODELS_DIR, file))
                except Exception as e:
                    print(f"⚠️ Помилка завантаження {file}: {e}")
        print(f"✅ Завантажено моделей: {len(models)}")
        return models

    def build_ui(self):
        ttk.Label(self.root, text="Введіть параметри будинку:", font=("Arial", 14)).pack(pady=10)

        frame = ttk.Frame(self.root)
        frame.pack()

        features = list(self.df.columns)
        for i, feature in enumerate(features):
            min_val = float(self.df[feature].min())
            max_val = float(self.df[feature].max())
            avg_val = round(float(self.df[feature].mean()))

            var = tk.DoubleVar(value=avg_val)
            self.input_vars[feature] = var

            subframe = ttk.Frame(frame)
            subframe.grid(row=i // 2, column=i % 2, padx=10, pady=5, sticky="w")

            ttk.Label(subframe, text=feature).pack(anchor="w")

            slider_frame = ttk.Frame(subframe)
            slider_frame.pack()

            slider = ttk.Scale(slider_frame, from_=min_val, to=max_val, variable=var,
                               orient="horizontal", length=200,
                               command=lambda v, f=feature: self.update_slider_value(f))
            slider.pack(side="left")

            value_label = ttk.Label(slider_frame, text=f"{avg_val:.2f}", width=10)
            value_label.pack(side="left", padx=5)
            self.value_labels[feature] = value_label

        # Фрейм для результатів усіх моделей
        self.results_frame = ttk.Frame(self.root)
        self.results_frame.pack(pady=15)

        self.update_prediction()

    def update_slider_value(self, feature):
        value = round(self.input_vars[feature].get())
        self.input_vars[feature].set(value)
        self.value_labels[feature].config(text=f"{value:.2f}")
        self.update_prediction()

    def update_prediction(self):
        input_data = {k: var.get() for k, var in self.input_vars.items()}
        input_df = pd.DataFrame([input_data])

        # Очистити попередні прогнози
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        meta_model_name = "MetaModel"
        if meta_model_name in self.models:
            model = self.models[meta_model_name]
            try:
                prediction = model.predict(input_df)[0]
                label = ttk.Label(self.results_frame, text=f"{meta_model_name}: ${prediction:,.2f}",
                                  font=("Arial", 14, "bold"))
                label.pack(anchor="w", pady=5)
                self.prediction_labels[meta_model_name] = label
            except Exception as e:
                print(f"⚠️ Помилка прогнозу для {meta_model_name}: {e}")
        else:
            label = ttk.Label(self.results_frame, text="❌ Метамодель не знайдена.", font=("Arial", 12, "italic"))
            label.pack(anchor="w", pady=5)

        # Вивести прогноз для кожної моделі
        '''
        for name, model in self.models.items():
            try:
                prediction = model.predict(input_df)[0]
                label = ttk.Label(self.results_frame, text=f"{name}: ${prediction:,.2f}", font=("Arial", 12))
                label.pack(anchor="w", pady=2)
                self.prediction_labels[name] = label
            except Exception as e:
                print(f"⚠️ Помилка прогнозу для {name}: {e}")
'''

if __name__ == "__main__":
    root = tk.Tk()
    app = HousePriceApp(root)
    root.mainloop()
