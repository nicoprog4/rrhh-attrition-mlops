import joblib
import os

def load_model(path):
    if os.path.exists(path):
        try:
            m = joblib.load(path)
            return m
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return None
    else:
        print(f"Modelo no encontrado en {path}. Usando fallback.")
        return None