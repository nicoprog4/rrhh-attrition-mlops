"""
Entrena modelo simple (Regresión Logística) y guarda artifacts (.joblib).
También usa mlflow.sklearn.autolog() para tracking.
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///workspace/mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.sklearn.autolog()

DATA_PATH = os.getenv("DATA_PATH", "data/employee_attrition.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        # ejemplo mínimo si no hay CSV
        data = {
            'Edad': [25,34,45,23,50,31,29,40],
            'Salario': [2000,4500,7000,1800,8000,3500,2900,6000],
            'Nivel_Satisfaccion': [3,4,5,2,4,3,2,4],
            'Horas_Extras': [1,0,0,1,0,1,1,0],
            'Renuncio': [1,0,0,1,0,0,1,0]
        }
        return pd.DataFrame(data)
    return pd.read_csv(path)

def preprocess(df):
    X = df[['Edad','Salario','Nivel_Satisfaccion','Horas_Extras']]
    y = df['Renuncio']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, y, scaler

def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="reg_log_rrhh") as run:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        # guardar scaler y modelo
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
        mlflow.log_artifact(MODEL_PATH)
        print(f"Modelo guardado en {MODEL_PATH}")
        return run.info.run_id

if __name__ == "__main__":
    train_and_save()