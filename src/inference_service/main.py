from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from src.inference_service.model_utils import load_model

app = FastAPI(title="API ERP - Módulo Recursos Humanos")

# carga modelo (joblib) - si no existe, la app sigue funcionando con stub
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
model = load_model(MODEL_PATH)

class Empleado(BaseModel):
    edad: int
    salario: float
    nivel_satisfaccion: int
    horas_extras: int

@app.get("/")
def root():
    return {"mensaje": "API de predicción de renuncia funcionando"}

@app.post("/api/empleados/riesgo-renuncia")
def predecir(empleado: Empleado):
    """
    Formato simple: transformamos a vector y pedimos predicción.
    Ajusta preprocesado según tu dataset real.
    """
    try:
        if model is None:
            # fallback simple si no hay modelo
            riesgo = "Alto riesgo" if empleado.horas_extras == 1 else "Bajo riesgo"
            return {"prediccion": riesgo, "info": "modelo no cargado (modo fallback)"}

        # ejemplo de entrada: [Edad, Salario, Nivel_Satisfaccion, Horas_Extras]
        x = [[empleado.edad, empleado.salario, empleado.nivel_satisfaccion, empleado.horas_extras]]
        pred = model.predict(x)
        riesgo = "Alto riesgo" if int(pred[0]) == 1 else "Bajo riesgo"
        return {"prediccion": riesgo}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))