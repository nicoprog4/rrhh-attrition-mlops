from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Empleado(BaseModel):
    edad:int
    salario:float
    horas_extras:int

@app.get("/")
def home():
    return {"mensaje": "API de predicción de renuncia funcionando"}

@app.post("/api/empleados/riesgo-renuncia")
def predecir(empleado:Empleado):

    if empleado.horas_extras == 1:
        resultado = "Alto riesgo de renuncia"
    else:
        resultado = "Bajo riesgo"

    return {
        "prediccion": resultado
    }