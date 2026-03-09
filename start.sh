#!/usr/bin/env bash
# start.sh: comando que Render usará para iniciar el servicio
# Asegúrate de marcarlo ejecutable antes de hacer git commit (ver comandos abajo)

uvicorn src.inference_service.main:app --host 0.0.0.0 --port $PORT