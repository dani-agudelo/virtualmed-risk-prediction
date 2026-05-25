"""
main.py
=======
Entrypoint de la API FastAPI de predicción de riesgo cardiovascular — VirtualMed.

Variables de entorno requeridas:
    MODEL_DIR      Ruta base donde están los modelos (ej. ./models)
    MODEL_VERSION  Versión a cargar (ej. v1)

Uso en desarrollo:
    uvicorn main:app --reload --port 8000

Uso en producción:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.predictor import load_model, is_loaded
from app.routers.predict import router as predict_router
from app.schemas import HealthResponse


# ── Configuración desde entorno ───────────────────────────────────────────────

MODEL_DIR     = os.getenv("MODEL_DIR",     "./models")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")


# ── Lifespan: carga el modelo al arrancar ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al arrancar. Falla rápido si no lo encuentra."""
    try:
        load_model(MODEL_DIR, MODEL_VERSION)
        print(f"[startup]  API lista — modelo cardiovascular {MODEL_VERSION} cargado.")
    except RuntimeError as e:
        print(f"[startup]  ERROR: {e}")
        print("[startup]  La API arrancó pero /predict/cardiovascular retornará 503.")
    yield
    # Cleanup si fuera necesario (liberar recursos, cerrar conexiones, etc.)
    print("[shutdown] API detenida.")


# ── Aplicación ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VirtualMed — Risk Prediction API",
    description=(
        "Microservicio de predicción de riesgo cardiovascular para VirtualMed.\n\n"
        "Los scores son **orientativos** y no constituyen diagnóstico médico. "
        "Ver `disclaimer_version` en cada respuesta.\n\n"
        "Consumido por el backend .NET a través de `IRiskPredictionClient`."
    ),
    version="1.0.0",
    contact={"name": "Equipo VirtualMed"},
    lifespan=lifespan,
)

app.include_router(predict_router)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Sistema"],
    summary="Estado del servicio",
)
def health():
    if not is_loaded():
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_version": MODEL_VERSION},
        )
    return HealthResponse(status="ok", model_version=MODEL_VERSION)


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {"message": "VirtualMed Risk Prediction API", "docs": "/docs", "health": "/health"}
