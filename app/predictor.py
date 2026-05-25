"""
predictor.py
============
Carga el modelo entrenado y aplica la lógica de inferencia:
  1. Imputación de opcionales con medianas del train
  2. Construcción del vector de features en el orden correcto
  3. Predicción de probabilidad
  4. Mapeo probabilidad → score (0-100) y risk_level

El modelo se carga una sola vez al arrancar la API (startup event en main.py).
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.schemas import CardiovascularRiskRequest, RiskPredictionResponse


# ── Estado global de la aplicación ───────────────────────────────────────────
# Estos objetos se populan en startup y se reutilizan en cada request.

_model = None
_feature_list: list[str] = []
_imputation_values: dict = {}
_metadata: dict = {}


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_model(model_dir: str, version: str) -> None:
    """
    Carga modelo, feature_list.json, imputation_values.json y metadata.json
    desde model_dir/cvd_risk/{version}/.

    Llamada una sola vez en el evento startup de FastAPI.
    Lanza RuntimeError si algún archivo requerido no existe.
    """
    global _model, _feature_list, _imputation_values, _metadata

    base = Path(model_dir) / "cvd_risk" / version

    required_files = {
        "model":       base / "model.joblib",
        "features":    base / "feature_list.json",
        "imputation":  base / "imputation_values.json",
        "metadata":    base / "metadata.json",
    }

    for name, path in required_files.items():
        if not path.exists():
            raise RuntimeError(f"Archivo requerido no encontrado: {path}")

    _model            = joblib.load(required_files["model"])
    _feature_list     = json.loads(required_files["features"].read_text())["features"]
    _imputation_values= json.loads(required_files["imputation"].read_text())
    _metadata         = json.loads(required_files["metadata"].read_text())

    print(f"[predictor]  Modelo cargado: cardiovascular_risk/{version}")
    print(f"[predictor]  Features: {_feature_list}")
    print(f"[predictor]  Imputación: {_imputation_values}")


def is_loaded() -> bool:
    return _model is not None


# ── Inferencia ────────────────────────────────────────────────────────────────

def _build_feature_vector(request: CardiovascularRiskRequest) -> pd.DataFrame:
    """
    Convierte el request Pydantic en un DataFrame de una fila con las features
    en el orden exacto definido en feature_list.json.

    Los campos opcionales que lleguen como None se imputan con la mediana
    del conjunto de entrenamiento.
    """
    raw = {
        "age":                    request.age,
        "sex":                    request.sex,
        "bmi":                    request.bmi,
        "systolic_bp":            request.systolic_bp,
        "diastolic_bp":           request.diastolic_bp,
        "smoker":                 request.smoker,
        "physical_activity_level":request.physical_activity_level,
        "family_history_cvd":     request.family_history_cvd,
        "cholesterol_total":      request.cholesterol_total,
        "glucose_mg_dl":          request.glucose_mg_dl,
    }

    # Imputar opcionales con mediana del train si vienen como None
    for col, median_val in _imputation_values.items():
        if raw.get(col) is None and median_val is not None:
            raw[col] = median_val

    # Construir DataFrame respetando el orden del contrato
    df = pd.DataFrame([raw])[_feature_list]
    return df


def _probability_to_score(prob: float) -> int:
    """Escala probabilidad [0.0-1.0] a score entero [0-100]."""
    return int(round(prob * 100))


def _score_to_risk_level(score: int, thresholds: dict) -> str:
    """
    Mapea score a risk_level usando los umbrales definidos en metadata.json.
    Por defecto: <30=low, 30-60=medium, >60=high.
    """
    low_max = thresholds.get("low", {}).get("max_score", 30)
    high_min = thresholds.get("high", {}).get("min_score", 61)

    if score < low_max:
        return "low"
    elif score >= high_min:
        return "high"
    else:
        return "medium"


def predict_cardiovascular(request: CardiovascularRiskRequest) -> RiskPredictionResponse:
    """
    Punto de entrada principal para inferencia cardiovascular.

    Flujo:
      1. Construir vector de features con imputación
      2. Predecir probabilidad con el modelo calibrado
      3. Convertir a score y risk_level
      4. Retornar respuesta tipada
    """
    if not is_loaded():
        raise RuntimeError("Modelo no disponible. Verifica MODEL_DIR y MODEL_VERSION.")

    feature_vector = _build_feature_vector(request)

    # predict_proba retorna [[prob_0, prob_1]] — tomamos prob de clase positiva
    prob = float(_model.predict_proba(feature_vector)[0][1])
    score = _probability_to_score(prob)

    thresholds = _metadata.get("thresholds", {})
    risk_level = _score_to_risk_level(score, thresholds)
    model_version = _metadata.get("version", "unknown")

    return RiskPredictionResponse(
        score=score,
        risk_level=risk_level,
        model_version=model_version,
        disclaimer_version="PLACEHOLDER",
        factors=[],   # v2: SHAP o permutation importance
    )
