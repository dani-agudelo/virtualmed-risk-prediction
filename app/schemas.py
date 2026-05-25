"""
schemas.py
==========
Schemas Pydantic para request y response de la API de predicción de riesgo.

Los rangos de validación están alineados con RISK_PREDICTION_API_CONTRACT.md
y con los filtros clínicos de prepare_dataset.py.
"""

from typing import Annotated, Optional
from pydantic import BaseModel, Field, model_validator


# ── Request ───────────────────────────────────────────────────────────────────

class CardiovascularRiskRequest(BaseModel):
    # Obligatorias — la API retorna 422 si faltan o están fuera de rango
    age:                    Annotated[int,   Field(ge=30,   le=65,   description="Edad en años")]
    sex:                    Annotated[int,   Field(ge=0,    le=1,    description="0=mujer, 1=hombre")]
    bmi:                    Annotated[float, Field(ge=10.0, le=70.0, description="Índice de masa corporal (kg/m²)")]
    systolic_bp:            Annotated[int,   Field(ge=70,   le=250,  description="Presión arterial sistólica (mmHg)")]
    diastolic_bp:           Annotated[int,   Field(ge=40,   le=170,  description="Presión arterial diastólica (mmHg)")]
    smoker:                 Annotated[int,   Field(ge=0,    le=1,    description="0=no fumador, 1=fumador")]
    physical_activity_level:Annotated[int,   Field(ge=0,    le=1,    description="0=sedentario, 1=activo")]

    # Opcionales — null se imputa con mediana del train en el predictor
    family_history_cvd: Annotated[Optional[int], Field(default=None, ge=0, le=1,
                                  description="Antecedentes familiares cardiovasculares. 0=no, 1=sí, null=desconocido")]
    cholesterol_total:  Annotated[Optional[int], Field(default=None, ge=1, le=3,
                                  description="Colesterol: 1=normal, 2=alto, 3=muy alto, null=desconocido")]
    glucose_mg_dl:      Annotated[Optional[int], Field(default=None, ge=1, le=3,
                                  description="Glucosa: 1=normal, 2=alta, 3=muy alta, null=desconocido")]

    @model_validator(mode="after")
    def systolic_above_diastolic(self):
        if self.systolic_bp <= self.diastolic_bp:
            raise ValueError(
                f"systolic_bp ({self.systolic_bp}) debe ser mayor que diastolic_bp ({self.diastolic_bp})"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 52,
                "sex": 1,
                "bmi": 27.4,
                "systolic_bp": 135,
                "diastolic_bp": 88,
                "smoker": 0,
                "physical_activity_level": 1,
                "family_history_cvd": 1,
                "cholesterol_total": 2,
                "glucose_mg_dl": None,
            }
        }
    }


# ── Response ──────────────────────────────────────────────────────────────────

class RiskFactor(BaseModel):
    name:         str
    value:        float | int | str | None
    contribution: Optional[float] = Field(
        default=None,
        description="Contribución relativa de esta feature al score. Implementación futura (v2)."
    )


class RiskPredictionResponse(BaseModel):
    score:               int   = Field(ge=0, le=100, description="Riesgo calculado en escala 0-100")
    risk_level:          str   = Field(description="low | medium | high")
    model_version:       str
    disclaimer_version:  str
    factors:             list[RiskFactor] = Field(
        default=[],
        description="Factores contribuyentes. Vacío en v1 — implementación futura."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "score": 73,
                "risk_level": "high",
                "model_version": "v1",
                "disclaimer_version": "PLACEHOLDER",
                "factors": [],
            }
        }
    }


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    model_version: str
