"""
routers/predict.py
==================
Endpoints de predicción de riesgo.
 
  POST /predict/cardiovascular  — modelo activo
  POST /predict/diabetes        — no implementado (modelo descartado en v1)
"""
 
from fastapi import APIRouter, HTTPException, status
 
from app.predictor import predict_cardiovascular
from app.schemas import CardiovascularRiskRequest, RiskPredictionResponse
 
router = APIRouter(prefix="/predict", tags=["Predicción de riesgo"])
 
 
@router.post(
    "/cardiovascular",
    response_model=RiskPredictionResponse,
    summary="Calcular riesgo cardiovascular",
    description=(
        "Recibe un vector de features del paciente y devuelve un score orientativo "
        "de riesgo cardiovascular en escala 0-100, junto con el nivel de riesgo "
        "(low / medium / high).\n\n"
        "**Disclaimer:** el resultado es orientativo y no constituye diagnóstico médico."
    ),
)
def predict_cardiovascular_endpoint(request: CardiovascularRiskRequest) -> RiskPredictionResponse:
    try:
        return predict_cardiovascular(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno durante la predicción: {str(e)}",
        )
