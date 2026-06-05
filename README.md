# VirtualMed — Risk Prediction Service

Microservicio de predicción de riesgo cardiovascular para la plataforma VirtualMed. Expone una API HTTP que recibe datos clínicos básicos de un paciente y devuelve un score de riesgo orientativo en escala 0–100.

> **Aviso:** el score es orientativo y no constituye diagnóstico médico.

---

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo.
- El modelo entrenado debe existir en `models/cardiovascular_risk/v1/` antes de construir la imagen. Ver [Entrenamiento](#entrenamiento) si necesitas generarlo.

---

## Inicio rápido

```bash
# 1. Construir la imagen
docker build -t virtualmed-risk-prediction .

# 2. Levantar el contenedor
docker run -d --name risk-api -p 8000:8000 virtualmed-risk-prediction

# 3. Verificar que funciona
curl http://localhost:8000/health
```

La API queda disponible en `http://localhost:8000`.  
Documentación interactiva (Swagger): `http://localhost:8000/docs`

---

## Uso del endpoint principal

```bash
curl -X POST http://localhost:8000/predict/cardiovascular \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "bmi": 27.4,
    "systolic_bp": 135,
    "diastolic_bp": 88,
    "smoker": 0,
    "physical_activity_level": 1,
    "family_history_cvd": 1,
    "cholesterol_total": 2,
    "glucose_mg_dl": null
  }'
```

Respuesta:

```json
{
  "score": 73,
  "risk_level": "high",
  "model_version": "v1",
  "disclaimer_version": "PLACEHOLDER",
  "factors": []
}
```

---

## Entrenamiento

Solo necesario si el modelo no existe o se quiere reentrenar.

**Requisitos adicionales:** Python 3.11+, dependencias en `requirements.txt`, y el dataset original descargado desde [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).

```bash
pip install -r requirements.txt

# Paso 1 — Preparar dataset
python prepare_dataset.py \
  --input cardio_train.csv \
  --output data/processed/v1/ \
  --models models/cardiovascular_risk/v1/

# Paso 2 — Entrenar modelo
python train_cardiovascular_risk.py \
  --data data/processed/v1/ \
  --models models/cardiovascular_risk/v1/
```

Luego volver al [Inicio rápido](#inicio-rápido) para construir la imagen Docker.

---

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `MODEL_DIR` | `./models` | Ruta base de los modelos |
| `MODEL_VERSION` | `v1` | Versión del modelo a cargar |
| `PORT` | `8000` | Puerto de escucha |

---

## Documentación

| Documento | Descripción |
|---|---|
| `docs/ml/MANUAL_SERVICIO.md` | Manual completo del servicio |
| `docs/ml/RISK_PREDICTION_API_CONTRACT.md` | Contrato de integración con el backend .NET |
| `docs/ml/DATASET_PROVENANCE.md` | Origen del dataset, licencia y cumplimiento Ley 1581 |
