# Risk Prediction API — Contrato compartido

**Responsable:** Equipo VirtualMed  
**Última actualización:** 2026-05-20  
**Versión del contrato:** 1.0  
**Alineado con:** `IRiskPredictionClient` (.NET), `AI-U04`, `BE-U07`

---

## 1. Información general

| Campo | Detalle |
|---|---|
| Base URL (dev) | `http://localhost:8000` |
| Base URL (prod) | Configurado en `RiskPrediction:BaseUrl` (appsettings) |
| Protocolo | HTTP/1.1, JSON |
| Autenticación | Sin autenticación externa — servicio interno de red privada |
| Timeout recomendado (.NET) | 10 segundos |
| OpenAPI interactivo | `GET {BaseUrl}/docs` |
| Schema JSON | `GET {BaseUrl}/openapi.json` |

---

## 2. Endpoints

### `POST /predict/cardiovascular`

Calcula el score de riesgo cardiovascular para un paciente dado un vector de features.

#### Request

**Content-Type:** `application/json`

```json
{
  "age": 52,
  "sex": 1,
  "bmi": 27.4,
  "systolic_bp": 135,
  "diastolic_bp": 88,
  "smoker": 0,
  "physical_activity_level": 1,
  "family_history_cvd": 1,
  "family_history_diabetes": 0,
  "weight_kg": 82.0,
  "height_m": 1.73,
  "cholesterol_total": 2,
  "glucose_mg_dl": null
}
```

##### Descripción de campos

| Campo | Tipo | Obligatorio | Rango / valores válidos | Notas |
|---|---|---|---|---|
| `age` | int | ✅ | [30, 65] | Años completos |
| `sex` | int | ✅ | 0 = mujer, 1 = hombre | |
| `bmi` | float | ✅ | [10.0, 70.0] | kg/m² |
| `systolic_bp` | int | ✅ | [70, 250] | mmHg |
| `diastolic_bp` | int | ✅ | [40, 170] | mmHg |
| `smoker` | int | ✅ | 0 = no, 1 = sí | |
| `physical_activity_level` | int | ✅ | 0 = sedentario, 1 = activo | |
| `family_history_cvd` | int \| null | ⬜ | 0 = no, 1 = sí, null = desconocido | Se imputa con mediana del train si null |
| `family_history_diabetes` | int \| null | ⬜ | 0 = no, 1 = sí, null = desconocido | Se imputa con mediana del train si null |
| `weight_kg` | float \| null | ⬜ | [30.0, 200.0] | Ignorado si `bmi` ya está presente |
| `height_m` | float \| null | ⬜ | [1.30, 2.20] | Ignorado si `bmi` ya está presente |
| `cholesterol_total` | int \| null | ⬜ | 1 = normal, 2 = alto, 3 = muy alto, null = desconocido | Valor ordinal, **no mg/dL** |
| `glucose_mg_dl` | int \| null | ⬜ | 1 = normal, 2 = alto, 3 = muy alto, null = desconocido | Valor ordinal, **no mg/dL** |

> **Nota sobre `cholesterol_total` y `glucose_mg_dl`:** los valores enteros (1/2/3) son categorías ordinales heredadas del dataset de entrenamiento, no unidades de laboratorio. El .NET debe mapear los valores disponibles en el expediente clínico a esta escala antes de enviarlos, o enviar `null` si no hay información suficiente para mapear.

#### Response — `200 OK`

```json
{
  "score": 73,
  "risk_level": "high",
  "model_version": "v1",
  "disclaimer_version": "<!-- PLACEHOLDER: definir versión del disclaimer legal -->",
  "factors": []
}
```

##### Descripción de campos de respuesta

| Campo | Tipo | Descripción |
|---|---|---|
| `score` | int | Riesgo calculado, escala 0–100 |
| `risk_level` | string | `"low"` / `"medium"` / `"high"` — ver umbrales §3 |
| `model_version` | string | Versión del modelo cargado, ej. `"v1"` |
| `disclaimer_version` | string | Versión del texto legal mostrado al usuario |
| `factors` | array | Lista de factores contribuyentes — **vacío en v1**, implementación futura (ver §5) |

#### Errores

| Código | Causa | Body |
|---|---|---|
| `422 Unprocessable Entity` | Campo obligatorio ausente o valor fuera de rango | Body estándar de FastAPI con detalle por campo |
| `503 Service Unavailable` | Modelo no cargado o error interno del microservicio | `{ "detail": "Model not available" }` |

---

## 3. Umbrales de `risk_level`

Los umbrales se aplican en la capa FastAPI sobre la probabilidad bruta del modelo (0.0–1.0), que luego se escala a 0–100 para el campo `score`.

| `risk_level` | Rango de probabilidad | Rango de `score` |
|---|---|---|
| `"low"` | < 0.30 | 0–29 |
| `"medium"` | 0.30 – 0.60 | 30–60 |
| `"high"` | > 0.60 | 61–100 |

> Los umbrales están definidos en `models/cvd_risk/v{n}/metadata.json` bajo la clave `thresholds`. Pueden ajustarse sin reentrenar el modelo actualizando ese archivo y reiniciando la API.

---

## 4. Ejemplo de integración — `IRiskPredictionClient` (.NET)

El backend .NET construye el payload en `CalculateRiskScoreCommand` y llama a este endpoint. A continuación el contrato esperado en C#:

```csharp
// Request DTO — alineado con el schema Pydantic de FastAPI
public record CardiovascularRiskRequest(
    int Age,
    int Sex,
    double Bmi,
    int SystolicBp,
    int DiastolicBp,
    int Smoker,
    int PhysicalActivityLevel,
    int? FamilyHistoryCvd,
    int? FamilyHistoryDiabetes,
    double? WeightKg,
    double? HeightM,
    int? CholesterolTotal,
    int? GlucoseMgDl
);

// Response DTO — alineado con la respuesta de FastAPI
public record RiskPredictionResponse(
    int Score,
    string RiskLevel,
    string ModelVersion,
    string DisclaimerVersion,
    List<RiskFactor> Factors
);

public record RiskFactor(
    string Name,
    object Value,
    double? Contribution  // null en v1
);
```

**Nombres de campos:** el .NET debe serializar en `snake_case` al llamar a FastAPI. Configurar `JsonSerializerOptions` con `JsonNamingPolicy.SnakeCaseLower` o usar atributos `[JsonPropertyName]` explícitos.

**Timeout:** configurar `HttpClient` con timeout de 10 segundos. Ante `TaskCanceledException` por timeout, retornar error amigable al usuario sin propagar la excepción técnica.

---

## 5. Campo `factors` — implementación futura

En v1 el campo `factors` se devuelve como lista vacía `[]`. La implementación futura incluirá las contribuciones individuales de cada feature al score, calculadas mediante **permutation importance** o **SHAP values** sobre XGBoost/Random Forest.

Cuando se implemente, cada elemento de `factors` tendrá la siguiente estructura:

```json
{
  "name": "systolic_bp",
  "value": 135,
  "contribution": 0.18
}
```

El campo `contribution` representa la contribución relativa de esa feature al score final (valores positivos aumentan el riesgo, negativos lo reducen). El .NET y el frontend deben estar preparados para recibir esta lista poblada sin cambios de contrato, ya que el campo existe desde v1.

---

## 6. Variables de entorno — FastAPI

| Variable | Descripción | Ejemplo |
|---|---|---|
| `MODEL_DIR` | Ruta base donde se encuentran los modelos versionados | `/models` |
| `MODEL_VERSION` | Versión a cargar al arrancar | `v1` |
| `DATA_DIR` | Ruta al dataset | `cardio_train.csv` |

El modelo se carga en memoria al arrancar la aplicación (`startup` event). Una llamada a `/predict/cardiovascular` con el modelo no cargado retorna `503`.

---

## 7. `feature_list.json` — contrato de orden de features

El archivo `models/cvd_risk/v{n}/feature_list.json` define el orden exacto en que el modelo espera recibir las features. Cualquier cambio en este archivo implica una nueva versión de modelo (`v{n+1}`) y debe comunicarse al equipo de backend antes de desplegar.

```json
{
  "version": "v1",
  "features": [
    "age",
    "sex",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "smoker",
    "physical_activity_level",
    "family_history_cvd",
    "family_history_diabetes",
    "cholesterol_total",
    "glucose_mg_dl"
  ],
  "optional": [
    "family_history_cvd",
    "family_history_diabetes",
    "cholesterol_total",
    "glucose_mg_dl"
  ],
  "imputation_file": "imputation_values.json"
}
```

---

## 8. Healthcheck

`GET /health` — retorna `{ "status": "ok", "model_version": "v1" }` con código `200`. El backend .NET puede usar este endpoint para circuit breaker básico o monitoreo de disponibilidad.
