# Dataset Provenance — Módulo de IA VirtualMed

**Responsable:** Equipo VirtualMed  
**Última actualización:** 2026-05-20  
**Modelo cubierto:** Riesgo cardiovascular (`cvd_risk`)

---

## 1. Fuente principal

| Campo | Detalle |
|---|---|
| Nombre | Cardiovascular Disease Dataset |
| Autor original | Svetlana Ulianova |
| Plataforma | Kaggle |
| URL | https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset |
| Archivo | `cardio_train.csv` |
| Registros | 70 000 |
| Features originales | 11 + target (`cardio`) |
| Formato | CSV separado por `;` |
| Fecha de descarga | 2026-05-20 |

### Licencia

El dataset está publicado bajo licencia **CC0: Public Domain**. No impone restricciones de uso, redistribución ni atribución. Se puede usar libremente para investigación, desarrollo y despliegue en producción.

---

## 2. Cumplimiento — Ley 1581 de 2012 (Colombia)

Este dataset **no contiene datos personales identificables de usuarios de VirtualMed**. Los registros provienen de una fuente pública anónima externa y no guardan ninguna relación con la base de pacientes de la plataforma.

Se establecen las siguientes restricciones operativas para mantener este cumplimiento:

- **Prohibido** incorporar registros clínicos reales de pacientes de VirtualMed al dataset de entrenamiento sin consentimiento informado explícito y aprobación del delegado de protección de datos.
- **Prohibido** almacenar el `InputSnapshot` (features enviadas al modelo) de forma que permita reidentificar a un paciente sin las salvaguardas definidas en la política de privacidad de VirtualMed.
- El campo `InputSnapshot` en la entidad `RiskScore` se considera dato de salud sensible y debe tratarse según el artículo 6 de la Ley 1581 (datos sensibles requieren consentimiento explícito).
- Cualquier reetrenamiento futuro con datos reales de VirtualMed debe pasar por revisión legal antes de ejecutarse.

---

## 3. Variables originales y transformaciones aplicadas

| Columna original | Tipo original | Feature resultante | Transformación |
|---|---|---|---|
| `age` | int (días) | `age` | `round(age / 365)` → años enteros |
| `gender` | int (1=mujer, 2=hombre) | `sex` | Renombrado; recodificado a 0=mujer, 1=hombre |
| `height` | int (cm) | `height_m` | `height / 100` → metros |
| `weight` | float (kg) | `weight_kg` | Sin transformación |
| `height` + `weight` | — | `bmi` | `weight / (height/100)²` |
| `ap_hi` | int (mmHg) | `systolic_bp` | Renombrado |
| `ap_lo` | int (mmHg) | `diastolic_bp` | Renombrado |
| `cholesterol` | int ordinal (1/2/3) | `cholesterol_total` | Mantenido como ordinal. **No representa mg/dL** — ver limitaciones §6 |
| `gluc` | int ordinal (1/2/3) | `glucose_mg_dl` | Mantenido como ordinal. **No representa mg/dL** — ver limitaciones §6 |
| `smoke` | int (0/1) | `smoker` | Renombrado |
| `active` | int (0/1) | `physical_activity_level` | Renombrado |
| `alco` | int (0/1) | — | **Eliminado.** No forma parte del contrato de features de la API |
| `id` | int | — | **Eliminado.** Identificador sin valor predictivo |
| `cardio` | int (0/1) | target | Sin transformación |

### Variables sintéticas añadidas

Las siguientes variables forman parte del contrato de la API pero no existen en el dataset original. Se generaron sintéticamente con distribuciones de prevalencia clínicamente razonables:

| Feature | Método de generación | Prevalencia asumida | Missing artificiales |
|---|---|---|---|
| `family_history_cvd` | Bernoulli | 25 % positivo | 30 % de registros con valor ausente |
| `family_history_diabetes` | Bernoulli | 20 % positivo | 30 % de registros con valor ausente |

La generación se realiza con semilla fija (`random_state=42`) en `prepare_dataset.py` para garantizar reproducibilidad.

---

## 4. Preprocesamiento aplicado (`prepare_dataset.py`)

1. **Eliminación de outliers clínicos:** se descartan registros con `systolic_bp < 70`, `systolic_bp > 250`, `diastolic_bp < 40`, `diastolic_bp > 170`, `height_m < 1.30`, `height_m > 2.20`, `weight_kg < 30`, `weight_kg > 200`. Estos valores son clínicamente imposibles y representan errores de entrada.
2. **Imputación de opcionales:** las features `cholesterol_total`, `glucose_mg_dl`, `family_history_cvd` y `family_history_diabetes` pueden llegar como `null` desde la API. Se imputa con la **mediana del conjunto de entrenamiento**. Los valores de mediana se exportan en `models/cvd_risk/v{n}/imputation_values.json` y se cargan en la API FastAPI al arrancar.
3. **División:** 70 % entrenamiento / 15 % validación / 15 % test. División estratificada por target para preservar balance de clases. Semilla `random_state=42`.
4. **Exportación de feature list:** se genera `feature_list.json` con el orden exacto de features que el modelo espera. Este archivo es consumido por la API FastAPI y por el backend .NET al construir el payload.

---

## 5. Balance de clases

El dataset original presenta clases balanceadas (~50 % positivo / ~50 % negativo). No se aplicó oversampling ni undersampling. Si versiones futuras del dataset presentan desbalance significativo (>65/35), se evaluará `class_weight='balanced'` en el modelo y se documentará en `metadata.json`.

---

## 6. Limitaciones y sesgos conocidos

| Limitación | Impacto | Mitigación |
|---|---|---|
| `cholesterol_total` y `glucose_mg_dl` son valores ordinales (1/2/3), no unidades clínicas reales (mg/dL) | El modelo no puede interpretar valores numéricos exactos de laboratorio | Documentado en el contrato de API; la UI debe reflejar esto |
| Población del dataset presumiblemente no latinoamericana | El modelo puede subestimar o sobreestimar riesgo en población colombiana | El score se presenta como **orientativo**, no diagnóstico. Disclaimer obligatorio en UI |
| Variables de historia familiar generadas sintéticamente | Su peso predictivo real puede diferir del esperado clínicamente | Se monitorea en `feature_importance.json`; si el peso es anómalo se considera eliminarlas |
| Rango de edad: 30–65 años | El modelo no fue validado fuera de este rango | La API valida el rango y retorna error 422 si `age` está fuera de [30, 65] |
| Dataset estático, sin actualizaciones | El modelo puede degradarse con el tiempo | Se recomienda reevaluar métricas con datos reales anualmente |

---

## 7. Versionado de modelos

Cada reentrenamiento genera una versión en `models/cvd_risk/v{n}/` con su propio `metadata.json`. El dataset procesado correspondiente se archiva en `data/processed/v{n}/` para reproducibilidad completa. No se sobreescriben versiones anteriores.
