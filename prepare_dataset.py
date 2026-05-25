"""
prepare_dataset.py
==================
Preprocesamiento del dataset de riesgo cardiovascular para VirtualMed.

Fuente: Cardiovascular Disease Dataset (Sulianova, Kaggle)
        https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Uso:
    python prepare_dataset.py --output data/processed/

Salidas:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    models/cvd_risk/feature_list.json
    models/cvd_risk/imputation_values.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# ── Configuración ─────────────────────────────────────────────────────────────

RANDOM_STATE = 42
MISSING_RATE_OPTIONAL = 0.30  # Proporción de missings artificiales en opcionales

# Features opcionales: llegan como null desde la API y se imputan con mediana del train
OPTIONAL_FEATURES = ["cholesterol_total", "glucose_mg_dl", "family_history_cvd"]

# Orden exacto que el modelo va a esperar (contrato con FastAPI y .NET)
FEATURE_ORDER = [
    "age",
    "sex",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "smoker",
    "physical_activity_level",
    "family_history_cvd",
    "cholesterol_total",
    "glucose_mg_dl",
]

TARGET = "cardio"

# Rangos clínicos válidos — registros fuera de rango se descartan como errores de entrada
CLINICAL_BOUNDS = {
    "systolic_bp":  (70,   250),
    "diastolic_bp": (40,   170),
    "height_m":     (1.30, 2.20),
    "weight_kg":    (30.0, 200.0),
    "age":          (30,   65),
}


# ── Funciones ─────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Carga el CSV original de Sulianova (separador ';')."""
    df = pd.read_csv(path, sep=",")
    print(f"[load]    {len(df):,} registros cargados — columnas: {list(df.columns)}")
    return df


def rename_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas al esquema de VirtualMed y deriva las que faltan.
    No elimina nada aquí — eso se hace en drop_unused().
    """
    df = df.copy()

    # age: días → años
    df["age"] = (df["age"] / 365).round().astype(int)

    # gender (1=mujer, 2=hombre) → sex (0=mujer, 1=hombre)
    df["sex"] = (df["gender"] - 1).astype(int)

    # presión arterial
    df.rename(columns={"ap_hi": "systolic_bp", "ap_lo": "diastolic_bp"}, inplace=True)

    # altura y peso
    df["height_m"]  = df["height"] / 100.0
    df["weight_kg"] = df["weight"].astype(float)

    # BMI calculado
    df["bmi"] = (df["weight_kg"] / df["height_m"] ** 2).round(2)

    # smoker y physical_activity_level
    df.rename(columns={"smoke": "smoker", "active": "physical_activity_level"}, inplace=True)

    # cholesterol y glucose como ordinales (1/2/3)
    df.rename(columns={"cholesterol": "cholesterol_total", "gluc": "glucose_mg_dl"}, inplace=True)

    return df


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas que no forman parte del contrato de features."""
    cols_to_drop = ["id", "alco", "gender", "height", "weight"]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[drop]    Columnas eliminadas: {existing}")
    return df


def remove_clinical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta registros con valores clínicamente imposibles."""
    before = len(df)
    for col, (lo, hi) in CLINICAL_BOUNDS.items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]
    removed = before - len(df)
    print(f"[filter]  {removed:,} registros eliminados por outliers clínicos — quedan {len(df):,}")
    return df


def add_synthetic_family_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera family_history_cvd sintéticamente.

    Es una feature obligatoria en el contrato de la API pero no existe en el
    dataset original. Se genera con prevalencia clínica razonable (25 % positivo)
    y semilla fija para reproducibilidad.

    Se documenta en DATASET_PROVENANCE.md.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    df = df.copy()
    df["family_history_cvd"] = rng.binomial(n=1, p=0.25, size=len(df)).astype(int)
    pos = df["family_history_cvd"].mean() * 100
    print(f"[synth]   family_history_cvd generado — {pos:.1f} % positivos")
    return df


def split_dataset(df: pd.DataFrame):
    """División estratificada 70/15/15 por target."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"[split]   train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_imputation_values(X_train: pd.DataFrame) -> dict:
    """
    Calcula la mediana del conjunto de entrenamiento para cada feature opcional.
    Estos valores se exportan y la API FastAPI los carga al arrancar para imputar
    requests con campos null.
    """
    imputation = {}
    for col in OPTIONAL_FEATURES:
        if col in X_train.columns:
            median_val = X_train[col].median()
            imputation[col] = float(median_val) if not np.isnan(median_val) else None
            print(f"[impute]  mediana de '{col}': {imputation[col]}")
    return imputation


def impute_train_data(X_train, X_val, X_test, imputation_values: dict):
    """Aplica imputación por mediana del train a los tres splits."""
    for col, median_val in imputation_values.items():
        if median_val is not None:
            for split in [X_train, X_val, X_test]:
                if col in split.columns:
                    split[col] = split[col].fillna(median_val)
    return X_train, X_val, X_test


def reorder_features(X: pd.DataFrame) -> pd.DataFrame:
    """Reordena columnas al orden exacto definido en FEATURE_ORDER."""
    missing_cols = [c for c in FEATURE_ORDER if c not in X.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing_cols}")
    extra_cols = [c for c in X.columns if c not in FEATURE_ORDER]
    if extra_cols:
        print(f"[order]   Columnas extra ignoradas: {extra_cols}")
    return X[FEATURE_ORDER]


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir: str):
    """Guarda los tres splits como CSV con el target incluido."""
    os.makedirs(output_dir, exist_ok=True)

    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        split_df = X.copy()
        split_df[TARGET] = y.values
        path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(path, index=False)
        print(f"[save]    {path} ({len(split_df):,} registros)")


def save_feature_list(output_dir: str, imputation_values: dict):
    """
    Exporta feature_list.json — contrato de orden de features para FastAPI y .NET.
    Este archivo es la fuente de verdad compartida entre el módulo de IA y el backend.
    """
    os.makedirs(output_dir, exist_ok=True)
    feature_list = {
        "version": os.path.basename(output_dir.rstrip("/")),
        "features": FEATURE_ORDER,
        "optional": OPTIONAL_FEATURES,
        "imputation_file": "imputation_values.json",
    }
    path = os.path.join(output_dir, "feature_list.json")
    with open(path, "w") as f:
        json.dump(feature_list, f, indent=2)
    print(f"[export]  {path}")


def save_imputation_values(output_dir: str, imputation_values: dict):
    """
    Exporta imputation_values.json — medianas del train para uso en producción.
    La API FastAPI carga este archivo al arrancar y lo aplica a campos null del request.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "imputation_values.json")
    with open(path, "w") as f:
        json.dump(imputation_values, f, indent=2)
    print(f"[export]  {path}")


def print_summary(X_train, y_train):
    """Imprime un resumen básico del conjunto de entrenamiento."""
    print("\n── Resumen del conjunto de entrenamiento ──────────────────────────")
    print(f"  Shape:          {X_train.shape}")
    print(f"  Target positivo: {y_train.mean() * 100:.1f} %")
    print(f"  Nulls por columna:\n{X_train.isnull().sum().to_string()}")
    print(f"  Tipos:\n{X_train.dtypes.to_string()}")
    print("───────────────────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento dataset CVD — VirtualMed")
    parser.add_argument("--input",   default="cardio_train.csv",    help="Ruta al CSV original")
    parser.add_argument("--output",  default="data/processed",  help="Directorio de salida para los splits")
    parser.add_argument("--models",  default="models/cvd_risk", help="Directorio de salida para artefactos del modelo")
    args = parser.parse_args()

    load_dotenv()
    data_dir = os.getenv("DATA_DIR", args.input)
    model_version = os.getenv("MODEL_VERSION")
    models_dir = os.path.join(args.models, model_version) if model_version else args.models

    print(f"\n=== VirtualMed — prepare_dataset.py ===")
    print(f"Input:   {data_dir}")
    print(f"Output:  {args.output}")
    print(f"Models:  {models_dir}\n")

    # 1. Carga
    df = load_raw(data_dir)

    # 2. Renombrar y derivar columnas
    df = rename_and_derive(df)

    # 3. Eliminar columnas innecesarias
    df = drop_unused(df)

    # 4. Filtrar outliers clínicos
    df = remove_clinical_outliers(df)

    # 5. Añadir variable sintética obligatoria
    df = add_synthetic_family_history(df)

    # 6. División estratificada 70/15/15
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    # 7. Calcular imputación con medianas del train
    imputation_values = compute_imputation_values(X_train)

    # 8. Imputar splits
    X_train, X_val, X_test = impute_train_data(X_train, X_val, X_test, imputation_values)

    # 9. Reordenar features al orden del contrato
    X_train = reorder_features(X_train)
    X_val   = reorder_features(X_val)
    X_test  = reorder_features(X_test)

    # 10. Resumen
    print_summary(X_train, y_train)

    # 11. Guardar splits
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, args.output)

    # 12. Exportar artefactos
    save_feature_list(models_dir, imputation_values)
    save_imputation_values(models_dir, imputation_values)

    print("\n✓ Preprocesamiento completado.\n")


if __name__ == "__main__":
    main()