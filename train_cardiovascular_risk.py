"""
train_cardiovascular_risk.py
============================
Entrenamiento del modelo de riesgo cardiovascular para VirtualMed.

Entrena y compara Random Forest vs XGBoost, selecciona el de mayor AUC-ROC
en validación, calibra las probabilidades y exporta el artefacto final.

Prerequisito: haber ejecutado prepare_dataset.py

Uso:
    python train_cardiovascular_risk.py
    python train_cardiovascular_risk.py --data data/processed/ --models models/cvd_risk/

Salidas:
    models/cvd_risk/model.joblib
    models/cvd_risk/metadata.json
    models/cvd_risk/feature_importance.json
    models/cvd_risk/feature_importance.png
    models/cvd_risk/calibration_curve.png
"""

import argparse
import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuración ─────────────────────────────────────────────────────────────

RANDOM_STATE = 42
TARGET = "cardio"

# Umbrales para mapear probabilidad → risk_level (documentados en metadata.json)
THRESHOLD_LOW    = 0.30
THRESHOLD_HIGH   = 0.60

MODELS_TO_COMPARE = {
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,          # dataset balanceado, no se necesita ajuste
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    ),
}


# ── Carga de datos ─────────────────────────────────────────────────────────────

def load_splits(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_val,   y_val   = val.drop(columns=[TARGET]),   val[TARGET]
    X_test,  y_test  = test.drop(columns=[TARGET]),  test[TARGET]

    print(f"[load]  train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")
    print(f"[load]  features: {list(X_train.columns)}\n")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Entrenamiento y comparación ────────────────────────────────────────────────

def train_and_compare(X_train, y_train, X_val, y_val) -> tuple[str, object, float]:
    """
    Entrena todos los modelos en MODELS_TO_COMPARE y selecciona el de mayor
    AUC-ROC en el conjunto de validación.
    Retorna (nombre, modelo, auc_val).
    """
    results = {}
    print("── Comparación de modelos ────────────────────────────────────────")
    for name, clf in MODELS_TO_COMPARE.items():
        print(f"  Entrenando {name}...", end=" ", flush=True)
        clf.fit(X_train, y_train)
        prob_val = clf.predict_proba(X_val)[:, 1]
        auc_val  = roc_auc_score(y_val, prob_val)
        f1_val   = f1_score(y_val, (prob_val >= 0.5).astype(int))
        results[name] = {"model": clf, "auc": auc_val, "f1": f1_val}
        print(f"AUC-ROC={auc_val:.4f} | F1={f1_val:.4f}")

    best_name = max(results, key=lambda k: results[k]["auc"])
    best      = results[best_name]
    print(f"\n  ✓ Modelo seleccionado: {best_name} (AUC-ROC val={best['auc']:.4f})\n")
    return best_name, best["model"], best["auc"]


# ── Calibración ───────────────────────────────────────────────────────────────

def calibrate_model(model, X_train, y_train):
    """
    Calibra las probabilidades con isotonic regression usando 5-fold CV
    sobre el conjunto de entrenamiento.

    Nota: cv="prefit" fue eliminado en scikit-learn >= 1.3. La alternativa
    correcta es cv=5, que entrena clones del modelo en folds internos para
    ajustar la calibración sin data leakage.
    """
    print("[calibration]  Aplicando isotonic regression (cv=5)...", end=" ", flush=True)
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    print("✓\n")
    return calibrated


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(model, X, y, split_name: str) -> dict:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    auc_roc = roc_auc_score(y, prob)
    f1      = f1_score(y, pred)
    brier   = brier_score_loss(y, prob)   # calibración básica: menor es mejor

    print(f"  [{split_name:5s}]  AUC-ROC={auc_roc:.4f} | F1={f1:.4f} | Brier={brier:.4f}")
    return {"auc_roc": round(auc_roc, 4), "f1": round(f1, 4), "brier_score": round(brier, 4)}


# ── Feature importance ────────────────────────────────────────────────────────

def export_feature_importance(model, feature_names: list, output_dir: str):
    """
    Extrae importancias del modelo base (antes de calibración) y las exporta
    como JSON y PNG ordenadas de mayor a menor.
    """
    # CalibratedClassifierCV envuelve al modelo base — acceder con .estimator
    base = model.estimator if hasattr(model, "estimator") else model
    importances = base.feature_importances_

    importance_dict = dict(zip(feature_names, importances.tolist()))
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    # JSON
    json_path = os.path.join(output_dir, "feature_importance.json")
    with open(json_path, "w") as f:
        json.dump(sorted_importance, f, indent=2)
    print(f"[export]  {json_path}")

    # PNG
    fig, ax = plt.subplots(figsize=(8, 5))
    features_sorted = list(sorted_importance.keys())
    values_sorted   = list(sorted_importance.values())

    bars = ax.barh(features_sorted[::-1], values_sorted[::-1], color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Importancia (Gini / gain)", fontsize=11)
    ax.set_title("Feature Importance — Riesgo Cardiovascular", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[export]  {png_path}")

    return sorted_importance


# ── Curva de calibración ──────────────────────────────────────────────────────

def plot_calibration_curve(model, X_test, y_test, model_name: str, output_dir: str):
    prob = model.predict_proba(X_test)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, fraction_pos, marker="o", label=model_name, color="#4C72B0")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
    ax.set_xlabel("Probabilidad predicha promedio")
    ax.set_ylabel("Fracción de positivos reales")
    ax.set_title("Curva de calibración — Test set", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(output_dir, "calibration_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[export]  {path}")


# ── Curva ROC ─────────────────────────────────────────────────────────────────

def plot_roc_curve(model, X_test, y_test, output_dir: str):
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC — Test set", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[export]  {path}")


# ── Guardado de artefactos ────────────────────────────────────────────────────

def save_model(model, output_dir: str):
    path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, path)
    print(f"[export]  {path}")


def save_metadata(
    model_name: str,
    feature_names: list,
    metrics_val: dict,
    metrics_test: dict,
    feature_importance: dict,
    version: str,
    output_dir: str,
):
    metadata = {
        "model_name": model_name,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "features": feature_names,
        "optional_features": ["cholesterol_total", "glucose_mg_dl", "family_history_cvd"],
        "thresholds": {
            "low":    {"max_probability": THRESHOLD_LOW,  "max_score": int(THRESHOLD_LOW * 100)},
            "medium": {"min_probability": THRESHOLD_LOW,  "max_probability": THRESHOLD_HIGH,
                       "min_score": int(THRESHOLD_LOW * 100), "max_score": int(THRESHOLD_HIGH * 100)},
            "high":   {"min_probability": THRESHOLD_HIGH, "min_score": int(THRESHOLD_HIGH * 100)},
        },
        "metrics": {
            "validation": metrics_val,
            "test":       metrics_test,
        },
        "feature_importance": feature_importance,
        "calibration": "isotonic (CalibratedClassifierCV, cv=prefit)",
        "notes": (
            "Dataset: Cardiovascular Disease Dataset (Sulianova, Kaggle, CC0). "
            "family_history_cvd generada sintéticamente (Bernoulli p=0.25). "
            "cholesterol_total y glucose_mg_dl son valores ordinales 1/2/3, no unidades clínicas reales. "
            "Score es orientativo — no constituye diagnóstico médico."
        ),
    }

    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[export]  {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento modelo CVD — VirtualMed")
    parser.add_argument("--data",    default="data/processed/",        help="Directorio con train/val/test.csv")
    parser.add_argument("--models",  default="models/cvd_risk",       help="Directorio de salida de artefactos")
    parser.add_argument("--version", default="v1",                     help="Versión del modelo")
    args = parser.parse_args()

    output_dir = os.path.join(args.models, args.version) if args.version else args.models
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== VirtualMed — train_cardiovascular_risk.py ===")
    print(f"Data:    {args.data}")
    print(f"Output:  {output_dir}")
    print(f"Version: {args.version}\n")

    # 1. Carga
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(args.data)
    feature_names = list(X_train.columns)

    # 2. Comparar y seleccionar mejor modelo
    best_name, best_model, _ = train_and_compare(X_train, y_train, X_val, y_val)

    # 3. Calibrar sobre train (cv=5)
    calibrated_model = calibrate_model(best_model, X_train, y_train)

    # 4. Métricas en val y test
    print("── Métricas finales (modelo calibrado) ───────────────────────────")
    metrics_val  = compute_metrics(calibrated_model, X_val,   y_val,   "val")
    metrics_test = compute_metrics(calibrated_model, X_test,  y_test,  "test")
    print()

    # 5. Feature importance
    feature_importance = export_feature_importance(calibrated_model, feature_names, output_dir)

    # 6. Gráficos de calibración y ROC
    plot_calibration_curve(calibrated_model, X_test, y_test, best_name, output_dir)
    plot_roc_curve(calibrated_model, X_test, y_test, output_dir)

    # 7. Guardar modelo y metadata
    save_model(calibrated_model, output_dir)
    save_metadata(
        model_name=best_name,
        feature_names=feature_names,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        feature_importance=feature_importance,
        version=args.version,
        output_dir=output_dir,
    )

    print(f"\n✓ Entrenamiento completado — artefactos en {output_dir}\n")

if __name__ == "__main__":
    main()
