"""Evaluation helpers: classification metrics, ROC curve, calibration summary."""
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, brier_score_loss, classification_report
)

def evaluate_model(pipe, X_test, y_test, output_dir: Path) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "brier_score": float(brier_score_loss(y_test, y_proba)),
    }

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Gradient Boosting (Test)")
    plt.legend(loc="lower right")
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight", dpi=150)
    plt.close()

    # Text report
    report = classification_report(y_test, y_pred, zero_division=0)
    (output_dir / "report.txt").write_text(report + "\n\n" + str(metrics))

    return metrics