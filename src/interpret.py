"""Interpretability utilities:
- Model-specific feature importances (tree-based)
- Permutation importance on held-out test data
- Partial Dependence + ICE for top features
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance, PartialDependenceDisplay

def _feature_names_from_pipeline(pipe, original_feature_names: List[str]) -> List[str]:
    """Attempt to extract transformed feature names from the pipeline's ColumnTransformer.
    Falls back to the original names if not available.
    """
    try:
        ct = pipe.named_steps["preprocess"]
        names = list(ct.get_feature_names_out())
        # We used verbose_feature_names_out=False so these should match original names
        return names
    except Exception:
        return original_feature_names

def plot_model_specific_importance(pipe, feature_names: List[str], output_dir: Path) -> pd.DataFrame:
    """Plot tree-based model-specific feature importance (Gain-based for GBM)."""
    model = pipe.named_steps["model"]
    names = _feature_names_from_pipeline(pipe, feature_names)

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not expose feature_importances_. Use permutation importance instead.")

    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["feature"][:20][::-1], imp_df["importance"][:20][::-1])
    plt.xlabel("Importance (model-specific)")
    plt.title("Feature Importance — Gradient Boosting")
    (output_dir / "feature_importance_model.png").parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_model.png", dpi=150)
    plt.close()
    return imp_df

def plot_permutation_importance(pipe, X_test, y_test, feature_names: List[str], output_dir: Path, n_repeats: int = 10) -> pd.DataFrame:
    """Compute & plot permutation importance on held-out test data."""
    names = _feature_names_from_pipeline(pipe, feature_names)
    r = permutation_importance(pipe, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring="roc_auc")
    imp_df = pd.DataFrame({"feature": names, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
    imp_df = imp_df.sort_values("importance_mean", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["feature"][:20][::-1], imp_df["importance_mean"][:20][::-1], xerr=imp_df["importance_std"][:20][::-1])
    plt.xlabel("Importance (permutation, ROC-AUC drop)")
    plt.title("Permutation Importance — Held-out Test")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_permutation.png", dpi=150)
    plt.close()
    return imp_df

def plot_pdp_ice(pipe, X, features: List[str], output_dir: Path, ice_on_first: bool = True) -> None:
    """Plot PDP for given features; overlay ICE for the first feature by default."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # PDP for each feature
    for i, feat in enumerate(features):
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        display = PartialDependenceDisplay.from_estimator(
            estimator=pipe,
            X=X,
            features=[feat],
            kind=("both" if (ice_on_first and i == 0) else "average"),
            ax=ax,
        )
        title_suffix = " (PDP + ICE)" if (ice_on_first and i == 0) else " (PDP)"
        plt.title(f"Partial Dependence — {feat}{title_suffix}")
        plt.tight_layout()
        out_name = f"pdp_{feat.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(output_dir / out_name, dpi=150)
        plt.close(fig)