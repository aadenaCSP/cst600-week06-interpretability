"""Entry point: runs data load, training, evaluation, and interpretability suite."""
from pathlib import Path
import json

from data_load import load_data
from model_train import build_pipeline
from evaluate import evaluate_model
from interpret import plot_model_specific_importance, plot_permutation_importance, plot_pdp_ice

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # 2) Build & fit pipeline
    pipe = build_pipeline(feature_names)
    pipe.fit(X_train, y_train)

    # 3) Evaluate
    metrics = evaluate_model(pipe, X_test, y_test, FIG_DIR)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 4) Interpretability
    # 4a) Model-specific feature importance (tree-based)
    imp_model_df = plot_model_specific_importance(pipe, feature_names, FIG_DIR)
    imp_model_df.to_csv(OUT_DIR / "feature_importance_model_specific.csv", index=False)

    # 4b) Permutation importance on held-out test
    imp_perm_df = plot_permutation_importance(pipe, X_test, y_test, feature_names, FIG_DIR, n_repeats=20)
    imp_perm_df.to_csv(OUT_DIR / "feature_importance_permutation.csv", index=False)

    # 4c) PDP + ICE for top 3 features from permutation importance
    top_feats = imp_perm_df.head(3)["feature"].tolist()
    # Use training data when computing PDP/ICE to align with typical usage
    plot_pdp_ice(pipe, X_train, top_feats, FIG_DIR, ice_on_first=True)

    # 5) Save a brief stakeholder-friendly takeaway for each visual
    takeaways = []
    if not imp_perm_df.empty:
        f0 = top_feats[0]
        takeaways.append(f"Risk increases as {f0} rises; a sharper slope suggests a clinically meaningful threshold worth monitoring.")
    if len(top_feats) > 1:
        f1 = top_feats[1]
        takeaways.append(f"{f1} shows a monotonic pattern; incremental changes may have diminishing returns beyond a mid-range band.")
    if len(top_feats) > 2:
        f2 = top_feats[2]
        takeaways.append(f"Patient-level ICE lines for {f2} indicate heterogeneity; consider cohort-specific guidance for nuanced decisions.")

    (OUT_DIR / "clinician_takeaways.txt").write_text("\n".join(takeaways))

    print("Done. Figures saved in:", FIG_DIR)
    print("Metrics:", metrics)
    print("Top features (permutation):", top_feats)

if __name__ == "__main__":
    main()