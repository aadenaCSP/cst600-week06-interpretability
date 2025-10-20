# Week 6 — Interpreting Complex Models in Healthcare

**Goal:** apply multiple interpretability techniques (feature importance, permutation importance, PDP + ICE; optional boundary slices) to make a complex model transparent to clinical stakeholders.

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** — de‑identified tabular dataset curated by the UCI ML Repository (originally from the University of Wisconsin).  
- UCI page: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  
- Accessed through `scikit-learn` helper loader: `sklearn.datasets.load_breast_cancer()`

Why this dataset?
- Widely used, de‑identified, tabular, binary classification (malignant vs. benign).
- Suits exploration of clinical interpretability patterns (e.g., thresholds, monotonic trends).

**Notes & constraints**
- No direct PHI; avoid adding sensitive attributes or proxies.
- Pipeline ensures train/test separation to prevent leakage.
- Fixed `random_state` for reproducibility.

---

## Environment Setup

```bash
# (recommended) inside this folder
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## How to Run

```bash
# From the repo root (this folder)
python src/main.py
```

This will:
- Load & split data with a **stratified** hold‑out test set.
- Train a **GradientBoostingClassifier** inside a scikit‑learn **Pipeline**.
- Evaluate metrics on the test set (Accuracy, Precision, Recall, F1, ROC‑AUC, Brier score).
- Generate interpretability outputs:
  - **Model‑specific feature importances** (from the tree model).
  - **Permutation importance** computed on the **held‑out test** split.
  - **Partial Dependence Plots (PDP) and ICE** for 2–3 important features.
- Save all figures to `figures/` and a short summary report to `outputs/` (auto‑created).

---

## Project Structure (example)

```
<repo>/
  README.md
  requirements.txt
  .gitignore
  src/
    data_load.py
    model_train.py
    evaluate.py
    interpret.py
    main.py
  figures/
  data/
    raw/
```

---

## Methods & Findings (concise)

- **Model:** Gradient Boosting (tree‑based) with standardized numeric inputs via `Pipeline`.
- **Why GBM?** Handles nonlinearities and interactions common in clinical data while remaining compact.
- **Feature importance:** Tree‑based importance highlights key predictors; permutation importance provides a robust, model‑agnostic baseline checked on **held‑out** data.
- **PDP + ICE:** Reveal monotonic patterns and threshold effects for top predictors, and visualize patient‑level heterogeneity.
- **Evaluation:** Reports Accuracy, macro Precision/Recall/F1, ROC‑AUC, and Brier score (for probability calibration context).
- **Limitations:** Correlated features can distort PDP/importance; small‑sample variance; potential dataset shift in practice.
- **Next steps:** Consider ALE profiles, cohort‑stratified PDPs, calibration curves, and domain expert review before deployment.
