"""Data loading and splitting utilities for Week 6 interpretability project."""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def load_data(test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset and split into train/test.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    ds = load_breast_cancer(as_frame=True)
    X = ds.data.copy()
    y = ds.target.copy()  # 0=malignant, 1=benign in sklearn loader

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, feature_names