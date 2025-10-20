"""Model training utilities with scikit-learn Pipeline to prevent leakage."""
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier

RANDOM_STATE = 42

def build_pipeline(feature_names: List[str]) -> Pipeline:
    """Create a preprocessing + model pipeline.

    For this dataset all features are numeric, but a ColumnTransformer is kept
    to illustrate best practice for mixed-type data.
    """
    numeric_features = feature_names  # all numeric in this dataset

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True), numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    gbm = GradientBoostingClassifier(random_state=RANDOM_STATE)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", gbm),
        ]
    )
    return pipe