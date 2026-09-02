"""Baseline recidivism classifier (XGBoost / logistic regression) on
COMPAS + NIJ data. This is the pre-causal sanity-check model — see
`models/causal/dml.py` for the treatment-effect estimator.

Protected attributes (race_ethnicity, sex) are deliberately excluded from
the model's input features. They aren't predictors here — they're audited
for disparate impact in `fairness_audit.py` after the fact, which is where
the ProPublica-style critique actually needs them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.schema import validate_recidivism_schema

NUMERIC_FEATURES = ["age", "prior_convictions"]
CATEGORICAL_FEATURES = ["offense_type", "supervision_intensity"]
OUTCOME_COLUMN = "recidivated"


@dataclass
class BaselineResult:
    pipeline: Pipeline
    feature_columns: list[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    protected_test: pd.DataFrame  # protected-attribute values aligned to X_test/y_test, for fairness_audit
    metrics: dict[str, float]


def _build_pipeline(model: str, random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    if model == "xgboost":
        estimator = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, random_state=random_state, eval_metric="logloss"
        )
    elif model == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Unknown model {model!r}; use 'xgboost' or 'logistic_regression'.")

    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def train_baseline(
    df: pd.DataFrame,
    model: str = "xgboost",
    protected_attributes: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BaselineResult:
    """Train the baseline recidivism classifier and hold out a test split.

    A protected attribute absent from `df` (e.g. `race_ethnicity` for a
    jurisdiction that doesn't record it, per RECIDIVISM_SCHEMA) is silently
    skipped rather than raising, so this runs the same way across sources.
    """
    validate_recidivism_schema(df)
    protected_attributes = [a for a in (protected_attributes or []) if a in df.columns]

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[OUTCOME_COLUMN].astype(int)

    split = train_test_split(
        X, y, df.index.to_series(), test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test, y_train, y_test, idx_train, idx_test = split

    pipeline = _build_pipeline(model, random_state)
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {"accuracy": accuracy_score(y_test, y_pred)}
    if y_test.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)

    protected_test = df.loc[idx_test, protected_attributes] if protected_attributes else pd.DataFrame(index=idx_test)

    return BaselineResult(
        pipeline=pipeline,
        feature_columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        protected_test=protected_test,
        metrics=metrics,
    )
