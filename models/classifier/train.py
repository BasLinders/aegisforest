"""Baseline recidivism classifier (XGBoost / logistic regression) on
COMPAS + NIJ data. This is the pre-causal sanity-check model — see
`models/causal/dml.py` for the treatment-effect estimator.
"""

from __future__ import annotations

import pandas as pd


def train_baseline(df: pd.DataFrame, model: str = "xgboost", **kwargs):
    raise NotImplementedError
