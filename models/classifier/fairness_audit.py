"""Fairness audit: calibration + false-positive-rate parity across
demographic subgroups, replicating the ProPublica COMPAS critique as a
sanity check.

Works off plain (y_true, y_pred / y_pred_proba, protected_df) inputs rather
than a specific model type, so the same functions run on the baseline
classifier (models/classifier/train.py) now and on CATE outputs
(models/causal/dml.py) later — per the handoff constraint that the
fairness audit covers both Module A layers.

Subgroups smaller than `subgroup_min_n` are dropped rather than reported:
a metric computed on a handful of rows is noise, not a finding.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def _iter_subgroups(
    protected_df: pd.DataFrame, subgroup_min_n: int
) -> Iterator[tuple[str, object, int, np.ndarray]]:
    for attribute in protected_df.columns:
        counts = protected_df[attribute].value_counts()
        for subgroup, n in counts.items():
            if n < subgroup_min_n:
                continue
            mask = (protected_df[attribute] == subgroup).to_numpy()
            yield attribute, subgroup, int(n), mask


def audit_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    protected_df: pd.DataFrame,
    subgroup_min_n: int = 30,
) -> pd.DataFrame:
    """Per-subgroup calibration: mean predicted probability vs. observed
    reoffense rate. A large `calibration_gap` means the model over- or
    under-predicts risk for that subgroup relative to what actually
    happened.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    rows = []
    for attribute, subgroup, n, mask in _iter_subgroups(protected_df, subgroup_min_n):
        mean_predicted = float(y_pred_proba[mask].mean())
        observed_rate = float(y_true[mask].mean())
        rows.append(
            {
                "attribute": attribute,
                "subgroup": subgroup,
                "n": n,
                "mean_predicted": mean_predicted,
                "observed_rate": observed_rate,
                "calibration_gap": mean_predicted - observed_rate,
            }
        )
    return pd.DataFrame(rows)


def audit_fpr_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_df: pd.DataFrame,
    subgroup_min_n: int = 30,
) -> pd.DataFrame:
    """Per-subgroup false-positive rate: among subjects who did not
    reoffend, the share the model flagged as high-risk anyway. This is the
    metric at the center of the ProPublica COMPAS critique (materially
    higher FPR for Black defendants than White defendants).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rows = []
    for attribute, subgroup, n, mask in _iter_subgroups(protected_df, subgroup_min_n):
        negatives = mask & (y_true == 0)
        n_negatives = int(negatives.sum())
        if n_negatives == 0:
            continue
        false_positives = int((y_pred[negatives] == 1).sum())
        rows.append(
            {
                "attribute": attribute,
                "subgroup": subgroup,
                "n": n,
                "n_negatives": n_negatives,
                "false_positive_rate": false_positives / n_negatives,
            }
        )
    return pd.DataFrame(rows)


def run_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    protected_df: pd.DataFrame,
    metrics: list[str],
    subgroup_min_n: int = 30,
) -> dict[str, pd.DataFrame]:
    """Run the configured fairness metrics (config/default.yaml's
    `module_a.fairness_audit.metrics`) and return one dataframe per metric.
    """
    results: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        if metric == "calibration":
            results[metric] = audit_calibration(y_true, y_pred_proba, protected_df, subgroup_min_n)
        elif metric == "false_positive_rate_parity":
            results[metric] = audit_fpr_parity(y_true, y_pred, protected_df, subgroup_min_n)
        else:
            raise ValueError(f"Unknown fairness metric {metric!r}")
    return results
