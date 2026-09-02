import numpy as np
import pandas as pd
import pytest

from models.classifier.fairness_audit import (
    audit_calibration,
    audit_fpr_parity,
    run_fairness_audit,
)
from models.classifier.train import train_baseline
from tests.test_classifier_train import _fixture_df


def _biased_fixture(n_per_group: int = 100, random_state: int = 0):
    """Two groups, same true reoffense rate, but the model over-predicts
    risk for group B — a synthetic, unambiguous calibration/FPR gap to
    assert against."""
    rng = np.random.default_rng(random_state)

    group = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    y_true = rng.random(2 * n_per_group) < 0.3  # same 30% base rate for both groups

    y_pred_proba = np.empty(2 * n_per_group)
    y_pred_proba[: n_per_group] = np.clip(rng.normal(0.30, 0.05, n_per_group), 0, 1)  # well-calibrated
    y_pred_proba[n_per_group:] = np.clip(rng.normal(0.60, 0.05, n_per_group), 0, 1)  # over-predicts

    y_pred = (y_pred_proba >= 0.5).astype(int)
    protected_df = pd.DataFrame({"group": group})
    return y_true.astype(int), y_pred, y_pred_proba, protected_df


def test_audit_calibration_detects_gap():
    y_true, y_pred, y_pred_proba, protected_df = _biased_fixture()
    result = audit_calibration(y_true, y_pred_proba, protected_df, subgroup_min_n=30)

    gap_a = result.loc[result["subgroup"] == "A", "calibration_gap"].item()
    gap_b = result.loc[result["subgroup"] == "B", "calibration_gap"].item()
    assert abs(gap_a) < 0.05
    assert gap_b > 0.2


def test_audit_fpr_parity_detects_disparity():
    y_true, y_pred, y_pred_proba, protected_df = _biased_fixture()
    result = audit_fpr_parity(y_true, y_pred, protected_df, subgroup_min_n=30)

    fpr_a = result.loc[result["subgroup"] == "A", "false_positive_rate"].item()
    fpr_b = result.loc[result["subgroup"] == "B", "false_positive_rate"].item()
    assert fpr_b > fpr_a


def test_small_subgroups_are_dropped():
    y_true, y_pred, y_pred_proba, protected_df = _biased_fixture(n_per_group=10)
    result = audit_calibration(y_true, y_pred_proba, protected_df, subgroup_min_n=30)
    assert result.empty


def test_run_fairness_audit_dispatches_configured_metrics():
    y_true, y_pred, y_pred_proba, protected_df = _biased_fixture()
    results = run_fairness_audit(
        y_true, y_pred, y_pred_proba, protected_df,
        metrics=["calibration", "false_positive_rate_parity"],
        subgroup_min_n=30,
    )
    assert set(results) == {"calibration", "false_positive_rate_parity"}


def test_run_fairness_audit_rejects_unknown_metric():
    y_true, y_pred, y_pred_proba, protected_df = _biased_fixture()
    with pytest.raises(ValueError, match="Unknown fairness metric"):
        run_fairness_audit(y_true, y_pred, y_pred_proba, protected_df, metrics=["not_a_metric"])


def test_integrates_with_train_baseline_output():
    df = _fixture_df(n=300)
    baseline = train_baseline(df, protected_attributes=["race_ethnicity", "sex"])

    results = run_fairness_audit(
        baseline.y_test.to_numpy(),
        baseline.y_pred,
        baseline.y_pred_proba,
        baseline.protected_test,
        metrics=["calibration", "false_positive_rate_parity"],
        subgroup_min_n=5,
    )
    assert set(results["calibration"]["attribute"]) <= {"race_ethnicity", "sex"}
