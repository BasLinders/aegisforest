"""EconML CausalForestDML wrapper estimating heterogeneous treatment
effects of supervision intensity / program enrollment on reoffense,
routed through DoWhy's econml estimator interface so the resulting
estimate carries the identified estimand `refutation.py` needs.

This produces an effect-of-intervention estimate (CATE), not a
risk-of-reoffense score. Per the handoff constraints, no output from here
should be surfaced anywhere (notebook, report, CLI) without the matching
refutation results attached — `estimate_treatment_effect` is the intended
entry point precisely because it can't return one without the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_identifier import IdentifiedEstimand

from models.causal.dag import build_causal_model
from models.causal.refutation import run_refuters


def fit_causal_forest_dml(
    causal_model: CausalModel,
    identified_estimand: IdentifiedEstimand,
    control_value: int = 0,
    treatment_value: int = 1,
    n_estimators: int = 500,
    random_state: int = 42,
    cv: int = 2,
) -> CausalEstimate:
    """Estimate the ATE of `treatment_value` vs. `control_value` with
    EconML's CausalForestDML, via DoWhy's `estimate_effect`.

    Low-level building block — prefer `estimate_treatment_effect` below,
    which pairs this with the required refutation tests.
    """
    return causal_model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.dml.CausalForestDML",
        control_value=control_value,
        treatment_value=treatment_value,
        target_units="ate",
        confidence_intervals=False,
        method_params={
            "init_params": {
                "n_estimators": n_estimators,
                "random_state": random_state,
                "discrete_treatment": True,
                "cv": cv,
            },
            "fit_params": {},
        },
    )


@dataclass
class CausalEffectResult:
    causal_model: CausalModel
    identified_estimand: IdentifiedEstimand
    estimate: CausalEstimate
    refutation_results: dict

    @property
    def ate(self) -> float:
        return self.estimate.value


def estimate_treatment_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    dag_path: str,
    refuters: list[str],
    control_value: int = 0,
    treatment_value: int = 1,
    n_estimators: int = 500,
    random_state: int = 42,
) -> CausalEffectResult:
    """Identify, estimate, and refute the treatment effect in one call.

    This is the entry point notebooks/reports/CLI should use: it always
    returns refutation results alongside the estimate, so there's no way
    to surface a CausalForestDML output without them.
    """
    causal_model = build_causal_model(df, treatment, outcome, dag_path)
    identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    estimate = fit_causal_forest_dml(
        causal_model,
        identified_estimand,
        control_value=control_value,
        treatment_value=treatment_value,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    refutation_results = run_refuters(causal_model, identified_estimand, estimate, refuters)

    return CausalEffectResult(
        causal_model=causal_model,
        identified_estimand=identified_estimand,
        estimate=estimate,
        refutation_results=refutation_results,
    )
