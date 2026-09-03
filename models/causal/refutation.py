"""DoWhy refutation tests for the CausalForestDML estimate: placebo
treatment, random common cause, and data-subset refuters.

Unconfoundedness is a stated assumption, not a proven property — these
tests are the closest thing to evidence for it, and every causal estimate
must ship with its refutation results attached (see
models/causal/dml.py::estimate_treatment_effect, the intended entry
point).

A refutation "passes" when the refuted effect stays close to the original
estimate (random_common_cause, data_subset_refuter) or collapses toward
zero (placebo_treatment_refuter) — this module doesn't make that pass/fail
call itself, since what counts as "close enough" is a judgment call for
whoever's reading the report, not a hardcoded threshold.
"""

from __future__ import annotations

from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_identifier import IdentifiedEstimand

# Default kwargs per refuter, passed to CausalModel.refute_estimate. Only
# random_common_cause needs none.
#
# placebo_treatment_refuter deliberately omits placebo_type="permute": that
# path (dowhy/causal_refuters/placebo_treatment_refuter.py's PERMUTE branch)
# does `data[treatment_names].values` where treatment_names is a list, which
# returns a 2D array even for a single treatment column. Older pandas
# silently handled assigning that as a new column; pandas 3.0's stricter
# maybe_convert_objects raises `ValueError: Buffer has wrong number of
# dimensions (expected 1, got 2)` instead. That's a real bug in dowhy 0.14
# against pandas 3.0, not something fixable from here — the default
# placebo_type (resample the treatment from a random distribution instead
# of permuting it) hits a different code path that works correctly and is
# still a legitimate placebo strategy, so that's what's used instead.
_REFUTER_KWARGS: dict[str, dict] = {
    "placebo_treatment_refuter": {},
    "random_common_cause": {},
    "data_subset_refuter": {"subset_fraction": 0.8},
}


def _summarize(refutation) -> dict:
    p_value = None
    if getattr(refutation, "refutation_result", None):
        p_value = refutation.refutation_result.get("p_value")
    return {
        "refutation_type": refutation.refutation_type,
        "estimated_effect": refutation.estimated_effect,
        "new_effect": refutation.new_effect,
        "p_value": p_value,
    }


def run_refuters(
    causal_model: CausalModel,
    identified_estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    refuters: list[str],
    random_state: int | None = None,
) -> dict[str, dict]:
    """Run each named refuter (config/default.yaml's
    `module_a.causal.refuters`) against `estimate` and return a summary
    dict per refuter: {refutation_type, estimated_effect, new_effect,
    p_value}.

    Raises on an unrecognized refuter name rather than silently skipping
    it — a typo here shouldn't quietly produce an unrefuted estimate.
    """
    unknown = set(refuters) - set(_REFUTER_KWARGS)
    if unknown:
        raise ValueError(f"Unknown refuter(s): {sorted(unknown)}")

    results = {}
    for name in refuters:
        kwargs = dict(_REFUTER_KWARGS[name])
        if random_state is not None:
            kwargs["random_seed"] = random_state
        refutation = causal_model.refute_estimate(
            identified_estimand, estimate, method_name=name, **kwargs
        )
        results[name] = _summarize(refutation)
    return results
