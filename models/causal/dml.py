"""EconML CausalForestDML wrapper estimating heterogeneous treatment
effects of supervision intensity / program enrollment on reoffense.

This produces an effect-of-intervention estimate (CATE), not a
risk-of-reoffense score. Per the handoff constraints, no output from here
should be surfaced anywhere (notebook, report, CLI) without the matching
refutation results from `models/causal/refutation.py` attached.
"""

from __future__ import annotations

import pandas as pd
from econml.dml import CausalForestDML


def fit_causal_forest_dml(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    n_estimators: int = 500,
    random_state: int = 42,
) -> CausalForestDML:
    raise NotImplementedError
