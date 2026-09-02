"""DoWhy causal graph definition for the supervision-intensity /
program-enrollment -> reoffense estimand, with ACEs-derived psychosocial
variables as confounders.
"""

from __future__ import annotations

import dowhy


def build_causal_model(df, treatment: str, outcome: str, dag_path: str):
    raise NotImplementedError
