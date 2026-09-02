"""DoWhy refutation tests for the CausalForestDML estimate: placebo
treatment, random common cause, and data-subset refuters.

Unconfoundedness is a stated assumption, not a proven property — these
tests are the closest thing to evidence for it, and every causal estimate
must ship with its refutation results attached.
"""

from __future__ import annotations


def run_refuters(causal_model, estimate, refuters: list[str]) -> dict:
    raise NotImplementedError
