"""Fairness audit: calibration + false-positive-rate parity across
demographic subgroups, replicating the ProPublica COMPAS critique as a
sanity check.

Runs on both Module A layers per the handoff constraints: the baseline
classifier's predictions, and separately the CATE outputs from
`models/causal/dml.py` (to check whether adding psychosocial confounders
shifts fairness metrics / introduces proxy discrimination).
"""

from __future__ import annotations

import pandas as pd


def audit_calibration(df: pd.DataFrame, protected_attributes: list[str]) -> pd.DataFrame:
    raise NotImplementedError


def audit_fpr_parity(df: pd.DataFrame, protected_attributes: list[str]) -> pd.DataFrame:
    raise NotImplementedError
