"""Synthetic psychosocial (ACEs) data generator, built from published
population-level ACEs prevalence rates by demographic stratum (CDC BRFSS ACE
module).

Has no external data dependency, so it unblocks pipeline development ahead
of `aces_real_loader`. Returns a dataframe conforming to
`data.schema.ACES_SCHEMA` with `source="aces_simulated"` — every row is
clearly labeled as synthetic, and that label must be propagated into every
downstream report.
"""

from __future__ import annotations

import pandas as pd


def generate_aces_simulated(n: int, random_state: int = 42) -> pd.DataFrame:
    raise NotImplementedError
