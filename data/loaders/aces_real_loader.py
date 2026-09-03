"""Loader for real, de-identified ACEs data.

Unlike `aces_simulated.py`, which can switch jurisdiction with a single
parameter (it's generating from a hand-set table either way), the real
data source is a genuinely different dataset per jurisdiction:

- US: CDC-Kaiser ACE Study public-use files, or YRBSS state-level
  adversity/justice-contact data.
- NL: no direct ACEs-survey equivalent exists; the nearest real source
  would be WODC/CBS adversity- and poverty-adjacent statistics by
  country of birth (see `aces_simulated.py`'s module docstring for the
  specific CBS reports this project already cites for the synthetic NL
  table). Loading real NL data here means picking one of those and
  reshaping it to `ACES_SCHEMA`, not just passing `jurisdiction="NL"`.

Returns a dataframe conforming to `data.schema.ACES_SCHEMA` (including
`jurisdiction`) with `source="aces_real"`.
"""

from __future__ import annotations

import pandas as pd

from data.schema import validate_aces_schema


def load_aces_real(path: str, jurisdiction: str = "US") -> pd.DataFrame:
    raise NotImplementedError
