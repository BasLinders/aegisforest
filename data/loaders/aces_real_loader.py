"""Loader for real, de-identified ACEs data (CDC-Kaiser ACE Study public-use
files, or YRBSS state-level adversity/justice-contact data).

Returns a dataframe conforming to `data.schema.ACES_SCHEMA` with
`source="aces_real"`.
"""

from __future__ import annotations

import pandas as pd

from data.schema import validate_aces_schema


def load_aces_real(path: str) -> pd.DataFrame:
    raise NotImplementedError
