"""Optional extension: lightweight NER (spaCy) + timeline/entity consistency
checks layered on top of the NLI contradiction scores.
"""

from __future__ import annotations

import pandas as pd


def check_timeline_consistency(
    statements: pd.DataFrame,
    ner_model: str = "en_core_web_sm",
) -> pd.DataFrame:
    raise NotImplementedError
