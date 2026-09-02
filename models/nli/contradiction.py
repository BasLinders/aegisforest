"""Pairwise NLI contradiction scoring across statement chunks from the same
subject across interview sessions (RoBERTa-large-MNLI or similar).

Output is a contradiction-likelihood score per sentence pair. This must
always be surfaced as "flagged for human review" — never as a
deception/guilt signal. That framing is enforced at the output-template
level (`reports/templates/`), not just in prose.
"""

from __future__ import annotations

import pandas as pd


def score_contradictions(
    statements: pd.DataFrame,
    checkpoint: str = "roberta-large-mnli",
    device: str = "auto",
) -> pd.DataFrame:
    raise NotImplementedError
