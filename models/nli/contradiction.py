"""Pairwise NLI contradiction scoring across statement chunks from the same
subject across interview sessions.

Output is a contradiction-likelihood score per sentence pair. This must
always be surfaced as "flagged for human review" — never as a
deception/guilt signal. `flagged` is the only field any output template
may treat as meaningful; that framing is enforced at the
output-template level (`reports/templates/`), not just in prose.

Jurisdiction: this module itself doesn't care what language the
statements are in — it just calls whatever `checkpoint` config gives it.
The default, `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`,
was chosen specifically because it's fine-tuned on ~105k Dutch NLI pairs
(not relying on cross-lingual transfer from English) alongside English
MNLI/XNLI, so it covers both US (English) and NL (Dutch) statements with
one checkpoint — no jurisdiction-conditional swap needed. It uses the
same entailment/neutral/contradiction label scheme as the
English-only RoBERTa-large-MNLI checkpoint this replaced, so
`_contradiction_score`'s label matching didn't need to change.

Benchmarked against real Dutch NLI data (SICK-NL, via
scripts/benchmark_nli_checkpoint.py): 67.6% accuracy / 65.6% macro-F1 on
a 500-pair sample, with high recall on real contradictions (90.5% at the
default threshold) but notably lower precision (43.5% — over-flags
neutral pairs). The threshold sweep in that script shows raising
`contradiction_threshold` trades that recall away for precision — but
config deliberately keeps it at 0.5: a missed real contradiction is
worse than an extra reviewer look for this use case, and `flagged`
already means "send to review," never a verdict, so the cost of
over-flagging is bounded. See BUILD_PLAN.md's open questions for the
full breakdown.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import pandas as pd

from data.schema import validate_statements_schema

RESULT_COLUMNS = [
    "subject_id",
    "session_id_a",
    "statement_id_a",
    "text_a",
    "session_id_b",
    "statement_id_b",
    "text_b",
    "contradiction_score",
    "flagged",
]

NliPipeline = Callable[[list[dict[str, str]]], list[list[dict[str, Any]]]]


def _resolve_device(device: str) -> int:
    """Map config's device setting to the integer transformers.pipeline
    expects: -1 for CPU, 0 for the first CUDA device."""
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    import torch

    return 0 if torch.cuda.is_available() else -1


def _build_pipeline(checkpoint: str, device: str) -> NliPipeline:
    from transformers import pipeline

    return pipeline("text-classification", model=checkpoint, top_k=None, device=_resolve_device(device))


def _contradiction_score(label_scores: list[dict[str, Any]]) -> float:
    for item in label_scores:
        if "contra" in item["label"].lower():
            return float(item["score"])
    raise ValueError(f"No contradiction label found in pipeline output: {label_scores}")


def _cross_session_pairs(statements: pd.DataFrame):
    """Yield (row_a, row_b) for every pair of statements from the same
    subject but different sessions. Same-session pairs are excluded: the
    point is inconsistency across interviews, not within one."""
    for _, group in statements.groupby("subject_id"):
        for row_a, row_b in combinations(group.itertuples(), 2):
            if row_a.session_id != row_b.session_id:
                yield row_a, row_b


def score_contradictions(
    statements: pd.DataFrame,
    checkpoint: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    device: str = "auto",
    contradiction_threshold: float = 0.5,
    nli_pipeline: NliPipeline | None = None,
) -> pd.DataFrame:
    """Score every cross-session statement pair for the same subject.

    `nli_pipeline`, if given, replaces the default checkpoint-loading
    pipeline — lets callers (including tests) inject a lightweight or fake
    pipeline instead of downloading the default checkpoint (~1.1GB) every
    run. It must behave like `transformers.pipeline("text-classification",
    top_k=None)`: called with a list of `{"text", "text_pair"}` dicts,
    returning one list of `{"label", "score"}` dicts per input.

    Returns one row per pair (`RESULT_COLUMNS`): subject_id,
    session_id_a/b, statement_id_a/b, text_a/b, contradiction_score, and
    `flagged` (score >= contradiction_threshold) — the only field any
    downstream report may treat as meaningful. `contradiction_score` on
    its own is not a deception signal.
    """
    validate_statements_schema(statements)

    pairs = list(_cross_session_pairs(statements))
    if not pairs:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    pipe = nli_pipeline or _build_pipeline(checkpoint, device)
    batch = [{"text": a.text, "text_pair": b.text} for a, b in pairs]
    outputs = pipe(batch)

    rows = []
    for (row_a, row_b), label_scores in zip(pairs, outputs):
        score = _contradiction_score(label_scores)
        rows.append(
            {
                "subject_id": row_a.subject_id,
                "session_id_a": row_a.session_id,
                "statement_id_a": row_a.statement_id,
                "text_a": row_a.text,
                "session_id_b": row_b.session_id,
                "statement_id_b": row_b.statement_id,
                "text_b": row_b.text,
                "contradiction_score": score,
                "flagged": score >= contradiction_threshold,
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
