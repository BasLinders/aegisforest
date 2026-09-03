"""Optional extension: lightweight NER (spaCy) + timeline/entity consistency
checks layered on top of the NLI contradiction scores.

Off by default (config's module_b.timeline_check.enabled). Where
contradiction.py asks "do these two statements logically contradict each
other," this asks a narrower, more mechanical question: across a
subject's cross-session statements, do the named locations, dates/times,
and people mentioned overlap at all, or are they completely disjoint for
a given entity type? Two entirely disjoint sets isn't proof of anything —
statements can legitimately describe different times or places — so this
is a "worth a second look" signal, same as contradiction.py's `flagged`,
never a verdict.

Jurisdiction: unlike contradiction.py's NLI checkpoint, there's no single
multilingual spaCy NER model of comparable quality to the per-language
"sm" models, so `ner_model` genuinely needs to be swapped per
jurisdiction: `en_core_web_sm` for US (English) statements,
`nl_core_news_sm` for NL (Dutch). Both were verified to share the same
entity label scheme (GPE, LOC, DATE, TIME, PERSON, ...), so the
comparison logic here doesn't change between them — only which model
config loads does. Both are lightweight ("sm") models and will miss
entities a larger model would catch (e.g. "vrijdag"/Friday not always
tagged DATE in nl_core_news_sm) — that's a real limitation, not just a
caveat: a missed entity looks identical to "nothing to compare," so this
check under-reports rather than over-reports on weak NER output.

No negation handling: "I was at the bar in Chicago" and "I never went to
Chicago" both mention Chicago as a GPE entity, so plain set-overlap does
NOT flag them — despite this being a genuine, easy contradiction (and
one contradiction.py's NLI scoring does catch). This isn't a bug to fix
here; it's why this module exists alongside NLI scoring rather than
instead of it, and why its output should be read together with
contradiction.py's, not as a substitute.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from data.schema import validate_statements_schema
from models.nli.contradiction import cross_session_pairs

RESULT_COLUMNS = [
    "subject_id",
    "session_id_a",
    "statement_id_a",
    "text_a",
    "session_id_b",
    "statement_id_b",
    "text_b",
    "entity_type",
    "entities_a",
    "entities_b",
    "flagged",
]

# The entity types actually relevant to a "timeline" check — spaCy's other
# types (MONEY, LAW, WORK_OF_ART, ...) aren't about where/when/who.
ENTITY_TYPES = ("GPE", "LOC", "DATE", "TIME", "PERSON")

NerFn = Callable[[str], list[tuple[str, str]]]


def _build_ner(ner_model: str) -> NerFn:
    import spacy

    nlp = spacy.load(ner_model)

    def ner(text: str) -> list[tuple[str, str]]:
        return [(ent.text, ent.label_) for ent in nlp(text).ents]

    return ner


def _entities_by_type(entities: list[tuple[str, str]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {entity_type: set() for entity_type in ENTITY_TYPES}
    for text, label in entities:
        if label in grouped:
            grouped[label].add(text.strip().lower())
    return grouped


def check_timeline_consistency(
    statements: pd.DataFrame,
    ner_model: str = "en_core_web_sm",
    ner_fn: NerFn | None = None,
) -> pd.DataFrame:
    """Compare named entities across every cross-session statement pair
    for the same subject.

    `ner_fn`, if given, replaces the default spaCy-model-loading NER —
    lets callers (including tests) inject a lightweight or fake function
    instead of downloading `ner_model`. It must behave like
    `lambda text: [(entity_text, entity_label), ...]`.

    Returns one row per (pair, entity_type) where *both* statements
    mention at least one entity of that type (`RESULT_COLUMNS`):
    subject_id, session_id_a/b, statement_id_a/b, text_a/b, entity_type,
    entities_a/b (sorted lists of the lowercased entity text), and
    `flagged` (the two entity sets are completely disjoint) — the only
    field any downstream report may treat as meaningful, same as
    contradiction.py's `flagged`.
    """
    validate_statements_schema(statements)

    pairs = list(cross_session_pairs(statements))
    if not pairs:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    extract = ner_fn or _build_ner(ner_model)

    entity_cache: dict[str, dict[str, set[str]]] = {}

    def entities_for(row) -> dict[str, set[str]]:
        if row.statement_id not in entity_cache:
            entity_cache[row.statement_id] = _entities_by_type(extract(row.text))
        return entity_cache[row.statement_id]

    rows = []
    for row_a, row_b in pairs:
        ents_a, ents_b = entities_for(row_a), entities_for(row_b)
        for entity_type in ENTITY_TYPES:
            set_a, set_b = ents_a[entity_type], ents_b[entity_type]
            if not set_a or not set_b:
                continue  # nothing to compare for this entity type
            rows.append(
                {
                    "subject_id": row_a.subject_id,
                    "session_id_a": row_a.session_id,
                    "statement_id_a": row_a.statement_id,
                    "text_a": row_a.text,
                    "session_id_b": row_b.session_id,
                    "statement_id_b": row_b.statement_id,
                    "text_b": row_b.text,
                    "entity_type": entity_type,
                    "entities_a": sorted(set_a),
                    "entities_b": sorted(set_b),
                    "flagged": set_a.isdisjoint(set_b),
                }
            )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
