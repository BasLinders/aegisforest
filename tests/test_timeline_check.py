import pandas as pd
import pytest

from models.nli.timeline_check import RESULT_COLUMNS, check_timeline_consistency

# statement_id -> entities as (text, label) pairs, keyed to keep the fake NER
# deterministic and cheap (no real spaCy model needed).
_FAKE_ENTITIES: dict[str, list[tuple[str, str]]] = {
    "t1": [("Amsterdam", "GPE"), ("Friday", "DATE"), ("John", "PERSON")],
    "t2": [("Rotterdam", "GPE"), ("Friday", "DATE")],  # GPE disjoint, DATE overlaps
    "t3": [("Amsterdam", "GPE")],  # GPE overlaps with t1
    "t4": [("some unrelated text", "ORG")],  # no entity types this module cares about
    "u1": [("Utrecht", "GPE")],
}


def _fake_ner(text: str) -> list[tuple[str, str]]:
    for statement_id, entities in _FAKE_ENTITIES.items():
        if text == statement_id:
            return entities
    return []


def _statements() -> pd.DataFrame:
    # `text` doubles as the lookup key into _FAKE_ENTITIES for this fixture
    return pd.DataFrame(
        [
            {"subject_id": "s1", "session_id": "session1", "statement_id": "t1", "text": "t1"},
            {"subject_id": "s1", "session_id": "session2", "statement_id": "t2", "text": "t2"},
            {"subject_id": "s1", "session_id": "session2", "statement_id": "t3", "text": "t3"},
            {"subject_id": "s1", "session_id": "session3", "statement_id": "t4", "text": "t4"},
            {"subject_id": "s2", "session_id": "session1", "statement_id": "u1", "text": "u1"},
        ]
    )


def test_disjoint_entity_type_is_flagged():
    result = check_timeline_consistency(_statements(), ner_fn=_fake_ner)
    row = result[
        (result["statement_id_a"] == "t1") & (result["statement_id_b"] == "t2") & (result["entity_type"] == "GPE")
    ].iloc[0]
    assert row["flagged"]
    assert row["entities_a"] == ["amsterdam"]
    assert row["entities_b"] == ["rotterdam"]


def test_overlapping_entity_type_is_not_flagged():
    result = check_timeline_consistency(_statements(), ner_fn=_fake_ner)
    row = result[
        (result["statement_id_a"] == "t1") & (result["statement_id_b"] == "t2") & (result["entity_type"] == "DATE")
    ].iloc[0]
    assert not row["flagged"]


def test_entity_type_skipped_when_only_one_side_has_it():
    result = check_timeline_consistency(_statements(), ner_fn=_fake_ner)
    # t1 has PERSON ("John"), t2 has none -> no PERSON row for this pair
    pair_rows = result[(result["statement_id_a"] == "t1") & (result["statement_id_b"] == "t2")]
    assert "PERSON" not in set(pair_rows["entity_type"])


def test_pair_with_no_relevant_entities_produces_no_rows():
    result = check_timeline_consistency(_statements(), ner_fn=_fake_ner)
    # t4 only has an ORG entity, which isn't in ENTITY_TYPES
    pair_rows = result[(result["statement_id_a"] == "t3") & (result["statement_id_b"] == "t4")]
    assert pair_rows.empty


def test_only_cross_session_same_subject_pairs_considered():
    result = check_timeline_consistency(_statements(), ner_fn=_fake_ner)
    assert set(result["subject_id"]) == {"s1"}  # s2 has only one statement
    # t1/t3 both in session2's counterpart... verify no same-session pair (t2, t3) appears
    same_session = result[(result["statement_id_a"] == "t2") & (result["statement_id_b"] == "t3")]
    assert same_session.empty


def test_no_pairs_returns_empty_with_correct_columns():
    single = pd.DataFrame(
        [{"subject_id": "s1", "session_id": "session1", "statement_id": "t1", "text": "t1"}]
    )
    result = check_timeline_consistency(single, ner_fn=_fake_ner)
    assert result.empty
    assert list(result.columns) == RESULT_COLUMNS


def test_ner_called_once_per_statement_not_per_pair():
    calls = []

    def counting_ner(text: str) -> list[tuple[str, str]]:
        calls.append(text)
        return _fake_ner(text)

    check_timeline_consistency(_statements(), ner_fn=counting_ner)
    # t1..t4 (s1's statements) each appear in multiple pairs but should
    # only be NER'd once each. u1 (s2's only statement) never appears in
    # any pair at all, so it's never processed.
    assert len(calls) == len(set(calls)) == 4
    assert "u1" not in calls


def test_missing_required_column_raises():
    statements = _statements().drop(columns=["session_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        check_timeline_consistency(statements, ner_fn=_fake_ner)
