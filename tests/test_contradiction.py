import pandas as pd
import pytest

from models.nli.contradiction import RESULT_COLUMNS, score_contradictions


def _fixture_statements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # subject s1: two statements in session 1, one in session 2
            {"subject_id": "s1", "session_id": "session1", "statement_id": "t1", "text": "I was home all night"},
            {"subject_id": "s1", "session_id": "session1", "statement_id": "t1b", "text": "I never left the house"},
            {"subject_id": "s1", "session_id": "session2", "statement_id": "t2", "text": "I was at the bar until 2am"},
            # subject s2: only one statement, no session to pair against
            {"subject_id": "s2", "session_id": "session1", "statement_id": "u1", "text": "I don't recall"},
        ]
    )


def _fake_pipeline(batch: list[dict]) -> list[list[dict]]:
    """Deterministic stand-in for a real NLI model: flags the pair
    involving "I was home all night" / "I was at the bar" as a
    contradiction, everything else as not."""
    outputs = []
    for item in batch:
        is_contradiction = "home all night" in item["text"] and "bar" in item["text_pair"]
        score = 0.92 if is_contradiction else 0.05
        outputs.append(
            [
                {"label": "CONTRADICTION", "score": score},
                {"label": "NEUTRAL", "score": (1 - score) * 0.7},
                {"label": "ENTAILMENT", "score": (1 - score) * 0.3},
            ]
        )
    return outputs


def test_only_cross_session_pairs_for_same_subject_are_scored():
    statements = _fixture_statements()
    result = score_contradictions(statements, nli_pipeline=_fake_pipeline)

    # s1 has 3 statements (2 in session1, 1 in session2) -> 2 valid cross-session
    # pairs; the same-session pair (t1, t1b) must not appear. s2 has only one
    # statement -> contributes zero pairs.
    assert len(result) == 2
    assert set(result["subject_id"]) == {"s1"}
    pairs = set(zip(result["statement_id_a"], result["statement_id_b"]))
    assert pairs == {("t1", "t2"), ("t1b", "t2")}


def test_flagged_reflects_threshold():
    statements = _fixture_statements()
    result = score_contradictions(statements, nli_pipeline=_fake_pipeline, contradiction_threshold=0.5)

    flagged_row = result[(result["statement_id_a"] == "t1") & (result["statement_id_b"] == "t2")].iloc[0]
    unflagged_row = result[(result["statement_id_a"] == "t1b") & (result["statement_id_b"] == "t2")].iloc[0]

    assert flagged_row["flagged"]
    assert flagged_row["contradiction_score"] == pytest.approx(0.92)
    assert not unflagged_row["flagged"]


def test_no_cross_session_pairs_returns_empty_with_correct_columns():
    statements = pd.DataFrame(
        [{"subject_id": "s1", "session_id": "session1", "statement_id": "t1", "text": "hello"}]
    )
    result = score_contradictions(statements, nli_pipeline=_fake_pipeline)
    assert result.empty
    assert list(result.columns) == RESULT_COLUMNS


def test_missing_required_column_raises():
    statements = _fixture_statements().drop(columns=["session_id"])
    with pytest.raises(ValueError, match="missing required columns"):
        score_contradictions(statements, nli_pipeline=_fake_pipeline)


def test_missing_contradiction_label_raises():
    def broken_pipeline(batch):
        return [[{"label": "NEUTRAL", "score": 1.0}] for _ in batch]

    with pytest.raises(ValueError, match="No contradiction label found"):
        score_contradictions(_fixture_statements(), nli_pipeline=broken_pipeline)
