import pandas as pd
import pytest

from data.schema import ACES_SCHEMA, validate_aces_schema


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["s1"],
            "source": ["aces_real"],
            "demographic_stratum": ["a"],
            "ace_score": [3],
            "adversity_flags": [["poverty"]],
        }
    )


def test_valid_dataframe_passes():
    validate_aces_schema(_valid_df())


def test_missing_column_raises():
    df = _valid_df().drop(columns=["ace_score"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_aces_schema(df)


def test_unknown_source_raises():
    df = _valid_df()
    df["source"] = "made_up_source"
    with pytest.raises(ValueError, match="Unknown ACEs source"):
        validate_aces_schema(df)
