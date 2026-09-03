import pandas as pd
import pytest

from data.loaders.aces_simulated import (
    ACE_CATEGORIES,
    COUNTRY_CATEGORIES,
    RACE_CATEGORIES,
    generate_aces_simulated,
)
from data.schema import validate_aces_schema


def test_generated_dataframe_matches_schema():
    df = generate_aces_simulated(500, random_state=42)
    validate_aces_schema(df)
    assert len(df) == 500


def test_all_rows_labeled_synthetic():
    df = generate_aces_simulated(200, random_state=1)
    assert (df["source"] == "aces_simulated").all()


def test_ace_score_bounds_and_flags_consistent():
    df = generate_aces_simulated(500, random_state=2)
    assert df["ace_score"].between(0, len(ACE_CATEGORIES)).all()
    assert (df["adversity_flags"].apply(len) == df["ace_score"]).all()
    all_flags = {flag for flags in df["adversity_flags"] for flag in flags}
    assert all_flags <= set(ACE_CATEGORIES)


def test_reproducible_with_same_seed():
    df1 = generate_aces_simulated(300, random_state=7)
    df2 = generate_aces_simulated(300, random_state=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seed_gives_different_data():
    df1 = generate_aces_simulated(300, random_state=7)
    df2 = generate_aces_simulated(300, random_state=8)
    assert df1["ace_score"].tolist() != df2["ace_score"].tolist()


def test_us_jurisdiction_is_default_and_uses_race_strata():
    df = generate_aces_simulated(300, random_state=3)
    assert (df["jurisdiction"] == "US").all()
    strata = {s.rsplit("_", 1)[0] for s in df["demographic_stratum"]}
    assert strata <= set(RACE_CATEGORIES)


def test_nl_jurisdiction_uses_country_of_birth_strata():
    df = generate_aces_simulated(300, random_state=3, jurisdiction="NL")
    assert (df["jurisdiction"] == "NL").all()
    validate_aces_schema(df)
    strata = {s.rsplit("_", 1)[0] for s in df["demographic_stratum"]}
    assert strata <= set(COUNTRY_CATEGORIES)
    # NL strata are country of birth, never a race category
    assert not (strata & set(RACE_CATEGORIES))


def test_unknown_jurisdiction_raises():
    with pytest.raises(ValueError, match="Unknown jurisdiction"):
        generate_aces_simulated(10, jurisdiction="DE")
