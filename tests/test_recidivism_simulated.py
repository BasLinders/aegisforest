import pandas as pd

from data.loaders.recidivism_simulated import generate_recidivism_simulated
from data.schema import validate_recidivism_schema


def test_generated_dataframe_matches_schema():
    df = generate_recidivism_simulated(300, random_state=1)
    validate_recidivism_schema(df)
    assert len(df) == 300


def test_all_rows_labeled_synthetic():
    df = generate_recidivism_simulated(200, random_state=2)
    assert (df["source"] == "recidivism_simulated").all()


def test_ace_score_column_present_and_in_range():
    df = generate_recidivism_simulated(300, random_state=3)
    assert "ace_score" in df.columns
    assert df["ace_score"].between(0, 10).all()


def test_us_jurisdiction_populates_race_ethnicity_not_country_of_birth():
    df = generate_recidivism_simulated(300, random_state=4, jurisdiction="US")
    assert "race_ethnicity" in df.columns
    assert "country_of_birth" not in df.columns
    assert df["race_ethnicity"].notna().all()


def test_nl_jurisdiction_populates_country_of_birth_not_race_ethnicity():
    df = generate_recidivism_simulated(300, random_state=4, jurisdiction="NL")
    assert "country_of_birth" in df.columns
    assert "race_ethnicity" not in df.columns
    assert df["country_of_birth"].notna().all()


def test_higher_supervision_correlates_with_lower_recidivism():
    df = generate_recidivism_simulated(4000, random_state=5)
    rates = df.groupby("supervision_intensity", observed=True)["recidivated"].mean()
    assert rates["high"] < rates["low"]


def test_reproducible_with_same_seed():
    df1 = generate_recidivism_simulated(300, random_state=7)
    df2 = generate_recidivism_simulated(300, random_state=7)
    pd.testing.assert_frame_equal(df1, df2)
