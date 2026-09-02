import numpy as np
import pandas as pd
import pytest

from models.classifier.train import train_baseline


def _fixture_df(n: int = 200, jurisdiction: str = "US", random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    prior_convictions = rng.integers(0, 10, size=n)
    age = rng.integers(18, 70, size=n)
    # outcome correlated with prior_convictions so the model has signal to learn
    p_recidivate = 1 / (1 + np.exp(-(0.3 * prior_convictions - 2)))
    recidivated = rng.random(n) < p_recidivate

    df = pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(n)],
            "source": "test_fixture",
            "jurisdiction": jurisdiction,
            "age": age,
            "sex": rng.choice(["Male", "Female"], size=n),
            "prior_convictions": prior_convictions,
            "offense_type": rng.choice(["property", "violent", "drug"], size=n),
            "supervision_intensity": rng.choice(["low", "medium", "high"], size=n),
            "recidivated": recidivated,
        }
    )
    if jurisdiction == "US":
        df["race_ethnicity"] = rng.choice(["White", "Black", "Hispanic", "Other"], size=n)
    return df


@pytest.mark.parametrize("model", ["xgboost", "logistic_regression"])
def test_train_baseline_runs_and_returns_metrics(model):
    df = _fixture_df()
    result = train_baseline(df, model=model, protected_attributes=["race_ethnicity", "sex"])

    assert len(result.X_test) == len(result.y_test) == len(result.y_pred_proba)
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert "roc_auc" in result.metrics
    assert set(result.protected_test.columns) == {"race_ethnicity", "sex"}


def test_train_baseline_skips_missing_protected_attribute():
    df = _fixture_df(jurisdiction="NL")
    assert "race_ethnicity" not in df.columns

    result = train_baseline(df, protected_attributes=["race_ethnicity", "sex"])

    assert "race_ethnicity" not in result.protected_test.columns
    assert "sex" in result.protected_test.columns


def test_train_baseline_rejects_unknown_model():
    df = _fixture_df()
    with pytest.raises(ValueError, match="Unknown model"):
        train_baseline(df, model="not_a_real_model")
