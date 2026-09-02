import numpy as np
import pandas as pd
import pytest

from models.causal.dag import build_causal_model, load_dag_spec
from models.causal.dml import estimate_treatment_effect
from models.causal.refutation import run_refuters

DAG_SPEC_PATH = "models/causal/dag_spec.yaml"


def _causal_fixture(n: int = 80, random_state: int = 0) -> pd.DataFrame:
    """Synthetic data with a known negative treatment effect (high
    supervision reduces recidivism probability), so the estimate's sign is
    a real assertion, not a smoke test."""
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 70, n)
    prior_convictions = rng.integers(0, 10, n)
    ace_score = rng.integers(0, 10, n)
    offense_type = rng.choice(["property", "violent", "drug"], size=n)

    p_high = 1 / (1 + np.exp(-(0.1 * prior_convictions + 0.1 * ace_score - 1)))
    supervision_intensity = np.where(rng.random(n) < p_high, "high", "low")
    treatment_effect = np.where(supervision_intensity == "high", 1, 0)

    p_recid = 1 / (1 + np.exp(-(0.2 * prior_convictions + 0.15 * ace_score - 0.8 * treatment_effect - 1)))
    recidivated = rng.random(n) < p_recid

    return pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(n)],
            "age": age,
            "prior_convictions": prior_convictions,
            "offense_type": offense_type,
            "ace_score": ace_score,
            "supervision_intensity": supervision_intensity,
            "recidivated": recidivated,
        }
    )


def test_load_dag_spec_has_required_keys():
    spec = load_dag_spec(DAG_SPEC_PATH)
    assert "age" in spec["common_causes"]
    assert "ace_score" in spec["effect_modifiers"]


def test_build_causal_model_one_hot_encodes_categorical_common_causes():
    df = _causal_fixture()
    causal_model = build_causal_model(df, "supervision_intensity", "recidivated", DAG_SPEC_PATH)
    # offense_type (categorical) should have been expanded into dummy columns
    assert not any(c == "offense_type" for c in causal_model._common_causes)
    assert any(c.startswith("offense_type_") for c in causal_model._common_causes)


def test_run_refuters_rejects_unknown_refuter_before_fitting_anything():
    with pytest.raises(ValueError, match="Unknown refuter"):
        run_refuters(causal_model=None, identified_estimand=None, estimate=None, refuters=["not_a_refuter"])


@pytest.mark.slow
def test_estimate_treatment_effect_end_to_end():
    """Full identify -> estimate -> refute pipeline. Deliberately tiny
    (n=80, n_estimators=4) and limited to one refuter — CausalForestDML
    refits are inherently slow (~1 min each even at this scale), so this
    is the one test in the suite allowed to take a while."""
    df = _causal_fixture()
    result = estimate_treatment_effect(
        df,
        treatment="supervision_intensity",
        outcome="recidivated",
        dag_path=DAG_SPEC_PATH,
        refuters=["random_common_cause"],
        control_value="low",
        treatment_value="high",
        n_estimators=4,
        random_state=42,
    )

    assert result.ate < 0  # matches the fixture's known negative effect
    assert "random_common_cause" in result.refutation_results
    refutation = result.refutation_results["random_common_cause"]
    assert set(refutation) == {"refutation_type", "estimated_effect", "new_effect", "p_value"}
