"""DoWhy causal graph definition for the supervision-intensity /
program-enrollment -> reoffense estimand, with ACEs-derived psychosocial
variables as confounders.

The DAG is specified as a YAML list of common causes (confounders) and
effect modifiers rather than a full DOT/GML graph (see dag_spec.yaml) —
this project's causal question is a single treatment -> single outcome
estimand with a fixed confounder set, not a network with multiple
competing causal paths, so a flat list captures it without pulling in a
graph-description dependency.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from dowhy import CausalModel


def load_dag_spec(dag_path: str) -> dict:
    """Load the {common_causes, effect_modifiers} spec from `dag_path`."""
    spec = yaml.safe_load(Path(dag_path).read_text())
    missing = {"common_causes", "effect_modifiers"} - set(spec)
    if missing:
        raise ValueError(f"DAG spec missing required keys: {sorted(missing)}")
    return spec


def _encode_categorical_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode any object/category-dtype columns in `columns`.

    DoWhy's econml estimator passes common-cause / effect-modifier columns
    straight through to the estimator's nuisance models (model_y/model_t),
    with no preprocessing of its own — a raw string column reaches
    scikit-learn and errors out. Numeric columns pass through unchanged.
    Returns the (possibly widened) dataframe and the updated column list.
    """
    categorical = [c for c in columns if df[c].dtype.name in ("object", "category")]
    if not categorical:
        return df, columns

    encoded = pd.get_dummies(df, columns=categorical, prefix=categorical)
    new_dummy_columns = [c for c in encoded.columns if c not in df.columns]
    updated_columns = [c for c in columns if c not in categorical] + new_dummy_columns
    return encoded, updated_columns


def build_causal_model(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    dag_path: str,
) -> CausalModel:
    """Build a DoWhy CausalModel for `treatment -> outcome`, with the
    confounders and effect modifiers declared in `dag_path`.

    Categorical common causes / effect modifiers are one-hot encoded first
    (see `_encode_categorical_columns`); `treatment` is left as-is since
    EconML's discrete-treatment estimators encode it internally.

    Unconfoundedness (no unmeasured common cause of treatment and outcome
    beyond what's listed) is an assumption this makes, not something it
    proves — see models/causal/refutation.py for the closest thing to
    evidence for it.
    """
    spec = load_dag_spec(dag_path)
    common_causes = spec["common_causes"]
    effect_modifiers = spec["effect_modifiers"]

    df, common_causes = _encode_categorical_columns(df, common_causes)
    df, effect_modifiers = _encode_categorical_columns(df, effect_modifiers)

    return CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes,
        effect_modifiers=effect_modifiers,
    )
