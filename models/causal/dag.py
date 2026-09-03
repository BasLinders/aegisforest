"""DoWhy causal graph definition for the supervision-intensity /
program-enrollment -> reoffense estimand, with ACEs-derived psychosocial
variables as confounders.

The DAG is specified as a YAML list of common causes (confounders) and
effect modifiers rather than a full DOT/GML graph (see dag_spec.yaml) —
this project's causal question is a single treatment -> single outcome
estimand with a fixed confounder set, not a network with multiple
competing causal paths, so a flat list captures it without pulling in a
graph-description dependency.

Jurisdiction-agnostic by construction: dag_spec.yaml's confounders (age,
prior_convictions, offense_type) and effect modifier (ace_score) are all
plain RECIDIVISM_SCHEMA/ACES_SCHEMA columns with no US- or NL-specific
category baked in here — unlike the classifier's protected-attribute
handling, this module never touches race_ethnicity or demographic_stratum
at all.
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
    """One-hot encode any string/category-dtype columns in `columns`.

    DoWhy's econml estimator passes common-cause / effect-modifier columns
    straight through to the estimator's nuisance models (model_y/model_t),
    with no preprocessing of its own — a raw string column reaches
    scikit-learn and errors out. Numeric columns pass through unchanged.

    Uses pandas' is_string_dtype/is_categorical_dtype rather than matching
    dtype names directly: pandas 3.0 introduced a native `str` dtype for
    text columns (distinct from the legacy `object` dtype), so a literal
    `dtype.name in ("object", "category")` check silently misses string
    columns on pandas >=3.0.

    Returns the (possibly widened) dataframe and the updated column list.
    """
    categorical = [
        c for c in columns
        if pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
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
    (see `_encode_categorical_columns`); `treatment` is left unencoded
    since EconML's discrete-treatment estimators encode it internally —
    but it IS cast to pandas' `category` dtype if it's a string/object
    column and not category already. dowhy 0.14's placebo_treatment_refuter
    (the default, non-"permute" path) dispatches on
    `type_dict[treatment_names[0]].name` and only has branches for
    float/bool/int/category — a plain string-dtype treatment column hits
    none of them and raises `UnboundLocalError: cannot access local
    variable 'new_treatment'`, which is a genuinely confusing failure mode
    for something as simple as "the treatment column wasn't cast to
    category." Casting it here means callers don't need to know this.

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

    if pd.api.types.is_string_dtype(df[treatment]) and not isinstance(df[treatment].dtype, pd.CategoricalDtype):
        df = df.copy()
        df[treatment] = df[treatment].astype("category")

    return CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes,
        effect_modifiers=effect_modifiers,
    )
