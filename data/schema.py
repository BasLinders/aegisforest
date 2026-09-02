"""Shared schema that every ACEs data source (real or simulated) must satisfy.

Downstream pipeline code is source-agnostic: it reads this schema, not the
loader that produced the dataframe. `source` is the one field that varies by
loader and must be propagated into every report/output artifact so synthetic
results are never mistaken for real-data findings.
"""

from __future__ import annotations

ACES_SOURCES = ("aces_real", "aces_simulated")

# column_name -> pandas dtype string
ACES_SCHEMA: dict[str, str] = {
    "subject_id": "string",
    "source": "category",  # one of ACES_SOURCES, propagated to every output
    "demographic_stratum": "category",
    "ace_score": "int64",
    "adversity_flags": "object",  # list[str] of individual ACE indicators
}


def validate_aces_schema(df) -> None:
    """Raise ValueError if `df` does not conform to ACES_SCHEMA."""
    missing = set(ACES_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"ACEs dataframe missing required columns: {sorted(missing)}")

    bad_sources = set(df["source"].unique()) - set(ACES_SOURCES)
    if bad_sources:
        raise ValueError(f"Unknown ACEs source(s) in data: {sorted(bad_sources)}")


# Schema shared by every recidivism data source (COMPAS, NIJ, WODC Recidivism
# Monitor, ...) so the classifier and causal layer don't branch on which
# loader produced the dataframe. `race_ethnicity` is nullable: not every
# jurisdiction's registries record it — Dutch criminal-justice data
# generally doesn't, since it's treated as special-category data under the
# AVG/GDPR. Fairness/confounder code must handle it being absent for a given
# source rather than assume it's always populated.
RECIDIVISM_SCHEMA: dict[str, str] = {
    "subject_id": "string",
    "source": "category",  # e.g. "compas", "nij", "wodc_recidivism_monitor"
    "jurisdiction": "category",  # e.g. "US", "NL"
    "age": "int64",
    "sex": "category",
    "race_ethnicity": "category",  # nullable — see caveat above
    "prior_convictions": "int64",
    "offense_type": "category",
    "supervision_intensity": "category",  # the DML treatment variable
    "recidivated": "bool",  # outcome
}

REQUIRED_RECIDIVISM_COLUMNS = set(RECIDIVISM_SCHEMA) - {"race_ethnicity"}


def validate_recidivism_schema(df) -> None:
    """Raise ValueError if `df` does not conform to RECIDIVISM_SCHEMA.

    `race_ethnicity` is exempt from the required-columns check (see
    RECIDIVISM_SCHEMA docstring) but if present must still be a valid column.
    """
    missing = REQUIRED_RECIDIVISM_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Recidivism dataframe missing required columns: {sorted(missing)}")
