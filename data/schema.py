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
