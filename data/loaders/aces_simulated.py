"""Synthetic psychosocial (ACEs) data generator, built from published
population-level ACEs prevalence rates by demographic stratum (CDC BRFSS ACE
module).

Has no external data dependency, so it unblocks pipeline development ahead
of `aces_real_loader`. Returns a dataframe conforming to
`data.schema.ACES_SCHEMA` with `source="aces_simulated"` — every row is
clearly labeled as synthetic, and that label must be propagated into every
downstream report.

The ten adversity categories are the original Felitti et al. / CDC-Kaiser
ACE Study categories. Per-stratum mean ACE scores below are order-of-
magnitude approximations of published CDC BRFSS ACE module summaries
(national mean ~1.6 ACEs; ~16% of adults report 4+; women and some
race/ethnicity subgroups report modestly higher exposure in state-level
BRFSS reports) — they are placeholders to give the synthetic layer
realistic heterogeneity, not a literature-calibrated table. Replace with
cited per-subgroup figures before treating any downstream result as more
than illustrative (see BUILD_PLAN.md open questions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.schema import validate_aces_schema

ACE_CATEGORIES: tuple[str, ...] = (
    "emotional_abuse",
    "physical_abuse",
    "sexual_abuse",
    "emotional_neglect",
    "physical_neglect",
    "household_intimate_partner_violence",
    "household_substance_abuse",
    "household_mental_illness",
    "parental_separation_or_divorce",
    "household_member_incarcerated",
)

RACE_CATEGORIES: tuple[str, ...] = ("White", "Black", "Hispanic", "Asian", "Other")
COUNTRY_CATEGORIES: tuple[str, ...] = ("Türkiye", "Morocco", "Suriname", "Dutch Caribbean Islands", "Indonesia") # Specifically recorded in the Dutch criminal justice system
SEX_CATEGORIES: tuple[str, ...] = ("Male", "Female")

# stratum -> mean ACE score (of 10). See module docstring caveat.
_STRATUM_MEAN_ACE_SCORE: dict[tuple[str, str], float] = {
    ("White", "Male"): 1.4,
    ("White", "Female"): 1.6,
    ("Black", "Male"): 1.7,
    ("Black", "Female"): 1.9,
    ("Hispanic", "Male"): 1.6,
    ("Hispanic", "Female"): 1.8,
    ("Other", "Male"): 1.5,
    ("Other", "Female"): 1.7,
}

# Country categories need their own scores. These have yet to be determined (Centraal Bureau voor de Statistiek scores?). 


def _sample_ace_scores(strata: list[tuple[str, str]], rng: np.random.Generator) -> np.ndarray:
    """Draw an overdispersed ACE score (0-10) per row via a stratum-specific
    negative binomial, matching the right-skewed shape of published ACE
    score distributions better than a Poisson would."""
    dispersion = 2.0  # negative-binomial shape parameter (r); lower = more overdispersed
    scores = np.empty(len(strata), dtype=np.int64)
    for stratum, mean in _STRATUM_MEAN_ACE_SCORE.items():
        mask = np.array([s == stratum for s in strata], dtype=bool)
        n_stratum = int(mask.sum())
        if n_stratum == 0:
            continue
        p = dispersion / (dispersion + mean)
        draws = rng.negative_binomial(dispersion, p, size=n_stratum)
        scores[mask] = np.clip(draws, 0, len(ACE_CATEGORIES))
    return scores


def generate_aces_simulated(n: int, random_state: int = 42) -> pd.DataFrame:
    """Generate `n` synthetic ACEs records.

    Each row gets a demographic stratum (race x sex), an ACE score sampled
    from that stratum's approximate published mean, and a set of specific
    adversity categories consistent with that score.
    """
    rng = np.random.default_rng(random_state)

    stratum_keys = list(_STRATUM_MEAN_ACE_SCORE.keys())
    stratum_idx = rng.integers(0, len(stratum_keys), size=n)
    strata = [stratum_keys[i] for i in stratum_idx]

    ace_scores = _sample_ace_scores(strata, rng)

    adversity_flags = [
        sorted(rng.choice(ACE_CATEGORIES, size=score, replace=False).tolist())
        for score in ace_scores
    ]

    df = pd.DataFrame(
        {
            "subject_id": [f"sim-{i:07d}" for i in range(n)],
            "source": "aces_simulated",
            "demographic_stratum": [f"{race}_{sex}" for race, sex in strata],
            "ace_score": ace_scores,
            "adversity_flags": adversity_flags,
        }
    )
    df["source"] = df["source"].astype("category")
    df["demographic_stratum"] = df["demographic_stratum"].astype("category")

    validate_aces_schema(df)
    return df
