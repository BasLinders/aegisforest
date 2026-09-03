"""Synthetic psychosocial (ACEs) data generator, built from published
population-level adversity indicators by demographic stratum.

Has no external data dependency, so it unblocks pipeline development ahead
of `aces_real_loader`. Returns a dataframe conforming to
`data.schema.ACES_SCHEMA` with `source="aces_simulated"` — every row is
clearly labeled as synthetic, and that label must be propagated into every
downstream report.

The ten adversity categories are the original Felitti et al. / CDC-Kaiser
ACE Study categories, used for both jurisdictions below — there's no
Dutch equivalent inventory, so this project's ACE category list is
US-sourced regardless of which stratification table is used.

Two jurisdictions, two different stratification schemes:

- `jurisdiction="US"` strata by race x sex. Per-stratum mean ACE scores
  are order-of-magnitude approximations of published CDC BRFSS ACE module
  summaries (national mean ~1.6 ACEs; ~16% of adults report 4+; women and
  some race/ethnicity subgroups report modestly higher exposure in
  state-level BRFSS reports).

- `jurisdiction="NL"` strata by country of birth x sex, not race: Dutch
  criminal-justice registries generally don't record race (see
  RECIDIVISM_SCHEMA's race_ethnicity caveat), but WODC/CBS crime and
  justice statistics do report by migration background / country of
  birth (a subject's own or, for the Netherlands-born, their parents').
  There is no Dutch ACE survey to anchor this table to, so it uses a
  cross-domain proxy instead: CBS child-poverty rates by country of
  birth (2nd-generation, since that's the population actually appearing
  in Dutch youth-justice statistics — see WODC's 2025 finding that
  Netherlands-born minors with both parents born abroad are suspects at
  roughly 2.3x the rate of minors with no migration background).
  Poverty is not ACEs, and the mapping from a poverty-rate ratio to a
  mean-ACE-score placeholder is a modeling simplification, not a
  computed transform of the CBS figures — the *direction* (Morocco >
  Turkey ~ Dutch Caribbean > Suriname ~ Indonesia ~ Netherlands-origin
  baseline) follows the CBS data; the exact values don't.

Either way, these are placeholders to give the synthetic layer realistic
heterogeneity, not literature-calibrated tables. Replace with cited
per-subgroup figures before treating any downstream result as more than
illustrative (see BUILD_PLAN.md open questions).
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
SEX_CATEGORIES: tuple[str, ...] = ("Male", "Female")

# Specifically recorded in Dutch criminal-justice / CBS statistics as
# country-of-birth (migratieachtergrond) categories. "Netherlands" is the
# reference/baseline group — CBS always reports these ratios against it,
# and dropping it here would leave the NL table with no baseline stratum.
COUNTRY_CATEGORIES: tuple[str, ...] = (
    "Netherlands",
    "Türkiye",
    "Morocco",
    "Suriname",
    "Dutch Caribbean Islands",
    "Indonesia",
)

# jurisdiction -> {(group, sex): mean ACE score (of 10)}. See module
# docstring for how each table's numbers were derived and their caveats.
_STRATUM_MEAN_ACE_SCORE: dict[str, dict[tuple[str, str], float]] = {
    "US": {
        ("White", "Male"): 1.4,
        ("White", "Female"): 1.6,
        ("Black", "Male"): 1.7,
        ("Black", "Female"): 1.9,
        ("Hispanic", "Male"): 1.6,
        ("Hispanic", "Female"): 1.8,
        # Asian-American subgroup ACE prevalence is thinly reported in state
        # BRFSS modules (small subsample sizes, often suppressed) — this is
        # an even rougher placeholder than the other rows, set near the
        # lower end consistent with what limited BRFSS breakouts exist.
        ("Asian", "Male"): 1.3,
        ("Asian", "Female"): 1.5,
        ("Other", "Male"): 1.5,
        ("Other", "Female"): 1.7,
    },
    "NL": {
        # Baseline/reference group (CBS "Nederlandse herkomst").
        ("Netherlands", "Male"): 1.3,
        ("Netherlands", "Female"): 1.5,
        # ~baseline child-poverty risk per CBS Armoede en sociale uitsluiting
        # 2023 (2nd-generation Surinamese-Dutch: ~1 in 20, close to the
        # ~5% national rate) and CBS's Indonesian-heritage income mobility
        # findings ("virtually equal" long-term poverty risk to Dutch-origin).
        ("Suriname", "Male"): 1.4,
        ("Suriname", "Female"): 1.6,
        ("Indonesia", "Male"): 1.3,
        ("Indonesia", "Female"): 1.5,
        # ~2x baseline child-poverty risk (2nd-generation Turkish-Dutch and
        # Dutch-Caribbean/Antillean children: ~1 in 10, per the same CBS report).
        ("Türkiye", "Male"): 1.7,
        ("Türkiye", "Female"): 1.9,
        ("Dutch Caribbean Islands", "Male"): 1.7,
        ("Dutch Caribbean Islands", "Female"): 1.9,
        # Highest child-poverty risk of the groups CBS reports (2nd-generation
        # Moroccan-Dutch: ~16%, roughly 3x baseline).
        ("Morocco", "Male"): 2.0,
        ("Morocco", "Female"): 2.2,
    },
}


def _sample_ace_scores(
    strata: list[tuple[str, str]], jurisdiction: str, rng: np.random.Generator
) -> np.ndarray:
    """Draw an overdispersed ACE score (0-10) per row via a stratum-specific
    negative binomial, matching the right-skewed shape of published ACE
    score distributions better than a Poisson would."""
    dispersion = 2.0  # negative-binomial shape parameter (r); lower = more overdispersed
    scores = np.empty(len(strata), dtype=np.int64)
    for stratum, mean in _STRATUM_MEAN_ACE_SCORE[jurisdiction].items():
        mask = np.array([s == stratum for s in strata], dtype=bool)
        n_stratum = int(mask.sum())
        if n_stratum == 0:
            continue
        p = dispersion / (dispersion + mean)
        draws = rng.negative_binomial(dispersion, p, size=n_stratum)
        scores[mask] = np.clip(draws, 0, len(ACE_CATEGORIES))
    return scores


def generate_aces_simulated(n: int, random_state: int = 42, jurisdiction: str = "US") -> pd.DataFrame:
    """Generate `n` synthetic ACEs records for `jurisdiction` ("US" or "NL").

    Each row gets a demographic stratum (race x sex for US, country of
    birth x sex for NL — see module docstring), an ACE score sampled from
    that stratum's approximate mean, and a set of specific adversity
    categories consistent with that score.
    """
    if jurisdiction not in _STRATUM_MEAN_ACE_SCORE:
        raise ValueError(
            f"Unknown jurisdiction {jurisdiction!r}; use one of {sorted(_STRATUM_MEAN_ACE_SCORE)}"
        )

    rng = np.random.default_rng(random_state)

    stratum_keys = list(_STRATUM_MEAN_ACE_SCORE[jurisdiction].keys())
    stratum_idx = rng.integers(0, len(stratum_keys), size=n)
    strata = [stratum_keys[i] for i in stratum_idx]

    ace_scores = _sample_ace_scores(strata, jurisdiction, rng)

    adversity_flags = [
        sorted(rng.choice(ACE_CATEGORIES, size=score, replace=False).tolist())
        for score in ace_scores
    ]

    df = pd.DataFrame(
        {
            "subject_id": [f"sim-{i:07d}" for i in range(n)],
            "source": "aces_simulated",
            "jurisdiction": jurisdiction,
            "demographic_stratum": [f"{group}_{sex}" for group, sex in strata],
            "ace_score": ace_scores,
            "adversity_flags": adversity_flags,
        }
    )
    df["source"] = df["source"].astype("category")
    df["jurisdiction"] = df["jurisdiction"].astype("category")
    df["demographic_stratum"] = df["demographic_stratum"].astype("category")

    validate_aces_schema(df)
    return df
