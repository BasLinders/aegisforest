"""Synthetic recidivism data generator, joined with the ACEs psychosocial
layer at generation time so the classifier, fairness audit, and causal
layer all have one self-contained, schema-conforming dataframe to run on.

Exists for the same reason `aces_simulated.py` does: `nij_loader.py` and
`compas_loader.py` are still stubs, so there is currently no real
recidivism data source in this project at all. This unblocks the
Streamlit demo (and anyone testing the pipeline) without waiting on
either loader — every row is `source="recidivism_simulated"`, clearly
labeled as synthetic, and that label must be propagated into every
downstream report exactly like the ACEs case.

The join itself is the thing BUILD_PLAN.md's open questions flag as not
wired up for *real* data (matching a real recidivism dataset's subjects
to a real ACEs dataset's subjects by identity is a genuine data-linkage
problem this project hasn't solved). For synthetic data there's no such
problem: this module calls `generate_aces_simulated` directly and builds
the recidivism outcome as a function of the same subjects' ace_score, so
"joined" here just means "generated together," not "linked via a real
key."

Demographic consistency: rather than sampling age/sex/race-or-country a
second time independently, this reuses each subject's ACEs
demographic_stratum (`"{group}_{sex}"`) so a subject's recidivism-side
demographics agree with their ACEs-side stratum. For jurisdiction="US",
`group` is a race category and populates `race_ethnicity`
(`country_of_birth` stays null); for "NL", `group` is a country of birth
and populates `country_of_birth` (`race_ethnicity` stays null) — matching
RECIDIVISM_SCHEMA's nullable, mutually-exclusive design for these two
columns.

Recidivism probability is a hand-set function of prior_convictions,
ace_score, and supervision_intensity (negative effect, matching the
causal story models/causal/dag_spec.yaml is built around) — illustrative
only, not calibrated to any published recidivism-rate literature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.loaders.aces_simulated import generate_aces_simulated
from data.schema import validate_recidivism_schema

OFFENSE_TYPES = ("property", "violent", "drug")
SUPERVISION_LEVELS = ("low", "high")


def generate_recidivism_simulated(n: int, random_state: int = 42, jurisdiction: str = "US") -> pd.DataFrame:
    """Generate `n` synthetic recidivism records for `jurisdiction`, with
    an `ace_score` column already joined in from `generate_aces_simulated`.
    """
    rng = np.random.default_rng(random_state)

    aces = generate_aces_simulated(n, random_state=random_state, jurisdiction=jurisdiction)
    ace_score = aces["ace_score"].to_numpy()
    group, sex = zip(*(stratum.rsplit("_", 1) for stratum in aces["demographic_stratum"].astype(str)))

    age = rng.integers(18, 70, n)
    prior_convictions = np.clip(rng.poisson(0.5 + 0.6 * ace_score), 0, 20)
    offense_type = rng.choice(OFFENSE_TYPES, size=n)

    p_high_supervision = 1 / (1 + np.exp(-(0.15 * prior_convictions - 1.5)))
    supervision_intensity = np.where(rng.random(n) < p_high_supervision, "high", "low")
    treatment_effect = (supervision_intensity == "high").astype(int)

    logit = 0.25 * prior_convictions + 0.15 * ace_score - 0.9 * treatment_effect - 1.5
    recidivated = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "subject_id": aces["subject_id"],
            "source": "recidivism_simulated",
            "jurisdiction": jurisdiction,
            "age": age,
            "sex": list(sex),
            "prior_convictions": prior_convictions,
            "offense_type": offense_type,
            "supervision_intensity": supervision_intensity,
            "recidivated": recidivated,
            "ace_score": ace_score,
        }
    )
    if jurisdiction == "US":
        df["race_ethnicity"] = list(group)
    else:
        df["country_of_birth"] = list(group)

    categorical_columns = ["source", "jurisdiction", "sex", "offense_type", "supervision_intensity"]
    categorical_columns += ["race_ethnicity"] if jurisdiction == "US" else ["country_of_birth"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    validate_recidivism_schema(df)
    return df
