# aegisforest — Project Handoff

Repo: https://github.com/BasLinders/aegisforest
Status: Personal portfolio project. Not built for or deployed by any law-enforcement organization.
Core idea: Causal (not purely predictive) analysis of criminal-justice intervention effects, extended with contradiction detection across suspect statements. Two independent modules sharing one repo.

## 1. Project scope

### Module A — Causal recidivism analysis

- Baseline: XGBoost / logistic regression classifier on COMPAS + NIJ Recidivism Forecasting Challenge data, with a fairness audit (calibration + false-positive-rate parity across demographic subgroups — replicate the ProPublica COMPAS critique as a sanity check).
- Causal layer: Double Machine Learning / causal forests (EconML `CausalForestDML`, DoWhy for DAG specification and refutation tests) estimating heterogeneous treatment effects of supervision intensity / program enrollment on reoffense — not a risk-of-reoffense score, an effect-of-intervention estimate.
- Confounder layer: psychosocial variables (ACEs-style adversity indicators), with two interchangeable data sources, selectable via config:
  - `source: aces_real` (default) — real, de-identified, publicly available ACEs data (CDC-Kaiser ACE Study public-use files, or YRBSS state-level adversity/justice-contact data).
  - `source: aces_simulated` — synthetic psychosocial layer generated from published population-level ACEs prevalence rates by demographic strata (CDC BRFSS ACE module), clearly labeled as synthetic in all outputs and reports.
  - Both sources must conform to the same schema so downstream pipeline code is source-agnostic.

### Module B — Statement contradiction detection

- NLI model (RoBERTa-large-MNLI or similar) applied pairwise across statement chunks from the same subject across interview sessions.
- Output: contradiction-likelihood scores per sentence pair, surfaced as "flagged for human review," never as a deception/guilt signal.
- Optional extension: lightweight NER (spaCy) + timeline/entity consistency checks layered on top of NLI scores.

### Explicit non-goals

- No individual-level guilt, deception, or "risk of violence" scoring.
- No biological/medical predictors of aggression (deliberately excluded — see design notes below).
- No claim of causal validity beyond what DoWhy's refutation tests support; unconfoundedness is a stated assumption, not a proven property.

## 2. Repo structure

```
aegisforest/
├── pyproject.toml
├── README.md
├── BUILD_PLAN.md                # this file
├── app/
│   └── streamlit_app.py          # Streamlit frontend (module A + B views)
├── config/
│   └── default.yaml             # data source toggle (aces_real | aces_simulated), model params
├── data/
│   ├── loaders/
│   │   ├── nij_loader.py
│   │   ├── compas_loader.py
│   │   ├── aces_real_loader.py
│   │   └── aces_simulated.py    # synthetic generator from BRFSS prevalence tables
│   └── schema.py                 # shared schema both aces sources must satisfy
├── models/
│   ├── classifier/
│   │   ├── train.py              # XGBoost / LR baseline
│   │   └── fairness_audit.py     # calibration + FPR parity across subgroups
│   ├── causal/
│   │   ├── dag.py                # DoWhy causal graph definition
│   │   ├── dml.py                # EconML CausalForestDML wrapper
│   │   └── refutation.py         # placebo treatment, random common cause, subset refuter
│   └── nli/
│       ├── contradiction.py      # pairwise NLI scoring across statement sessions
│       └── timeline_check.py     # optional NER + entity consistency layer
├── notebooks/                    # exploratory analysis, not production code
├── reports/
│   └── templates/                # output templates emphasizing CATE framing, not risk scores
└── tests/
```

## 3. Key design constraints for implementation

1. Data source is a runtime config choice, not a code branch. `aces_real` and `aces_simulated` loaders must both return a dataframe matching `data/schema.py`; nothing downstream should need to know which was used, except a metadata flag propagated into every report/output artifact.
2. Every causal estimate ships with its refutation results. No `CausalForestDML` output should be surfaced (in notebooks, reports, or CLI output) without the corresponding DoWhy refuter results attached.
3. Fairness audit runs on both Module A layers — the baseline classifier and the CATE outputs — checking whether psychosocial confounders correlate with protected demographic variables and whether adding them shifts fairness metrics (proxy-discrimination risk).
4. Module B outputs are always framed as "flagged for review," never as a verdict. Enforce this at the output-template level (see `reports/templates/`), not just in prose documentation.
5. Synthetic data must be visibly labeled everywhere it's used — figure titles, report headers, CLI banners — so no output can be mistaken for real-data findings.

## 4. Suggested build order

1. Scaffold repo structure + `pyproject.toml` + `config/default.yaml`.
2. `data/schema.py` + both ACEs loaders (start with `aces_simulated.py`, since it has no external data dependency, to unblock pipeline development).
3. Module A baseline classifier + fairness audit (COMPAS/NIJ only, no causal layer yet) — get this working and validated first.
4. Module A causal layer (DAG → DML → refutation) on top of the working baseline.
5. Module B NLI contradiction detector as an independent track (can be built in parallel with Module A).
6. Reporting templates last, once both modules produce stable outputs to render.
7. Streamlit frontend (`app/streamlit_app.py`) wired to the real pipeline once both modules produce stable output — pure Python, no separate build/CI pipeline. Run locally with `streamlit run app/streamlit_app.py`.

## 5. Open questions to resolve during build

- Exact source/access method for CDC-Kaiser ACE public-use files (may require a data-use request even for public-use tier — confirm before assuming direct download).
- Whether NIJ dataset's supervision-level field is granular enough to serve as a clean binary/ordinal treatment variable for DML, or needs recoding.
- Final choice of pretrained NLI checkpoint (base vs. large trade-off given local compute).
