# aegisforest — Project Handoff

Repo: https://github.com/BasLinders/aegisforest
Status: Personal portfolio project. Not built for or deployed by any law-enforcement organization.
Core idea: Causal (not purely predictive) analysis of criminal-justice intervention effects, extended with contradiction detection across suspect statements. Two independent modules sharing one repo.

## 1. Project scope

### Module A — Causal recidivism analysis

- Baseline: XGBoost / logistic regression classifier on COMPAS + NIJ Recidivism Forecasting Challenge data, with a fairness audit (calibration + false-positive-rate parity across demographic subgroups — replicate the ProPublica COMPAS critique as a sanity check).
- Causal layer: Double Machine Learning / causal forests (EconML `CausalForestDML`, DoWhy for DAG specification and refutation tests) estimating heterogeneous treatment effects of supervision intensity / program enrollment on reoffense — not a risk-of-reoffense score, an effect-of-intervention estimate.
- Confounder layer: psychosocial variables (ACEs-style adversity indicators), with two interchangeable data sources, selectable via config:
  - `source: aces_real` (default) — real, de-identified, publicly available ACEs data (CDC-Kaiser ACE Study public-use files, or YRBSS state-level adversity/justice-contact data for US; no direct equivalent survey exists for NL — see `data/loaders/aces_real_loader.py`).
  - `source: aces_simulated` — synthetic psychosocial layer generated from published population-level ACEs prevalence rates by demographic strata, clearly labeled as synthetic in all outputs and reports. Jurisdiction-aware: US strata by race x sex (CDC BRFSS ACE module), NL strata by country of birth x sex (Dutch criminal-justice registries generally don't record race; the NL table uses CBS child-poverty rates by country of birth as a proxy — see `data/loaders/aces_simulated.py`).
  - Both sources must conform to the same schema (`data/schema.py`'s `ACES_SCHEMA`, which includes `jurisdiction`) so downstream pipeline code is source-agnostic. The classifier, fairness audit, and causal layer are all schema/column-driven and don't hardcode US-specific categories — but there's no NL recidivism data loader yet (`nij_loader.py`/`compas_loader.py` are both US-specific), so a full NL pipeline run needs that written first.

### Module B — Statement contradiction detection

- NLI model applied pairwise across statement chunks from the same subject across interview sessions. Default checkpoint is multilingual (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, trained on ~105k Dutch NLI pairs alongside English), so it covers both US and NL statements with one checkpoint — no jurisdiction-conditional swap needed.
- Output: contradiction-likelihood scores per sentence pair, surfaced as "flagged for human review," never as a deception/guilt signal.
- Optional extension (implemented, off by default): lightweight NER (spaCy) + timeline/entity consistency checks layered on top of NLI scores — flags cross-session statement pairs whose named locations/dates/people are completely disjoint for a given entity type. Complements the NLI check rather than replacing it: it has no negation handling, so "I was in Chicago" vs. "I never went to Chicago" isn't flagged by this alone.

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
│   └── streamlit_app.py         # Streamlit frontend (module A + B views)
├── config/
│   └── default.yaml             # data source toggle (aces_real | aces_simulated), model params
├── data/
│   ├── loaders/
│   │   ├── nij_loader.py
│   │   ├── compas_loader.py
│   │   ├── aces_real_loader.py
│   │   ├── aces_simulated.py     # synthetic generator, US (race) and NL (country of birth)
│   │   └── recidivism_simulated.py  # synthetic generator, joined with aces_simulated's output
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
│       └── timeline_check.py     # optional NER + entity consistency layer (off by default)
├── notebooks/                    # exploratory analysis, not production code
├── reports/
│   └── templates/                # output templates emphasizing CATE framing, not risk scores
├── scripts/
│   └── benchmark_nli_checkpoint.py  # evaluates the Module B checkpoint against real Dutch NLI data (SICK-NL)
└── tests/
```

## 3. Key design constraints for implementation

1. Data source is a runtime config choice, not a code branch. `aces_real` and `aces_simulated` loaders must both return a dataframe matching `data/schema.py`; nothing downstream should need to know which was used, except a metadata flag propagated into every report/output artifact.
2. Every causal estimate ships with its refutation results. No `CausalForestDML` output should be surfaced (in notebooks, reports, or CLI output) without the corresponding DoWhy refuter results attached.
3. Fairness audit runs on both Module A layers — the baseline classifier and the CATE outputs — checking whether psychosocial confounders correlate with protected demographic variables and whether adding them shifts fairness metrics (proxy-discrimination risk).
4. Module B outputs are always framed as "flagged for review," never as a verdict. Enforce this at the output-template level (see `reports/templates/`), not just in prose documentation.
5. Synthetic data must be visibly labeled everywhere it's used — figure titles, report headers, CLI banners — so no output can be mistaken for real-data findings.

## 4. Construction plan

1. Scaffold repo structure + `pyproject.toml` + `config/default.yaml`.
2. `data/schema.py` + both ACEs loaders (start with `aces_simulated.py`, since it has no external data dependency, to unblock pipeline development).
3. Module A baseline classifier + fairness audit (COMPAS/NIJ only, no causal layer yet) — get this working and validated first.
4. Module A causal layer (DAG → DML → refutation) on top of the working baseline.
5. Module B NLI contradiction detector as an independent track (can be built in parallel with Module A).
6. Reporting templates last, once both modules produce stable outputs to render.
7. ~~Streamlit frontend wired to the real pipeline~~ — done. `app/streamlit_app.py` calls the real classifier, fairness audit, causal layer, and NLI/timeline checks — no more placeholders. Run locally with `streamlit run app/streamlit_app.py`.
   Module A needed a data source to wire to and none existed (`nij_loader.py`/`compas_loader.py`/`aces_real_loader.py` are all still stubs), so `data/loaders/recidivism_simulated.py` was added: generates recidivism data joined with `generate_aces_simulated`'s output at generation time (see that module's docstring — this is a synthetic-only stand-in for the real cross-dataset join, which is still an open problem). Added `country_of_birth` to `RECIDIVISM_SCHEMA` (nullable, mutually exclusive with `race_ethnicity`) so NL-jurisdiction synthetic subjects carry their demographic stratum without misusing the race field.
   The causal fit is genuinely slow in the UI (CausalForestDML + all three refuters, several minutes even at small n/n_estimators) — gated behind a button, cached by its inputs (`st.cache_resource`, not `st.cache_data`: `CausalEffectResult` wraps a live DoWhy `CausalModel` that isn't reliably picklable). Module B's NLI checkpoint (~1.1GB) is cached the same way.

## 5. Open questions to resolve during construction

- Exact source/access method for CDC-Kaiser ACE public-use files (may require a data-use request even for public-use tier — confirm before assuming direct download).
- Whether NIJ dataset's supervision-level field is granular enough to serve as a clean binary/ordinal treatment variable for DML, or needs recoding.
- ~~Final choice of pretrained NLI checkpoint~~ — resolved: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, chosen over Dutch-only alternatives (e.g. `LoicDL/bert-base-dutch-cased-finetuned-snli`) specifically so one checkpoint covers both US and NL statements, and over plain-XNLI multilingual models (e.g. `joeddav/xlm-roberta-large-xnli`) since XNLI's 15 languages don't include Dutch.
  ~~Benchmarked against real Dutch statement data~~ — resolved: `scripts/benchmark_nli_checkpoint.py` against [SICK-NL](https://github.com/gijswijnholds/sick_nl) (first 500 of 4906 test pairs — the full run is CPU-heavy, ~27 min): **67.6% accuracy, 65.6% macro-F1**. Per-label recall: ENTAILMENT 71.1%, NEUTRAL 61.7%, CONTRADICTION 90.5%. Caveat: SICK-NL is machine-translated general-domain image-caption pairs, not naturally-authored Dutch interview/interrogation statements — no domain-matched Dutch benchmark exists, so this is a proxy evaluation, same spirit as the ACEs synthetic table's poverty-rate proxy.
  ~~Whether the high contradiction false-positive rate needs mitigating~~ — resolved: keep `contradiction_threshold` at 0.5 (config default), deliberately. The script's threshold sweep shows raising it trades recall for precision (0.5: 43.5% precision / 90.5% recall → 0.9: 61.5% precision / 63.5% recall → 0.95, best F1: 71.2% precision / 58.7% recall), but a missed real contradiction (false negative) is worse than an extra reviewer look (false positive) for this use case — investigations are already hectic enough that contradictions going unflagged is a live problem, and `flagged` already means "send to review," never a verdict, so the cost of over-flagging is bounded. Don't raise this threshold later without re-litigating that tradeoff explicitly.
- No NL recidivism data loader exists yet — only the ACEs confounder layer is jurisdiction-aware so far. A real one would likely draw on WODC's Recidivism Monitor or CBS/politie open crime data (see project memory / earlier discussion on public Dutch sources).
- Real NL ACEs data has no direct survey equivalent to CDC-Kaiser/YRBSS; `aces_real_loader.py`'s NL path would mean reshaping WODC/CBS adversity-adjacent statistics to `ACES_SCHEMA`, not loading a comparable existing dataset.
- No real cross-dataset join exists between recidivism and ACEs data by subject identity — `recidivism_simulated.py` sidesteps this by generating both together, which only works because there's no real-world linkage problem for synthetic subjects.

## 6. Known upstream bugs (dowhy 0.14 + pandas 3.0)

Found while wiring the Streamlit frontend to the real pipeline — both are real bugs in the dowhy/pandas version combination this project pins, not bugs in this project's code, but worth recording so they aren't rediscovered the hard way after a dependency bump:

- `placebo_treatment_refuter`'s `placebo_type="permute"` path does `data[treatment_names].values` (a list-indexed selection, so it returns a 2D array even for one column). Older pandas silently handled assigning that back as a column; pandas 3.0's stricter `maybe_convert_objects` raises `ValueError: Buffer has wrong number of dimensions (expected 1, got 2)` instead. Worked around in `models/causal/refutation.py` by not passing `placebo_type="permute"` — the default path (resample the treatment from a random distribution) hits different code and works, and is still a legitimate placebo strategy.
- That default path only has branches for float/bool/int/`category`-dtype treatment columns (dispatched via `type_dict[treatment_names[0]].name`) — a plain string/object-dtype treatment column matches none of them and raises `UnboundLocalError: cannot access local variable 'new_treatment'`. Worked around in `models/causal/dag.py::build_causal_model`, which now casts the treatment column to `category` dtype if it isn't already, so callers don't need to know this.

Both are covered by `tests/test_causal_layer.py`'s slow end-to-end test, which runs `placebo_treatment_refuter` specifically so a regression (e.g. from reverting either workaround, or a dowhy/pandas upgrade reintroducing the issue) fails a test rather than surfacing as a confusing runtime error.
