# aegisforest

Causal effect analysis of justice-system interventions, extended with statement contradiction flagging for human review. Personal portfolio project — not built for or deployed by any law-enforcement organization.

Two independent modules:

- **Module A** — causal (not purely predictive) analysis of criminal-justice intervention effects: a baseline recidivism classifier with a fairness audit, plus a causal layer (DoWhy + EconML `CausalForestDML`) estimating heterogeneous treatment effects of supervision intensity / program enrollment on reoffense. Schema-driven and jurisdiction-aware for the ACEs confounder layer (US: race x sex; NL: country of birth x sex, since Dutch registries generally don't record race) — see [BUILD_PLAN.md](BUILD_PLAN.md) for what's actually wired up per jurisdiction today.
- **Module B** — contradiction detection across a subject's statements over multiple interview sessions, using pairwise NLI. Output is always framed as "flagged for human review," never as a deception or guilt signal.

Explicit non-goals: no individual-level guilt/deception/violence-risk scoring, no biological/medical predictors of aggression, no causal claims beyond what DoWhy's refutation tests support.

See [BUILD_PLAN.md](BUILD_PLAN.md) for full scope, design constraints, and build order.

## Setup

DoWhy caps at Python <3.14, so this project needs a dedicated interpreter
rather than whatever Python happens to be newest on your machine:

```
py -3.13 -m venv .venv
.venv\Scripts\pip install -e ".[dev,app]"
```

## Running the frontend

```
.venv\Scripts\streamlit run app/streamlit_app.py
```

## Testing

```
.venv\Scripts\pytest              # fast suite (~3s)
.venv\Scripts\pytest -m slow      # includes the causal-layer integration test, which
                                   # refits a real CausalForestDML and takes ~1 minute
```

## Benchmarking the NLI checkpoint

`scripts/benchmark_nli_checkpoint.py` evaluates Module B's NLI checkpoint
against [SICK-NL](https://github.com/gijswijnholds/sick_nl), a human-annotated
Dutch NLI dataset — not part of the test suite (needs network access, takes
tens of minutes on CPU for the full ~4900-pair test split):

```
.venv\Scripts\pip install -e ".[benchmark]"
.venv\Scripts\python scripts/benchmark_nli_checkpoint.py
.venv\Scripts\python scripts/benchmark_nli_checkpoint.py --limit 500  # quicker, less precise
```
