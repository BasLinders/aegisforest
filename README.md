# aegisforest

Personal portfolio project. Not built for or deployed by any law-enforcement organization.

Two independent modules:

- **Module A** — causal (not purely predictive) analysis of criminal-justice intervention effects: a baseline recidivism classifier with a fairness audit, plus a causal layer (DoWhy + EconML `CausalForestDML`) estimating heterogeneous treatment effects of supervision intensity / program enrollment on reoffense.
- **Module B** — contradiction detection across a subject's statements over multiple interview sessions, using pairwise NLI. Output is always framed as "flagged for human review," never as a deception or guilt signal.

Explicit non-goals: no individual-level guilt/deception/violence-risk scoring, no biological/medical predictors of aggression, no causal claims beyond what DoWhy's refutation tests support.

See [BUILD_PLAN.md](BUILD_PLAN.md) for full scope, design constraints, and build order.

## Running the frontend

```
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```
