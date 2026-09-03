"""Streamlit frontend: renders Module A (treatment-effect) and Module B
(statement review) outputs. Pure-Python, no build step — run locally with:

    streamlit run app/streamlit_app.py

Module A runs on synthetic data (data/loaders/recidivism_simulated.py):
nij_loader.py/compas_loader.py/aces_real_loader.py are all still stubs, so
there is currently no real recidivism data source anywhere in this
project — this demo can't be pointed at real data yet, and its SYNTHETIC
DATA banner is unconditional here, not read from config's aces_source
(which defaults to aces_real, an aspirational setting for when a real
loader exists — trusting it here would silently hide the banner on data
that is, in fact, always synthetic in this app today).

Module A's causal fit is genuinely slow (real CausalForestDML plus all
three configured refuters, several minutes even at small n/n_estimators)
— it's gated behind a button, not run on every rerun, and cached by its
inputs. Module B's NLI checkpoint is a
~1.1GB download on first use — cached as a Streamlit resource so it's
loaded once per server process, not once per click.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from data.loaders.recidivism_simulated import generate_recidivism_simulated
from models.causal.dml import estimate_treatment_effect
from models.classifier.fairness_audit import run_fairness_audit
from models.classifier.train import train_baseline
from models.nli.contradiction import score_contradictions
from models.nli.timeline_check import check_timeline_consistency
from reports.render import render_module_a_report, render_module_b_reports

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


@st.cache_data(show_spinner=False)
def _generate_recidivism_data(n: int, random_state: int, jurisdiction: str) -> pd.DataFrame:
    return generate_recidivism_simulated(n, random_state=random_state, jurisdiction=jurisdiction)


@st.cache_resource(show_spinner="Fitting CausalForestDML + refuters — this genuinely takes several minutes...")
def _run_causal_layer(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    dag_path: str,
    refuters: tuple[str, ...],
    n_estimators: int,
    random_state: int,
):
    return estimate_treatment_effect(
        df,
        treatment=treatment,
        outcome=outcome,
        dag_path=dag_path,
        refuters=list(refuters),
        control_value="low",
        treatment_value="high",
        n_estimators=n_estimators,
        random_state=random_state,
    )


@st.cache_resource(show_spinner="Loading NLI model (first run downloads ~1.1GB)...")
def _load_nli_pipeline(checkpoint: str):
    from transformers import pipeline

    return pipeline("text-classification", model=checkpoint, top_k=None)


def render_module_a(config: dict) -> None:
    st.header("Module A — Treatment Effects")
    st.warning(
        "SYNTHETIC DATA — nij_loader.py/compas_loader.py/aces_real_loader.py are all "
        "still stubs, so this demo always runs on data/loaders/recidivism_simulated.py, "
        "regardless of config's aces_source setting. Findings are illustrative only."
    )

    data_config = config["module_a"]["data"]
    classifier_config = config["module_a"]["classifier"]
    causal_config = config["module_a"]["causal"]

    col1, col2, col3 = st.columns(3)
    jurisdiction = col1.selectbox("Jurisdiction", ["US", "NL"], index=["US", "NL"].index(data_config["jurisdiction"]))
    n = col2.number_input("Sample size", min_value=100, max_value=5000, value=300, step=100)
    random_state = col3.number_input("Random seed", value=classifier_config["random_state"], step=1)

    df = _generate_recidivism_data(int(n), int(random_state), jurisdiction)
    st.caption(f"Generated {len(df)} synthetic subjects (jurisdiction={jurisdiction}).")

    protected_attributes = classifier_config["protected_attributes"] + ["country_of_birth"]
    result = train_baseline(
        df,
        model=classifier_config["model"],
        protected_attributes=protected_attributes,
        test_size=classifier_config["test_size"],
        random_state=int(random_state),
    )

    st.subheader("Baseline classifier")
    metric_cols = st.columns(len(result.metrics))
    for col, (name, value) in zip(metric_cols, result.metrics.items()):
        col.metric(name, f"{value:.3f}")

    fairness = run_fairness_audit(
        result.y_test.to_numpy(),
        result.y_pred,
        result.y_pred_proba,
        result.protected_test,
        metrics=config["module_a"]["fairness_audit"]["metrics"],
        subgroup_min_n=config["module_a"]["fairness_audit"]["subgroup_min_n"],
    )
    st.subheader("Fairness audit")
    for metric_name, metric_df in fairness.items():
        st.caption(metric_name)
        st.dataframe(metric_df, use_container_width=True)

    st.subheader("Causal layer")
    st.caption(
        f"CausalForestDML: {causal_config['treatment']} → {causal_config['outcome']}. "
        "Genuinely slow (several minutes — fits the forest, then all three refuters) — click to run."
    )
    if st.button("Run causal analysis"):
        causal_result = _run_causal_layer(
            df,
            treatment=causal_config["treatment"],
            outcome=causal_config["outcome"],
            dag_path=causal_config["dag_path"],
            refuters=tuple(causal_config["refuters"]),
            n_estimators=20,  # small on purpose: config's 500 would take far too long for an interactive demo
            random_state=int(random_state),
        )
        report_text = render_module_a_report(
            causal_result,
            treatment=causal_config["treatment"],
            outcome=causal_config["outcome"],
            aces_source="aces_simulated",  # always true in this demo — see module docstring
        )
        st.markdown(report_text)


def render_module_b(config: dict) -> None:
    st.header("Module B — Statement Review")
    nli_config = config["module_b"]["nli"]
    st.caption(f"NLI checkpoint: `{nli_config['checkpoint']}`")

    mode_label = st.radio(
        "Contradiction flagging",
        list(_THRESHOLD_MODES),
        help="Analyst call, not automatic — Recall is the default for a reason: "
        "a missed real contradiction is worse than an extra reviewer look. Only "
        "switch to Balanced for cases you've judged low-risk or short on review "
        "time.",
    )
    mode = _THRESHOLD_MODES[mode_label]
    st.caption(
        f"contradiction_threshold = {mode['threshold']} "
        f"(precision {mode['precision']:.0%}, recall {mode['recall']:.0%} on the SICK-NL benchmark)"
    )

    run_timeline_check = st.checkbox(
        "Also run the timeline/entity consistency check",
        value=config["module_b"]["timeline_check"]["enabled"],
        help="Optional extension (models/nli/timeline_check.py) — complements the NLI "
        "check but has no negation handling, so read its results alongside the NLI "
        "flags, not instead of them.",
    )
    timeline_language = st.selectbox(
        "Statement language (for the timeline check's NER model)",
        list(_NER_MODELS),
        disabled=not run_timeline_check,
        help="en_core_web_sm and nl_core_news_sm share the same entity label scheme, "
        "so only the model changes here, not the comparison logic.",
    )

    st.write("Enter each session's statements one per line — a real analyst would paste transcript excerpts here.")
    col1, col2 = st.columns(2)
    # `value=`, not `placeholder=`: a placeholder is just a visual hint and
    # leaves the field empty, so "Analyze" would find nothing to score.
    # Prefilled with an example pair so the demo works without the analyst
    # having to type anything first.
    session_a_text = col1.text_area("Session 1", height=150, value="I was alone the entire evening.")
    session_b_text = col2.text_area("Session 2", height=150, value="John and I were together the whole evening.")

    if st.button("Analyze statements"):
        statements = _statements_from_text(session_a_text, session_b_text)
        if len(statements) < 2:
            st.error("Enter at least one statement in each session.")
            return

        pipe = _load_nli_pipeline(nli_config["checkpoint"])
        contradictions = score_contradictions(
            statements,
            checkpoint=nli_config["checkpoint"],
            contradiction_threshold=mode["threshold"],
            nli_pipeline=pipe,
        )

        st.subheader("Contradiction flags")
        if contradictions.empty:
            st.info("No cross-session statement pairs to compare.")
        else:
            reports = render_module_b_reports(contradictions)
            if reports:
                for report_text in reports.values():
                    st.markdown(report_text)
            else:
                st.info("No pairs scored above the contradiction threshold.")
            with st.expander("All scored pairs (including unflagged)"):
                st.dataframe(contradictions, use_container_width=True)

        if run_timeline_check:
            st.subheader("Timeline/entity consistency")
            timeline = check_timeline_consistency(statements, ner_model=_NER_MODELS[timeline_language])
            if timeline.empty:
                st.info("No comparable entities found across sessions.")
            else:
                st.dataframe(timeline, use_container_width=True)


# Precision/recall/F1 at these two thresholds come from
# scripts/benchmark_nli_checkpoint.py's real sweep against SICK-NL (see
# BUILD_PLAN.md's open questions). RECALL is the config default and stays
# the analyst's default here too — a missed real contradiction is worse
# than an extra reviewer look for this use case. BALANCED trades that
# recall away for fewer false positives, for analysts who explicitly
# decide the case is low-risk/resource-constrained enough to accept it;
# it is never selected automatically.
_THRESHOLD_MODES = {
    "Recall (default) — flag more, miss less": {
        "threshold": 0.5,
        "precision": 0.435,
        "recall": 0.905,
    },
    "Balanced — fewer false positives, may miss some contradictions": {
        "threshold": 0.95,
        "precision": 0.712,
        "recall": 0.587,
    },
}

# Both share the same entity label scheme (see models/nli/timeline_check.py's
# docstring) — only the model itself needs to match the statements' language.
_NER_MODELS = {
    "English": "en_core_web_sm",
    "Dutch": "nl_core_news_sm",
}


def _statements_from_text(session_a_text: str, session_b_text: str) -> pd.DataFrame:
    rows = []
    for session_id, text in (("session1", session_a_text), ("session2", session_b_text)):
        for i, line in enumerate(line.strip() for line in text.splitlines()):
            if not line:
                continue
            rows.append(
                {
                    "subject_id": "demo-subject",
                    "session_id": session_id,
                    "statement_id": f"{session_id}-{i}",
                    "text": line,
                }
            )
    return pd.DataFrame(rows, columns=["subject_id", "session_id", "statement_id", "text"])


def main() -> None:
    st.set_page_config(page_title="aegisforest", layout="wide")
    st.title("aegisforest")
    st.caption("Personal portfolio project — not built for or deployed by any law-enforcement organization.")

    config = load_config()
    module = st.sidebar.radio("Module", ["Module A — Treatment Effects", "Module B — Statement Review"])

    if module.startswith("Module A"):
        render_module_a(config)
    else:
        render_module_b(config)


if __name__ == "__main__":
    main()
