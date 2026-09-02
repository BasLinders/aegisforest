"""Streamlit frontend: renders Module A (treatment-effect) and Module B
(statement review) outputs. Pure-Python, no build step — run locally with:

    streamlit run app/streamlit_app.py

Scaffold only: the pipeline functions it will eventually call
(`models/causal/dml.py`, `models/nli/contradiction.py`, etc.) are still
stubs, so this renders placeholder sections rather than calling into them.
Wire this up once each module produces real output, and keep the framing
constraints from `data/schema.py` / `reports/templates/` intact:
CATE estimates always paired with refutation results, Module B scores
always "flagged for review," synthetic ACEs data always labeled.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def render_module_a(config: dict) -> None:
    st.header("Module A — Treatment Effects")
    aces_source = config["module_a"]["data"]["aces_source"]
    if aces_source == "aces_simulated":
        st.warning("SYNTHETIC DATA — psychosocial confounders are simulated, not observed.")
    st.caption(f"ACEs data source: `{aces_source}`")
    st.info("Pipeline not yet implemented — this section will show CATE estimates "
            "for supervision intensity / program enrollment on reoffense, always "
            "paired with their DoWhy refutation results.")


def render_module_b(config: dict) -> None:
    st.header("Module B — Statement Review")
    st.caption(f"NLI checkpoint: `{config['module_b']['nli']['checkpoint']}`")
    st.info("Pipeline not yet implemented — this section will list statement "
            "pairs flagged for human review. Scores are never a deception or "
            "guilt signal.")


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
