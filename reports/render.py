"""Renders Module A / Module B outputs through the templates in
reports/templates/, which are where the handoff's output-framing
constraints (refutation results required, synthetic-data labeling,
flagged-for-review only) are actually enforced — this module just supplies
the data, it doesn't decide the wording.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from models.causal.dml import CausalEffectResult

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _template_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),  # rendering Markdown, not HTML
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _write(output_dir: str | Path, filename: str, text: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


def render_module_a_report(
    result: CausalEffectResult,
    treatment: str,
    outcome: str,
    aces_source: str,
    output_dir: str | Path | None = None,
) -> str:
    """Render the treatment-effect report for one CausalEffectResult.

    `aces_source` must be passed explicitly (not read off `result`): the
    causal layer works on an already-merged dataframe and doesn't itself
    track which ACEs loader produced the ace_score column feeding it — see
    BUILD_PLAN.md's open question on wiring up that join.
    """
    template = _template_env().get_template("module_a_cate_report.md.j2")
    text = template.render(
        treatment=treatment,
        outcome=outcome,
        aces_source=aces_source,
        ate=result.ate,
        refutation_results=result.refutation_results,
    )
    if output_dir is not None:
        _write(output_dir, "module_a_treatment_effect_report.md", text)
    return text


def render_module_b_reports(
    contradictions: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Render one review-flag report per subject with at least one flagged
    pair (subjects with none don't get a report — there's nothing to
    review). Returns {subject_id: rendered_markdown}.
    """
    template = _template_env().get_template("module_b_review_flag.md.j2")
    flagged = contradictions[contradictions["flagged"]]

    rendered = {}
    for subject_id, group in flagged.groupby("subject_id"):
        text = template.render(subject_id=subject_id, flagged_pairs=group.to_dict("records"))
        rendered[subject_id] = text
        if output_dir is not None:
            _write(output_dir, f"module_b_{subject_id}_review_flags.md", text)
    return rendered
