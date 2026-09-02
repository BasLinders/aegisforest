import pandas as pd
import pytest

from models.causal.dml import CausalEffectResult
from reports.render import render_module_a_report, render_module_b_reports


def _fake_effect_result(ate: float = -0.25, refutation_results: dict | None = None) -> CausalEffectResult:
    if refutation_results is None:
        refutation_results = {
            "random_common_cause": {
                "refutation_type": "Refute: Add a random common cause",
                "estimated_effect": ate,
                "new_effect": ate + 0.01,
                "p_value": 0.62,
            }
        }
    return CausalEffectResult(
        causal_model=None,
        identified_estimand=None,
        estimate=type("FakeEstimate", (), {"value": ate})(),
        refutation_results=refutation_results,
    )


def test_module_a_report_shows_synthetic_banner_when_simulated():
    result = _fake_effect_result()
    text = render_module_a_report(result, "supervision_intensity", "recidivated", aces_source="aces_simulated")
    assert "SYNTHETIC DATA" in text
    assert "-0.250" in text
    assert "random_common_cause" in text
    assert "p=0.620" in text


def test_module_a_report_omits_synthetic_banner_when_real():
    result = _fake_effect_result()
    text = render_module_a_report(result, "supervision_intensity", "recidivated", aces_source="aces_real")
    assert "SYNTHETIC DATA" not in text


def test_module_a_report_flags_missing_refutation():
    result = _fake_effect_result(refutation_results={})
    text = render_module_a_report(result, "supervision_intensity", "recidivated", aces_source="aces_real")
    assert "must not be treated as" in text


def test_module_a_report_writes_file(tmp_path):
    result = _fake_effect_result()
    render_module_a_report(
        result, "supervision_intensity", "recidivated", aces_source="aces_real", output_dir=tmp_path
    )
    written = tmp_path / "module_a_treatment_effect_report.md"
    assert written.exists()
    assert "Treatment Effect Report" in written.read_text(encoding="utf-8")


def _fixture_contradictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "s1",
                "session_id_a": "session1",
                "statement_id_a": "t1",
                "text_a": "I was home",
                "session_id_b": "session2",
                "statement_id_b": "t2",
                "text_b": "I was at the bar",
                "contradiction_score": 0.9,
                "flagged": True,
            },
            {
                "subject_id": "s1",
                "session_id_a": "session1",
                "statement_id_a": "t1b",
                "text_a": "It was raining",
                "session_id_b": "session2",
                "statement_id_b": "t2",
                "text_b": "I was at the bar",
                "contradiction_score": 0.1,
                "flagged": False,
            },
            {
                "subject_id": "s2",
                "session_id_a": "session1",
                "statement_id_a": "u1",
                "text_a": "text a",
                "session_id_b": "session2",
                "statement_id_b": "u2",
                "text_b": "text b",
                "contradiction_score": 0.05,
                "flagged": False,
            },
        ]
    )


def test_module_b_report_only_includes_flagged_pairs():
    rendered = render_module_b_reports(_fixture_contradictions())
    assert set(rendered) == {"s1"}  # s2 has no flagged pairs, gets no report
    assert "I was home" in rendered["s1"]
    assert "It was raining" not in rendered["s1"]


def test_module_b_report_never_uses_verdict_language():
    rendered = render_module_b_reports(_fixture_contradictions())
    text_lower = rendered["s1"].lower()
    for banned in ("deception", "guilt", "lying", "lied"):
        assert banned not in text_lower or "not a determination of" in text_lower


def test_module_b_report_writes_one_file_per_subject(tmp_path):
    render_module_b_reports(_fixture_contradictions(), output_dir=tmp_path)
    written = tmp_path / "module_b_s1_review_flags.md"
    assert written.exists()
    assert not (tmp_path / "module_b_s2_review_flags.md").exists()


def test_module_b_report_returns_empty_when_nothing_flagged():
    df = _fixture_contradictions()
    df["flagged"] = False
    rendered = render_module_b_reports(df)
    assert rendered == {}
