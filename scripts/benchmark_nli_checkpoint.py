"""Benchmark a Module B NLI checkpoint against real Dutch NLI data.

Loads SICK-NL (maximedb/sick_nl on the Hugging Face Hub — a Dutch
translation of SICK, see https://github.com/gijswijnholds/sick_nl),
runs the checkpoint on its test split, and reports two things:

1. 3-way accuracy/macro-F1 (argmax over ENTAILMENT/NEUTRAL/CONTRADICTION)
   — a general quality signal.
2. A threshold sweep over the binary decision `score_contradictions()`
   actually makes in production: is the raw CONTRADICTION-label score
   >= `contradiction_threshold`? This is the metric that matters for
   Module B's `flagged` output — (1) can look fine while this is bad,
   since argmax ignores where the CONTRADICTION score sits relative to
   the configured threshold.

Not part of the test suite: needs network access to the Hub and takes
tens of minutes on CPU for the full test split (~4900 pairs). Run it
directly:

    python scripts/benchmark_nli_checkpoint.py
    python scripts/benchmark_nli_checkpoint.py --checkpoint <other-model> --limit 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    # Makes `requests` (used by huggingface_hub) trust the OS certificate
    # store, not just certifi's bundled CAs. Needed behind a TLS-intercepting
    # corporate/sandbox proxy, where `requests` otherwise fails with
    # SSLCertVerificationError even though plain `urllib` succeeds (urllib
    # already trusts the OS store). Harmless no-op if not installed or not
    # needed: `pip install pip-system-certs`.
    import pip_system_certs.wrapt_requests  # noqa: F401
except ImportError:
    pass

from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import pipeline

DEFAULT_CHECKPOINT = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
LABELS = ("ENTAILMENT", "NEUTRAL", "CONTRADICTION")
SWEEP_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _predicted_label(label_scores: list[dict]) -> str:
    best = max(label_scores, key=lambda item: item["score"])
    return best["label"].upper()


def _contradiction_score(label_scores: list[dict]) -> float:
    """Same extraction as models/nli/contradiction.py::_contradiction_score
    — matches on 'contra' in the label, case-insensitively, so this stays
    correct regardless of a checkpoint's exact label casing."""
    for item in label_scores:
        if "contra" in item["label"].lower():
            return float(item["score"])
    raise ValueError(f"No contradiction label found in pipeline output: {label_scores}")


def _threshold_sweep(y_is_contradiction: list[bool], contradiction_scores: list[float]) -> list[dict]:
    """For each candidate threshold, compute precision/recall/F1 for the
    binary decision score_contradictions() actually makes in production:
    flagged = contradiction_score >= threshold."""
    rows = []
    for threshold in SWEEP_THRESHOLDS:
        y_pred = [score >= threshold for score in contradiction_scores]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_is_contradiction, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_flagged": sum(y_pred),
            }
        )
    return rows


def run_benchmark(checkpoint: str, limit: int | None, batch_size: int) -> dict:
    dataset = load_dataset("maximedb/sick_nl")["test"]
    if limit is not None:
        dataset = dataset.select(range(limit))

    pipe = pipeline("text-classification", model=checkpoint, top_k=None)
    batch = [{"text": row["sentence_A"], "text_pair": row["sentence_B"]} for row in dataset]

    t0 = time.time()
    outputs = pipe(batch, batch_size=batch_size)
    elapsed = time.time() - t0

    y_true = [row["entailment_label"] for row in dataset]
    y_pred = [_predicted_label(o) for o in outputs]

    report = classification_report(y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))

    contradiction_scores = [_contradiction_score(o) for o in outputs]
    y_is_contradiction = [label == "CONTRADICTION" for label in y_true]
    sweep = _threshold_sweep(y_is_contradiction, contradiction_scores)
    best = max(sweep, key=lambda row: row["f1"])

    return {
        "checkpoint": checkpoint,
        "dataset": "maximedb/sick_nl (test split)",
        "n_examples": len(dataset),
        "elapsed_seconds": elapsed,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_label": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in LABELS
        },
        "confusion_matrix": {"labels": list(LABELS), "matrix": matrix.tolist()},
        "contradiction_threshold_sweep": sweep,
        "best_threshold_by_f1": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output",
        default="reports/output/nli_benchmark_sick_nl.json",
        help="Where to write the JSON results",
    )
    args = parser.parse_args()

    results = run_benchmark(args.checkpoint, args.limit, args.batch_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"n={results['n_examples']}  accuracy={results['accuracy']:.4f}  macro_f1={results['macro_f1']:.4f}")
    print(f"elapsed: {results['elapsed_seconds']:.1f}s")
    print()
    print("contradiction_threshold sweep (binary: is this pair a real contradiction?):")
    print(f"{'threshold':>10}  {'precision':>10}  {'recall':>10}  {'f1':>10}  {'n_flagged':>10}")
    for row in results["contradiction_threshold_sweep"]:
        print(
            f"{row['threshold']:>10.2f}  {row['precision']:>10.4f}  {row['recall']:>10.4f}  "
            f"{row['f1']:>10.4f}  {row['n_flagged']:>10d}"
        )
    best = results["best_threshold_by_f1"]
    print(f"best by F1: threshold={best['threshold']} (precision={best['precision']:.4f}, recall={best['recall']:.4f})")
    print()
    print(f"results written to {output_path}")


if __name__ == "__main__":
    main()
