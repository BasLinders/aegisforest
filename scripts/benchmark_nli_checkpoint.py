"""Benchmark a Module B NLI checkpoint against real Dutch NLI data.

Loads SICK-NL (maximedb/sick_nl on the Hugging Face Hub — a Dutch
translation of SICK, see https://github.com/gijswijnholds/sick_nl),
runs the checkpoint on its test split, and reports accuracy/macro-F1
against the human-annotated ENTAILMENT/NEUTRAL/CONTRADICTION labels.

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
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

DEFAULT_CHECKPOINT = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
LABELS = ("ENTAILMENT", "NEUTRAL", "CONTRADICTION")


def _predicted_label(label_scores: list[dict]) -> str:
    best = max(label_scores, key=lambda item: item["score"])
    return best["label"].upper()


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
    print(f"results written to {output_path}")


if __name__ == "__main__":
    main()
