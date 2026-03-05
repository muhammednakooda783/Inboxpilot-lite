from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate import EvalRecord, compute_metrics, load_dataset


def test_load_dataset_parses_jsonl(tmp_path: Path):
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "How do I reset my password?", "category": "question"}),
                json.dumps({"text": "I want a refund.", "category": "complaint"}),
            ]
        ),
        encoding="utf-8",
    )
    rows = load_dataset(dataset_path)
    assert rows == [
        ("How do I reset my password?", "question"),
        ("I want a refund.", "complaint"),
    ]


def test_compute_metrics_returns_expected_shape():
    records = [
        EvalRecord(
            text="Q",
            actual="question",
            predicted="question",
            confidence=0.9,
            latency_ms=5,
            classifier_used="rules",
            ok=True,
            error_message=None,
        ),
        EvalRecord(
            text="S",
            actual="sales",
            predicted="other",
            confidence=0.6,
            latency_ms=7,
            classifier_used="rules",
            ok=True,
            error_message=None,
        ),
    ]
    result = compute_metrics(records, mode="rules")
    assert result["total_examples"] == 2
    assert result["accuracy"] == 0.5
    assert "macro_f1" in result
    assert "per_category" in result
    assert "confusion_matrix" in result
