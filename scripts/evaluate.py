from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

Category = Literal["question", "complaint", "sales", "spam", "other"]
CATEGORIES: list[Category] = ["question", "complaint", "sales", "spam", "other"]


@dataclass
class EvalRecord:
    text: str
    actual: Category
    predicted: Category
    confidence: float
    latency_ms: int
    classifier_used: str
    ok: bool
    error_message: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inboxpilot-lite classifier.")
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.jsonl",
        help="Path to JSONL dataset with fields: text, category",
    )
    parser.add_argument(
        "--mode",
        choices=["rules", "lmstudio"],
        default="rules",
        help="Classifier mode to evaluate",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for evaluation JSON. Defaults to artifacts/eval/*.json",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[tuple[str, Category]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: list[tuple[str, Category]] = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        text = str(payload.get("text", "")).strip()
        category = str(payload.get("category", "")).strip().lower()
        if not text:
            raise ValueError(f"Invalid dataset row {line_num}: text is required")
        if category not in CATEGORIES:
            raise ValueError(f"Invalid dataset row {line_num}: category={category}")
        rows.append((text, category))  # type: ignore[arg-type]
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def get_classifier(mode: str):
    from app.core.config import get_settings
    from app.services.classifier import RulesClassifier
    from app.services.lmstudio_classifier import LMStudioClassifier

    if mode == "rules":
        return RulesClassifier()
    settings = get_settings()
    return LMStudioClassifier(
        fallback=RulesClassifier(),
        model=settings.lmstudio_model,
        base_url=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key,
        timeout_seconds=settings.lmstudio_timeout_seconds,
    )


async def evaluate(mode: str, dataset: list[tuple[str, Category]]) -> dict:
    from app.services.lmstudio_classifier import LMStudioClassifier

    classifier = get_classifier(mode)
    records: list[EvalRecord] = []

    for text, actual in dataset:
        started_at = time.perf_counter()
        if isinstance(classifier, LMStudioClassifier):
            result, classifier_used, ok, error_message = await classifier.classify_with_details(
                text
            )
        else:
            result = await classifier.classify(text)
            classifier_used = "rules"
            ok = True
            error_message = None
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        records.append(
            EvalRecord(
                text=text,
                actual=actual,
                predicted=result.category,
                confidence=result.confidence,
                latency_ms=latency_ms,
                classifier_used=classifier_used,
                ok=ok,
                error_message=error_message,
            )
        )

    return compute_metrics(records, mode)


def compute_metrics(records: list[EvalRecord], mode: str) -> dict:
    total = len(records)
    correct = sum(1 for record in records if record.actual == record.predicted)
    accuracy = safe_div(correct, total)

    per_category: dict[str, dict[str, float | int]] = {}
    for label in CATEGORIES:
        tp = sum(1 for r in records if r.actual == label and r.predicted == label)
        fp = sum(1 for r in records if r.actual != label and r.predicted == label)
        fn = sum(1 for r in records if r.actual == label and r.predicted != label)
        support = sum(1 for r in records if r.actual == label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_category[label] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    macro_f1 = safe_div(
        sum(float(per_category[label]["f1"]) for label in CATEGORIES),
        len(CATEGORIES),
    )

    confusion: dict[str, dict[str, int]] = {}
    for actual in CATEGORIES:
        confusion[actual] = {}
        for predicted in CATEGORIES:
            confusion[actual][predicted] = sum(
                1 for r in records if r.actual == actual and r.predicted == predicted
            )

    avg_latency_ms = safe_div(sum(r.latency_ms for r in records), total)
    avg_confidence = safe_div(sum(r.confidence for r in records), total)
    ok_rate = safe_div(sum(1 for r in records if r.ok), total)
    fallback_rate = safe_div(sum(1 for r in records if r.classifier_used == "rules"), total)
    lmstudio_rate = safe_div(sum(1 for r in records if r.classifier_used == "lmstudio"), total)

    errors = [
        {
            "text": r.text,
            "actual": r.actual,
            "predicted": r.predicted,
            "error_message": r.error_message,
        }
        for r in records
        if r.error_message
    ][:10]

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "total_examples": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "avg_confidence": round(avg_confidence, 4),
        "ok_rate": round(ok_rate, 4),
        "fallback_rate": round(fallback_rate, 4),
        "lmstudio_used_rate": round(lmstudio_rate, 4),
        "per_category": per_category,
        "confusion_matrix": confusion,
        "sample_errors": errors,
    }


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def default_output_path(mode: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / "eval" / f"{mode}_{timestamp}.json"


def save_results(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


async def main_async() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)
    result = await evaluate(args.mode, dataset)

    output_path = Path(args.output) if args.output else default_output_path(args.mode)
    save_results(output_path, result)

    print(json.dumps(result, indent=2))
    print(f"\nSaved evaluation report: {output_path}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
