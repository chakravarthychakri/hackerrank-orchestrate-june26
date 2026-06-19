from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Make code/src importable when run as python code/evaluation/main.py
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.constants import OUTPUT_COLUMNS  # noqa: E402
from src.io_utils import image_count_in_csv, read_csv_safe  # noqa: E402
from src.pipeline import EvidencePipeline  # noqa: E402


def exact_match(a: Any, b: Any) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def score(gold: pd.DataFrame, pred: pd.DataFrame) -> dict[str, Any]:
    n = min(len(gold), len(pred))
    out: dict[str, Any] = {"rows_gold": len(gold), "rows_pred": len(pred), "rows_scored": n}
    if n == 0:
        return out
    for col in ["claim_status", "issue_type", "object_part", "severity", "evidence_standard_met", "valid_image"]:
        if col in gold.columns and col in pred.columns:
            out[f"{col}_accuracy"] = round(sum(exact_match(gold.iloc[i][col], pred.iloc[i][col]) for i in range(n)) / n, 4)
    key_cols = [c for c in ["claim_status", "issue_type", "object_part", "severity"] if c in gold.columns and c in pred.columns]
    if key_cols:
        out["core_joint_accuracy"] = round(sum(all(exact_match(gold.iloc[i][c], pred.iloc[i][c]) for c in key_cols) for i in range(n)) / n, 4)
    return out


def run_strategy(args: argparse.Namespace, strategy: str, out_path: Path) -> pd.DataFrame:
    pipe = EvidencePipeline(
        repo_root=args.repo_root.resolve(),
        claims_csv=args.sample_csv,
        output_csv=out_path,
        user_history_csv=args.user_history_csv,
        evidence_requirements_csv=args.evidence_requirements_csv,
        provider=args.provider,
        model=args.model,
        strategy=strategy,
        temperature=0.0,
        max_images=args.max_images,
        cache_dir=args.repo_root / ".cache" / f"eval_{strategy}",
        log_path=args.log_path,
        rpm_sleep=args.rpm_sleep,
        no_cache=args.no_cache,
    )
    return pipe.run(limit=args.limit)


def write_report(path: Path, *, strict_metrics: dict[str, Any], balanced_metrics: dict[str, Any], sample_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace) -> None:
    sample_images = image_count_in_csv(sample_df)
    test_images = image_count_in_csv(test_df)
    test_rows = len(test_df)
    sample_rows = len(sample_df)
    # Conservative estimate: prompt + schema + metadata + conversations + visual tokens vary by provider.
    avg_input_tokens = 2200
    avg_output_tokens = 220
    total_calls = sample_rows * 2 + test_rows
    full_test_calls = test_rows
    report = f"""# Evaluation Report — Multi-Modal Evidence Review

Generated: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}

## Dataset summary

| Split | Rows | Image references |
|---|---:|---:|
| sample_claims.csv | {sample_rows} | {sample_images} |
| claims.csv | {test_rows} | {test_images} |

## Strategies compared

### Strategy A: strict visual-evidence verifier
- Uses conservative prompting.
- Returns `not_enough_information` when the relevant object part is not clearly visible, the angle is wrong, or the image quality is not enough.
- Best for avoiding false support.

Metrics on sample:

```json
{json.dumps(strict_metrics, indent=2)}
```

### Strategy B: balanced visual-evidence verifier
- Uses the best relevant submitted image when the object and part are visible enough.
- Less likely to over-reject minor blur or partial context.
- Best when sample labels reward practical human-level review.

Metrics on sample:

```json
{json.dumps(balanced_metrics, indent=2)}
```

## Final strategy

Default final strategy: **strict**.

Reason: in claim verification, a false `supported` decision is worse than a conservative `not_enough_information`. Images are the primary source of truth, and history only adds risk context.

## Operational analysis

- Provider used: `{args.provider}`
- Model used: `{args.model}`
- Approximate model calls for evaluation: `{sample_rows * 2}` calls because two strategies are compared on sample.
- Approximate model calls for final test output: `{full_test_calls}` calls, one multimodal call per claim row.
- Approximate images processed for final test output: `{test_images}` image references.
- Approximate input tokens per claim: `{avg_input_tokens}` text-equivalent tokens, excluding provider-specific image token accounting.
- Approximate output tokens per claim: `{avg_output_tokens}`.
- Approximate full test text tokens: `{test_rows * (avg_input_tokens + avg_output_tokens)}` text-equivalent tokens, plus image tokens.
- Approximate runtime: depends on model latency; with 44 test rows, expect several minutes if calls are sequential.
- TPM/RPM handling: deterministic temperature 0, local JSON cache, retry on 429/5xx, and optional `--rpm_sleep` throttling.
- Cost: model-dependent. Cost should be computed from the current provider pricing multiplied by `{full_test_calls}` calls and `{test_images}` images. The pipeline avoids repeated cost through cache files under `.cache/`.

## Failure modes checked

- Missing/unreadable local images become `not_enough_information`.
- User history flags are added to `risk_flags` but do not override clear visual evidence.
- Prompt-injection-like text inside the user claim is flagged as `text_instruction_present` and `manual_review_required`.
- Invalid model values are normalized to allowed schema values.
- Supporting image IDs are restricted to submitted image filename stems.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the damage evidence pipeline on sample_claims.csv.")
    p.add_argument("--repo_root", type=Path, default=Path("."))
    p.add_argument("--sample_csv", type=Path, default=None)
    p.add_argument("--claims_csv", type=Path, default=None)
    p.add_argument("--user_history_csv", type=Path, default=None)
    p.add_argument("--evidence_requirements_csv", type=Path, default=None)
    p.add_argument("--provider", choices=["gemini", "mock"], default="gemini")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--max_images", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--rpm_sleep", type=float, default=0.0)
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--log_path", type=Path, default=None)
    p.add_argument("--report_path", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.sample_csv = args.sample_csv or args.repo_root / "dataset" / "sample_claims.csv"
    args.claims_csv = args.claims_csv or args.repo_root / "dataset" / "claims.csv"
    args.user_history_csv = args.user_history_csv or args.repo_root / "dataset" / "user_history.csv"
    args.evidence_requirements_csv = args.evidence_requirements_csv or args.repo_root / "dataset" / "evidence_requirements.csv"
    args.report_path = args.report_path or CODE_DIR / "evaluation" / "evaluation_report.md"

    eval_dir = CODE_DIR / "evaluation"
    strict_pred = run_strategy(args, "strict", eval_dir / "sample_predictions_strict.csv")
    balanced_pred = run_strategy(args, "balanced", eval_dir / "sample_predictions_balanced.csv")

    gold = read_csv_safe(args.sample_csv)
    strict_metrics = score(gold, strict_pred)
    balanced_metrics = score(gold, balanced_pred)
    test_df = read_csv_safe(args.claims_csv)

    write_report(args.report_path, strict_metrics=strict_metrics, balanced_metrics=balanced_metrics, sample_df=gold, test_df=test_df, args=args)
    print(json.dumps({"strict": strict_metrics, "balanced": balanced_metrics, "report": str(args.report_path)}, indent=2))


if __name__ == "__main__":
    main()
