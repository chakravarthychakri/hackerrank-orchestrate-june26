from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import EvidencePipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-modal evidence review for HackerRank Orchestrate damage claims.")
    p.add_argument("--repo_root", type=Path, default=Path("."), help="Repository root containing dataset/ and problem_statement.md")
    p.add_argument("--claims_csv", type=Path, default=None, help="CSV to process. Defaults to <repo_root>/dataset/claims.csv")
    p.add_argument("--output_csv", type=Path, default=None, help="Output path. Defaults to <repo_root>/output.csv")
    p.add_argument("--user_history_csv", type=Path, default=None, help="Defaults to <repo_root>/dataset/user_history.csv")
    p.add_argument("--evidence_requirements_csv", type=Path, default=None, help="Defaults to <repo_root>/dataset/evidence_requirements.csv")
    p.add_argument("--provider", choices=["gemini", "mock"], default="gemini")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--strategy", choices=["strict", "balanced"], default="strict")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_images", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Debug only: limit rows")
    p.add_argument("--cache_dir", type=Path, default=None)
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--log_path", type=Path, default=None, help="Defaults to ~/hackerrank_orchestrate/log.txt")
    p.add_argument("--rpm_sleep", type=float, default=0.0, help="Seconds to sleep after each model call for rate-limit safety")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    claims_csv = args.claims_csv or repo_root / "dataset" / "claims.csv"
    output_csv = args.output_csv or repo_root / "output.csv"

    pipeline = EvidencePipeline(
        repo_root=repo_root,
        claims_csv=claims_csv,
        output_csv=output_csv,
        user_history_csv=args.user_history_csv,
        evidence_requirements_csv=args.evidence_requirements_csv,
        provider=args.provider,
        model=args.model,
        strategy=args.strategy,
        temperature=args.temperature,
        cache_dir=args.cache_dir,
        log_path=args.log_path,
        max_images=args.max_images,
        rpm_sleep=args.rpm_sleep,
        no_cache=args.no_cache,
    )
    pipeline.run(limit=args.limit)


if __name__ == "__main__":
    main()
