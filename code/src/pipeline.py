from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import OUTPUT_COLUMNS
from .image_tools import assess_image
from .io_utils import (
    compact_requirements,
    ensure_parent,
    load_history,
    load_requirements,
    make_cache_key,
    read_csv_safe,
    resolve_image_paths,
    write_csv_exact,
)
from .logger import RunLogger, default_log_path
from .model_clients import make_client
from .normalizer import normalize_prediction
from .prompts import build_prompt


class EvidencePipeline:
    def __init__(
        self,
        *,
        repo_root: Path,
        claims_csv: Path,
        output_csv: Path,
        user_history_csv: Path | None = None,
        evidence_requirements_csv: Path | None = None,
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
        strategy: str = "strict",
        temperature: float = 0.0,
        cache_dir: Path | None = None,
        log_path: Path | None = None,
        max_images: int = 4,
        rpm_sleep: float = 0.0,
        no_cache: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.claims_csv = claims_csv
        self.output_csv = output_csv
        self.user_history_csv = user_history_csv or self.repo_root / "dataset" / "user_history.csv"
        self.evidence_requirements_csv = evidence_requirements_csv or self.repo_root / "dataset" / "evidence_requirements.csv"
        self.provider = provider
        self.model = model
        self.strategy = strategy
        self.temperature = temperature
        self.cache_dir = cache_dir or self.repo_root / ".cache" / "damage_evidence"
        self.log = RunLogger(log_path or default_log_path())
        self.max_images = max_images
        self.no_cache = no_cache
        self.client = make_client(provider, model, rpm_sleep=rpm_sleep)

    def run(self, limit: int | None = None) -> pd.DataFrame:
        self.log.session_start(self.repo_root)
        df = read_csv_safe(self.claims_csv)
        if df.empty:
            raise FileNotFoundError(f"No rows found in {self.claims_csv}")
        if limit:
            df = df.head(limit).copy()

        history_map = load_history(self.user_history_csv)
        requirements = load_requirements(self.evidence_requirements_csv)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.log.append("RUN_CONFIG", {
            "claims_csv": str(self.claims_csv),
            "output_csv": str(self.output_csv),
            "user_history_csv": str(self.user_history_csv),
            "evidence_requirements_csv": str(self.evidence_requirements_csv),
            "rows": len(df),
            "provider": self.provider,
            "model": self.model,
            "strategy": self.strategy,
            "max_images": self.max_images,
        })

        outputs: list[dict[str, Any]] = []
        for idx, row_ser in df.iterrows():
            row = {k: str(v) for k, v in row_ser.to_dict().items()}
            image_paths = resolve_image_paths(self.repo_root, row.get("image_paths", ""))[: self.max_images]
            image_meta = []
            quality_flags: list[str] = []
            readable_images: list[tuple[str, Path]] = []
            for image_id, path in image_paths:
                q = assess_image(path)
                q["image_id"] = image_id
                image_meta.append(q)
                quality_flags.extend(q.get("flags", []))
                if q.get("exists") and q.get("readable"):
                    readable_images.append((image_id, path))

            history = history_map.get(row.get("user_id", ""), {})
            req_text = compact_requirements(requirements, row.get("claim_object", ""), row.get("user_claim", ""))
            cache_key = make_cache_key(row, readable_images, self.strategy, self.model)
            cache_path = self.cache_dir / f"{cache_key}.json"

            if not readable_images:
                raw = {
                    "evidence_standard_met": False,
                    "evidence_standard_met_reason": "No submitted image is available or readable for automated visual review.",
                    "risk_flags": ["damage_not_visible", "manual_review_required"],
                    "issue_type": "unknown",
                    "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "The referenced local image files could not be read, so the visual claim cannot be verified.",
                    "supporting_image_ids": [],
                    "valid_image": False,
                    "severity": "unknown",
                }
            elif cache_path.exists() and not self.no_cache:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                prompt = build_prompt(
                    row=row,
                    user_history=history,
                    requirements_text=req_text,
                    image_meta=image_meta,
                    strategy=self.strategy,
                )
                raw = self.client.complete(prompt, readable_images, temperature=self.temperature)
                if not self.no_cache:
                    cache_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

            out = normalize_prediction(
                raw,
                row,
                submitted_image_ids=[iid for iid, _ in readable_images],
                quality_flags=quality_flags,
                history=history,
            )
            outputs.append(out)

            self.log.append(f"CLAIM ROW {idx + 1}", {
                "user_id": row.get("user_id"),
                "claim_object": row.get("claim_object"),
                "image_ids": [iid for iid, _ in readable_images],
                "image_meta": image_meta,
                "history_flags": history.get("history_flags", "none"),
                "raw_model_output": raw,
                "final_output": out,
            })
            print(f"[{idx + 1}/{len(df)}] {row.get('user_id')} -> {out['claim_status']} / {out['issue_type']} / {out['object_part']}")

        write_csv_exact(outputs, self.output_csv, OUTPUT_COLUMNS)
        self.log.append("OUTPUT WRITTEN", {"path": str(self.output_csv), "rows": len(outputs), "columns": OUTPUT_COLUMNS})
        return pd.DataFrame(outputs, columns=OUTPUT_COLUMNS)
