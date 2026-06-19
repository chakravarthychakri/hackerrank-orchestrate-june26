from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # Engine handles CRLF and quoted conversations reliably.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_history(path: Path) -> dict[str, dict[str, str]]:
    df = read_csv_safe(path)
    if df.empty or "user_id" not in df.columns:
        return {}
    return {str(row["user_id"]): {k: str(v) for k, v in row.items()} for _, row in df.iterrows()}


def load_requirements(path: Path) -> list[dict[str, str]]:
    df = read_csv_safe(path)
    if df.empty:
        return []
    return [{k: str(v) for k, v in row.items()} for _, row in df.iterrows()]


def split_semicolon(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def image_id_from_path(path_like: str) -> str:
    return Path(path_like).stem or str(path_like)


def resolve_image_paths(repo_root: Path, image_paths_value: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for raw in split_semicolon(image_paths_value):
        p = Path(raw)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        candidates.append(repo_root / p)
        candidates.append(repo_root / "dataset" / p)
        # Useful when caller points repo_root at dataset/ directly.
        candidates.append(repo_root / p.name)
        found = next((c for c in candidates if c.exists()), candidates[0])
        out.append((image_id_from_path(raw), found))
    return out


def compact_requirements(requirements: list[dict[str, str]], claim_object: str, user_claim: str) -> str:
    claim_object = (claim_object or "").lower()
    claim_text = (user_claim or "").lower()
    chosen = []
    for r in requirements:
        obj = r.get("claim_object", "").lower()
        applies = r.get("applies_to", "").lower()
        if obj not in {"all", claim_object}:
            continue
        if obj == "all" or any(tok in claim_text for tok in applies.replace(",", " ").replace("or", " ").split() if len(tok) > 3):
            chosen.append(r)
    if not chosen:
        chosen = [r for r in requirements if r.get("claim_object", "").lower() in {"all", claim_object}]
    lines = []
    for r in chosen[:8]:
        lines.append(f"- {r.get('requirement_id','')}: {r.get('minimum_image_evidence','')}")
    return "\n".join(lines) if lines else "- Claimed object and relevant part must be clearly visible enough to evaluate."


def make_cache_key(row: dict[str, Any], image_paths: list[tuple[str, Path]], strategy: str, model: str) -> str:
    payload = {
        "row": row,
        "strategy": strategy,
        "model": model,
        "images": [
            {"id": iid, "path": str(path), "mtime": path.stat().st_mtime if path.exists() else None, "size": path.stat().st_size if path.exists() else None}
            for iid, path in image_paths
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv_exact(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def image_count_in_csv(df: pd.DataFrame) -> int:
    if "image_paths" not in df.columns:
        return 0
    return int(df["image_paths"].apply(lambda x: len(split_semicolon(x))).sum())
