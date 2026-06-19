from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_log_path() -> Path:
    return Path.home() / "hackerrank_orchestrate" / "log.txt"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def redact(text: str) -> str:
    # Basic protection for accidental key leakage.
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        val = os.environ.get(key)
        if val:
            text = text.replace(val, f"<{key}:REDACTED>")
    return text


class RunLogger:
    def __init__(self, path: Path | None = None, enabled: bool = True) -> None:
        self.path = path or default_log_path()
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, title: str, payload: Any) -> None:
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(f"\n## [{now_iso()}] {title}\n")
            if isinstance(payload, str):
                f.write(redact(payload) + "\n")
            else:
                f.write(redact(json.dumps(payload, ensure_ascii=False, indent=2, default=str)) + "\n")

    def session_start(self, repo_root: Path, language: str = "py") -> None:
        self.append("SESSION START", {
            "Agent": "DamageEvidencePipeline",
            "Repo Root": str(repo_root.resolve()),
            "Branch": "unknown",
            "Worktree": "main",
            "Parent Agent": "ChatGPT",
            "Language": language,
            "System": platform.platform(),
        })
