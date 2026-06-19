from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from .image_tools import encode_image_for_gemini


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None, timeout: int = 120, rpm_sleep: float = 0.0) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY. Use --provider mock only for smoke tests.")
        self.timeout = timeout
        self.rpm_sleep = rpm_sleep

    def complete(self, prompt: str, images: list[tuple[str, Path]], temperature: float = 0.0) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for image_id, path in images:
            if not path.exists():
                continue
            parts.append({"text": f"Image ID: {image_id}"})
            parts.append({"inline_data": encode_image_for_gemini(path)})

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                resp = requests.post(url, json=body, timeout=self.timeout)
                if resp.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0].get("text", "")
                if self.rpm_sleep:
                    time.sleep(self.rpm_sleep)
                return _extract_json(text)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Gemini call failed after retries: {last_err}")


class MockClient:
    """Offline smoke-test client. It does NOT inspect images; never use for final submission."""

    def complete(self, prompt: str, images: list[tuple[str, Path]], temperature: float = 0.0) -> dict[str, Any]:  # noqa: ARG002
        # Conservative fallback so generated files are schema-valid but not competitive.
        return {
            "extracted_claim": "unknown",
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "Mock mode cannot inspect visual damage evidence.",
            "risk_flags": ["manual_review_required"],
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "Offline mock mode does not perform visual review; use Gemini for final predictions.",
            "supporting_image_ids": [],
            "valid_image": bool(images),
            "severity": "unknown",
            "confidence": 0.0,
        }


def make_client(provider: str, model: str, rpm_sleep: float = 0.0):
    provider = (provider or "gemini").lower()
    if provider == "mock":
        return MockClient()
    if provider == "gemini":
        return GeminiClient(model=model, rpm_sleep=rpm_sleep)
    raise ValueError(f"Unsupported provider: {provider}. Use gemini or mock.")
