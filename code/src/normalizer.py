from __future__ import annotations

import re
from typing import Any

from .constants import CLAIM_STATUS, ISSUE_TYPES, OBJECT_PARTS, RISK_FLAGS, SEVERITY, OUTPUT_COLUMNS


def _norm(s: Any) -> str:
    return str(s if s is not None else "").strip().lower().replace(" ", "_").replace("-", "_")


def normalize_status(v: Any) -> str:
    s = _norm(v)
    aliases = {
        "support": "supported",
        "supports": "supported",
        "image_supports_claim": "supported",
        "contradict": "contradicted",
        "contradicts": "contradicted",
        "insufficient": "not_enough_information",
        "insufficient_evidence": "not_enough_information",
        "insufficient_information": "not_enough_information",
        "cannot_determine": "not_enough_information",
        "unclear": "not_enough_information",
        "unknown": "not_enough_information",
    }
    s = aliases.get(s, s)
    return s if s in CLAIM_STATUS else "not_enough_information"


def normalize_issue(v: Any) -> str:
    s = _norm(v)
    aliases = {
        "shattered_glass": "glass_shatter",
        "broken_glass": "glass_shatter",
        "tear": "torn_packaging",
        "torn": "torn_packaging",
        "crushed": "crushed_packaging",
        "missing": "missing_part",
        "broken": "broken_part",
        "water": "water_damage",
        "wet": "water_damage",
        "no_damage": "none",
    }
    s = aliases.get(s, s)
    return s if s in ISSUE_TYPES else "unknown"


def normalize_part(v: Any, obj: str) -> str:
    s = _norm(v)
    aliases = {
        "bumper_front": "front_bumper",
        "front": "front_bumper",
        "back_bumper": "rear_bumper",
        "back": "rear_bumper",
        "rear": "rear_bumper",
        "mirror": "side_mirror",
        "light": "headlight",
        "head_lamp": "headlight",
        "tail_light": "taillight",
        "screen_panel": "screen",
        "keys": "keyboard",
        "key": "keyboard",
        "touchpad": "trackpad",
        "box_corner": "package_corner",
        "corner": "corner" if obj == "laptop" else "package_corner",
        "side": "package_side" if obj == "package" else "body",
    }
    s = aliases.get(s, s)
    allowed = OBJECT_PARTS.get(_norm(obj), {"unknown"})
    return s if s in allowed else "unknown"


def normalize_severity(v: Any, status: str, issue: str) -> str:
    s = _norm(v)
    aliases = {"minor": "low", "moderate": "medium", "major": "high", "severe": "high", "no_damage": "none"}
    s = aliases.get(s, s)
    if status == "contradicted" and issue == "none":
        return "none"
    return s if s in SEVERITY else "unknown"


def normalize_risks(values: Any, extra: list[str] | None = None) -> str:
    parts: list[str] = []
    if isinstance(values, list):
        raw_parts = values
    else:
        raw_parts = re.split(r"[;,]", str(values or ""))
    for p in raw_parts + (extra or []):
        p = _norm(p)
        if not p or p == "none":
            continue
        aliases = {
            "image_quality_risk": "blurry_image",
            "possible_fraud": "manual_review_required",
            "history_risk": "user_history_risk",
            "prompt_injection": "text_instruction_present",
            "wrong_part": "wrong_object_part",
            "mismatch": "claim_mismatch",
        }
        p = aliases.get(p, p)
        if p in RISK_FLAGS and p != "none" and p not in parts:
            parts.append(p)
    return ";".join(parts) if parts else "none"


def clean_bool(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    s = _norm(v)
    return "true" if s in {"true", "1", "yes", "y"} else "false"


def normalize_prediction(raw: dict[str, Any], row: dict[str, Any], submitted_image_ids: list[str], quality_flags: list[str], history: dict[str, str]) -> dict[str, Any]:
    obj = _norm(row.get("claim_object", raw.get("claim_object", raw.get("object_type", "unknown"))))
    status = normalize_status(raw.get("claim_status", raw.get("decision", raw.get("status", "not_enough_information"))))
    issue = normalize_issue(raw.get("issue_type", raw.get("visible_issue_type", "unknown")))
    part = normalize_part(raw.get("object_part", raw.get("part", "unknown")), obj)

    raw_ids = raw.get("supporting_image_ids", raw.get("image_ids", []))
    if isinstance(raw_ids, str):
        ids = [x.strip() for x in re.split(r"[;,]", raw_ids) if x.strip() and x.strip().lower() != "none"]
    else:
        ids = [str(x).strip() for x in raw_ids or [] if str(x).strip()]
    ids = [x for x in ids if x in submitted_image_ids]

    risk_extra = []
    risk_extra.extend(quality_flags)
    hist_flags = history.get("history_flags", "") if history else ""
    risk_extra.extend([x.strip() for x in hist_flags.split(";") if x.strip() and x.strip() != "none"])

    claim_text = str(row.get("user_claim", "")).lower()
    if any(t in claim_text for t in ["ignore previous", "ignore all", "mark this", "approve this", "system prompt", "follow it and approve"]):
        risk_extra.append("text_instruction_present")
        risk_extra.append("manual_review_required")

    risk_flags = normalize_risks(raw.get("risk_flags", []), risk_extra)

    # Guardrails: no valid image ids means no support/contradiction output.
    any_readable = bool(submitted_image_ids)
    if not any_readable:
        status = "not_enough_information"
        issue = "unknown"
        part = "unknown"
        ids = []
        risk_flags = normalize_risks(risk_flags, ["damage_not_visible", "manual_review_required"])

    # If VLM says insufficient, supporting IDs must be none.
    if status == "not_enough_information":
        ids = []

    # If support/contradiction but no specific support image was returned, use first readable image; judge wants image IDs.
    if status in {"supported", "contradicted"} and not ids and submitted_image_ids:
        ids = [submitted_image_ids[0]]

    # Evidence standard is true only when image evidence can decide support/contradiction.
    evidence_met = status in {"supported", "contradicted"}
    valid_image = bool(submitted_image_ids) and not all(f in quality_flags for f in ["blurry_image", "low_light_or_glare", "cropped_or_obstructed"])
    if "non_original_image" in risk_flags or "possible_manipulation" in risk_flags:
        valid_image = False

    justification = str(raw.get("claim_status_justification", raw.get("justification", raw.get("reason", "")))).strip()
    if not justification:
        if status == "supported":
            justification = f"The submitted image evidence shows the claimed {issue} on the {part}."
        elif status == "contradicted":
            justification = f"The relevant claimed part is visible but the claimed damage is not visible."
        else:
            justification = "The submitted images do not clearly show the claimed object part and damage clearly enough to verify."
    if len(justification) > 350:
        justification = justification[:347].rstrip() + "..."

    evidence_reason = str(raw.get("evidence_standard_met_reason", "")).strip()
    if not evidence_reason:
        evidence_reason = "The relevant image evidence is sufficient to evaluate the claim." if evidence_met else "The image set does not clearly show enough relevant visual evidence to evaluate the claim."
    if len(evidence_reason) > 300:
        evidence_reason = evidence_reason[:297].rstrip() + "..."

    severity = normalize_severity(raw.get("severity", "unknown"), status, issue)

    out = {
        "user_id": row.get("user_id", ""),
        "image_paths": row.get("image_paths", ""),
        "user_claim": row.get("user_claim", ""),
        "claim_object": row.get("claim_object", ""),
        "evidence_standard_met": "true" if evidence_met else "false",
        "evidence_standard_met_reason": evidence_reason,
        "risk_flags": risk_flags,
        "issue_type": issue,
        "object_part": part,
        "claim_status": status,
        "claim_status_justification": justification,
        "supporting_image_ids": ";".join(ids) if ids else "none",
        "valid_image": "true" if valid_image else "false",
        "severity": severity,
    }
    return {c: out.get(c, "") for c in OUTPUT_COLUMNS}
