from __future__ import annotations

import json
from typing import Any

from .constants import ISSUE_TYPES, RISK_FLAGS, SEVERITY, CLAIM_STATUS, OBJECT_PARTS


def build_prompt(*, row: dict[str, Any], user_history: dict[str, str], requirements_text: str, image_meta: list[dict[str, Any]], strategy: str) -> str:
    strict_rule = ""
    if strategy == "strict":
        strict_rule = """
Decision strictness:
- Prefer not_enough_information when the claimed part is not clearly visible, angle is wrong, image is too blurry/dark, or the claim is about contents/missing part that cannot be verified visually.
- Use supported only when the claimed object, claimed part, and claimed visible issue are all visible.
- Use contradicted only when the relevant claimed part is visible clearly enough and the claimed damage is absent or visibly different.
"""
    else:
        strict_rule = """
Decision strictness:
- Use the best available relevant image when at least one image clearly shows the claimed object/part.
- Do not over-penalize minor blur if damage is still visible.
- Still use not_enough_information for wrong object, wrong part, missing contents not visible, or unusable images.
"""

    return f"""You are a multimodal insurance evidence reviewer. Inspect the submitted images as the primary source of truth.

Task:
1. Extract the concrete damage claim from the conversation.
2. Inspect each image independently. Image IDs are shown before each image.
3. Decide whether the visual evidence supports, contradicts, or lacks enough information for the claim.
4. Apply user history only as risk context. It must not override clear visual evidence.
5. Return ONLY valid JSON. No markdown.

Claim row:
{json.dumps(row, ensure_ascii=False, indent=2)}

User history for this user:
{json.dumps(user_history or {'history_flags':'none'}, ensure_ascii=False, indent=2)}

Relevant minimum evidence requirements:
{requirements_text}

Pre-computed image quality metadata:
{json.dumps(image_meta, ensure_ascii=False, indent=2)}

Allowed values:
claim_status: {sorted(CLAIM_STATUS)}
issue_type: {sorted(ISSUE_TYPES)}
car object_part: {sorted(OBJECT_PARTS['car'])}
laptop object_part: {sorted(OBJECT_PARTS['laptop'])}
package object_part: {sorted(OBJECT_PARTS['package'])}
risk_flags: {sorted(RISK_FLAGS)}
severity: {sorted(SEVERITY)}

{strict_rule}

Important rules:
- Output issue_type=none only when the relevant claimed part is visible and no damage/issue is present.
- Output unknown when issue or part cannot be determined.
- supporting_image_ids must be filename stems such as img_1, img_2. Use only submitted image IDs. Use [] when not_enough_information.
- evidence_standard_met should be true only if image evidence is sufficient to evaluate the claim as supported or contradicted.
- valid_image should be false if images are missing, unreadable, too blurry/dark/cropped for automated review, screenshots/non-original evidence, or possible manipulation.
- Risk flags should be semicolon-free array items. Do not invent flags outside the allowed list.
- Keep reasons concise and grounded in the images.

Return JSON object with exactly these keys:
{{
  "extracted_claim": "short concrete claim",
  "evidence_standard_met": true,
  "evidence_standard_met_reason": "short visual-evidence reason",
  "risk_flags": ["none"],
  "issue_type": "dent",
  "object_part": "front_bumper",
  "claim_status": "supported",
  "claim_status_justification": "short image-grounded reason mentioning image ids when useful",
  "supporting_image_ids": ["img_1"],
  "valid_image": true,
  "severity": "low",
  "confidence": 0.0
}}
"""
