# Evaluation Report — Multi-Modal Evidence Review

This file is included with the runnable evaluation workflow required by the challenge. Run the command below inside the cloned repo to regenerate the real metrics from `dataset/sample_claims.csv`:

```bash
python code/evaluation/main.py --repo_root . --provider gemini --model gemini-2.5-flash
```

## Strategies compared

### Strategy A: strict visual-evidence verifier
- Conservative evidence standard.
- Returns `not_enough_information` when the relevant part, issue, or object identity is not clearly visible.
- Best for avoiding false support decisions.

### Strategy B: balanced visual-evidence verifier
- Uses the strongest relevant image when the claimed object and part are visible enough.
- Less likely to reject because of minor blur or imperfect framing.
- Best for practical human-style review when the image is still readable.

## Final strategy

Default final strategy: **strict**.

Reason: for damage-claim verification, an unsupported approval is more harmful than a conservative request for more evidence. The problem statement also says images are the primary source of truth, so the final strategy prioritizes visible proof over conversation/history assumptions.

## Operational analysis

- Expected sample rows: 20.
- Expected test rows: 44.
- Expected sample image references: 29.
- Expected test image references: 82.
- Model calls: one multimodal call per claim row.
- Evaluation calls: two strategies × sample rows = about 40 calls.
- Final prediction calls: about 44 calls.
- Total development run: about 84 model calls if both evaluation strategies and final output are run once.
- Token usage estimate: roughly 1.5k–3k text-equivalent input tokens per row, plus provider-specific image token accounting; roughly 150–350 output tokens per row.
- Runtime estimate: several minutes sequentially, depending on model latency and image upload size.
- Cost control: local JSON cache under `.cache/`, max image count per row, image resizing to max side 1600px, temperature 0, retry logic for 429/5xx, and optional `--rpm_sleep` throttling.
- TPM/RPM: if the provider rate-limits, increase `--rpm_sleep` or process sample/final in separate runs.

## Failure modes handled

- Missing/unreadable images → `not_enough_information`.
- Bad quality images → allowed quality risk flags.
- User-history risk → added to `risk_flags`, never used alone to contradict visual proof.
- Prompt-injection text in claim conversation → `text_instruction_present` and `manual_review_required`.
- Invalid model outputs → normalized to allowed values.
- Bad supporting image IDs → removed unless they match submitted image filename stems.
