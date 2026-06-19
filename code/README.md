# Damage Evidence Review Agent

Runnable solution for HackerRank Orchestrate June 2026: multi-modal evidence review for car, laptop, and package damage claims.

## What it does

- Reads `dataset/claims.csv`.
- Reads `dataset/user_history.csv` and applies history only as risk context.
- Reads `dataset/evidence_requirements.csv` and injects the relevant minimum evidence checklist into the review prompt.
- Loads local images referenced by `image_paths`.
- Uses a multimodal Gemini model to inspect images.
- Normalizes every prediction to the exact required schema and allowed values.
- Writes `output.csv` with the required 14 columns in order.
- Includes an evaluation workflow under `code/evaluation/`.
- Appends run logs to `~/hackerrank_orchestrate/log.txt` unless `--log_path` is provided.

## Install

From the repository root:

```bash
cd code
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

## API key

Use environment variables only. Do not paste keys into code or logs.

```bash
export GEMINI_API_KEY="your_key_here"
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_key_here"
```

## Run final predictions

From the repository root:

```bash
python code/main.py \
  --repo_root . \
  --claims_csv dataset/claims.csv \
  --output_csv output.csv \
  --provider gemini \
  --model gemini-2.5-flash \
  --strategy strict
```

## Evaluate on sample

```bash
python code/evaluation/main.py \
  --repo_root . \
  --provider gemini \
  --model gemini-2.5-flash
```

This writes:

- `code/evaluation/sample_predictions_strict.csv`
- `code/evaluation/sample_predictions_balanced.csv`
- `code/evaluation/evaluation_report.md`

## Smoke test without an API key

Mock mode only verifies file reading, schema, and CSV writing. It does not inspect images and should not be used for the final submission.

```bash
python code/main.py --repo_root . --provider mock --limit 2 --output_csv smoke_output.csv
```

## Submission checklist

Before uploading:

```bash
python code/evaluation/main.py --repo_root . --provider gemini --model gemini-2.5-flash
python code/main.py --repo_root . --claims_csv dataset/claims.csv --output_csv output.csv --provider gemini --model gemini-2.5-flash --strategy strict
python - <<'PY'
import pandas as pd
claims=pd.read_csv('dataset/claims.csv')
out=pd.read_csv('output.csv')
print('claims rows:', len(claims))
print('output rows:', len(out))
print('columns:', list(out.columns))
PY
zip -r code.zip code -x "*/.venv/*" "*/__pycache__/*" "*/.pytest_cache/*" "*/.cache/*" "*/dataset/*" "*/data/*"
```

Upload these three files:

1. `code.zip`
2. `output.csv`
3. `~/hackerrank_orchestrate/log.txt`

## Important design choices

- Images are treated as primary truth.
- User claim conversation defines what part and issue must be checked.
- User history is only risk context and cannot override clear visual evidence.
- The system is conservative: if the relevant part is not visible, it returns `not_enough_information` instead of guessing.
- All model outputs pass through deterministic schema normalization and guardrails.
