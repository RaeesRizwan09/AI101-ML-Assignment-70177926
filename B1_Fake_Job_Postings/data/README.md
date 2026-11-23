# Data — B1: Detecting Fraudulent Job Postings

## What the dataset is

**Real or Fake Job Posting Prediction (EMSCAD)** — 17,880 job advertisements, 866 of them
fraudulent (4.84 %). Each row has five free-text fields (`title`, `company_profile`,
`description`, `requirements`, `benefits`), seven categorical fields, three binary flags and a
`salary_range`.

* Original Kaggle release (assembled for the EMSCAD study):
  <https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction>
* Hugging Face mirror used here (no Kaggle account needed, licence CC0):
  <https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction>

Both contain the same `fake_job_postings.csv` (50 MB, 17,880 × 18).

## How to obtain it

Option 1 — Hugging Face (what the notebook was run with):

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("victor/real-or-fake-fake-jobposting-prediction",
                    "fake_job_postings.csv", repo_type="dataset")
print(p)   # .../fake_job_postings.csv
PY
```

Option 2 — Kaggle CLI (needs an account and `~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction -p data/ --unzip
```

Then point the notebook at the file. The loading cell looks for `fake_job_postings.csv` in this
order and uses the first hit:

1. `$AI101_DATA` — a directory of your choice (recommended: `export AI101_DATA=~/ai101_data`)
2. `<task folder>/data/`
3. `<repo root>/data/`
4. `/mnt/DC5E14885E145D98/ai101_ws/data/` — the machine the notebook was executed on

If none exists the cell stops with `FileNotFoundError: fake_job_postings.csv not found - see
data/README.md` instead of silently training on nothing.

## Verification

* `df.shape == (17880, 18)`, `df["fraudulent"].sum() == 866` → 4.84 % positives (1 : 20).
* Columns: `job_id`, `title`, `location`, `department`, `salary_range`, `company_profile`,
  `description`, `requirements`, `benefits`, `telecommuting`, `has_company_logo`,
  `has_questions`, `employment_type`, `required_experience`, `required_education`, `industry`,
  `function`, `fraudulent`.

## Why this file is not committed

The CSV is 50 MB and the repository must not contain datasets or files larger than 50 MB, so
`data/` and `*.csv` are ignored by `.gitignore`. Download it with the snippet above; the
notebook rebuilds every table, figure and number from it.
