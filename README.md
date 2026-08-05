# diabetes-brfss-ml

Diabetes prediction on the **BRFSS 2015 Diabetes Health Indicators** dataset
([Kaggle: alexteboul](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)).

The raw dataset is heavily imbalanced (~86% negative / ~14% positive). This repo
handles the shared preprocessing step: **SMOTE-ENN resampling**, published as a
versioned **W&B artifact** so all three of us model on byte-identical data.

**Team:** Mahdi (resampling + artifact pipeline), Sadman, Abrar (modelling).

| Stage | Status |
| --- | --- |
| Data loading + validation | done |
| SMOTE-ENN resampling + W&B artifact | done |
| Modelling | not started — each person's `notebooks/<name>/` |

**Who does what:**

- **Everyone** does [1. Setup](#1-setup-everyone) once, right after cloning.
- **Mahdi** additionally does [2. Get the raw data](#2-get-the-raw-data-mahdi-only)
  and [3. Run the resampling](#3-run-the-resampling-mahdi).
- **Sadman & Abrar** skip straight to [4. Pull the datasets](#4-pull-the-datasets-sadman--abrar)
  after setup — no raw CSV, no resampling run needed.
- Everyone should read [5. Working agreement](#5-working-agreement) before
  reporting any model numbers — it covers a correctness caveat, not just process.

---

## 1. Setup (everyone)

Requires **Python 3.13**.

```bash
git clone git@github.com:sahil0319/diabetes-brfss-ml.git
cd diabetes-brfss-ml
python3.13 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
```

`requirements.txt` holds the direct dependencies at exact pinned versions.
`requirements.lock.txt` is the full `pip freeze` (every transitive dependency) —
use it instead if you need a bit-for-bit identical environment:

```bash
pip install -r requirements.lock.txt
```

### Weights & Biases

```bash
wandb login
```

Paste the key from [wandb.ai/authorize](https://wandb.ai/authorize).
Alternatively, copy `.env.example` to `.env` and set `WANDB_API_KEY` there —
`.env` is gitignored, so never commit a real key.

Log in with whichever account/entity has access to the shared
`diabetes-brfss-ml` W&B project — ask Mahdi to add you if you can't see it, or
if `use_artifact` later 404s.

Set `wandb.entity` in `configs/resample_smoteenn.yaml` to our shared team name
once we've created one; leaving it `null` uses your personal default entity, and
then the artifact won't be visible to the others.

### nbstripout — everyone must run this once, per clone

```bash
nbstripout --install --attributes .gitattributes
```

This installs a git filter that strips notebook **output cells** before they're
committed, so notebook diffs stay readable and the repo stays small with three
people committing notebooks independently.

It writes to `.git/config`, which is **not** part of the repo, so cloning never
sets it up for you — run it once per clone, even on a machine where you'd
already set it up for a different repo. Check it took with:

```bash
nbstripout --status
```

---

## 2. Get the raw data (Mahdi only)

> Pulling the finished datasets from W&B instead? Skip this section — go to
> [4. Pull the datasets](#4-pull-the-datasets-sadman--abrar).

Download `diabetes_binary_health_indicators_BRFSS2015.csv` from the Kaggle
dataset and put it in `data/raw/`:

```text
data/raw/diabetes_binary_health_indicators_BRFSS2015.csv
```

Grab the **imbalanced binary** file — 253,680 rows, 22 columns. Not
`diabetes_binary_5050split_...` (already balanced, defeats the point) and not
`diabetes_012_...` (three-class target; the loader will reject it).

CSVs under `data/` are gitignored — data moves between us as W&B artifacts, not
through git.

---

## 3. Run the resampling (Mahdi)

```bash
python scripts/run_resample.py --config configs/resample_smoteenn.yaml
```

This will:

1. Seed everything from `random_seed` in the config.
2. Load and validate the raw CSV (schema, dtypes, target is binary, no NaNs).
   Column names are lowercased on load, so `Diabetes_binary` → `diabetes_binary`.
3. **Split off a stratified holdout set from the raw data** (`data.test_size`,
   default 20%) — *before* any resampling touches anything. This holdout stays
   100% real, untouched rows.
4. Apply SMOTE-ENN to the remaining training rows only — SMOTE oversamples the
   minority class, then Edited Nearest Neighbours drops samples whose
   neighbours disagree with them. The holdout set never sees this.
5. Log the full config, before/after class distributions (W&B table + bar chart),
   and row counts.
6. Write `data/processed/diabetes_binary_smoteenn.csv` (resampled training data)
   and `data/processed/diabetes_binary_holdout_test.csv` (raw holdout).
7. Upload both as W&B artifacts: `brfss-smoteenn-resampled` (train) and
   `brfss-holdout-test` (evaluation).

**Why the split happens before resampling, not after:** SMOTE fabricates
synthetic minority rows and ENN deletes real "borderline" rows. A split taken
*after* resampling would leak synthetic rows into the test set and strip out
the hard real examples that make it a fair test — see the
[caveat below](#a-caveat-worth-knowing-before-you-report-numbers) for the full
explanation.

### Useful flags

```bash
python scripts/run_resample.py --config configs/resample_smoteenn.yaml --wandb-mode disabled
```

`--wandb-mode` overrides the config (`online` / `offline` / `disabled`) — handy
for testing the pipeline without creating runs.

### On runtime

SMOTE-ENN on the full 253,680 rows is the slow part. SMOTE first grows the data
to ~440k rows, then ENN runs a k-nearest-neighbour search across all of it.
Expect this to take a while (minutes to tens of minutes depending on your
machine) and to hold a few GB of RAM. For a quick end-to-end check first, set
`data.subsample_n` in the config to something like `20000` — it takes a
stratified subsample of the raw rows before the split/resample. Set it back to
`null` for the real run.

---

## 4. Pull the datasets (Sadman & Abrar)

**You do not need to run the resampling step, and you don't need the raw CSV.**
Make sure you've done [1. Setup](#1-setup-everyone) first (venv, `wandb login`,
`nbstripout --install`) — that's all the setup this section needs.

Two artifacts get published each run — pull both:

| Artifact | Contents | Use it for |
| --- | --- | --- |
| `brfss-smoteenn-resampled` | SMOTE-ENN'd training data | Fitting your model |
| `brfss-holdout-test` | Untouched raw rows, never seen by SMOTE-ENN | Reporting headline metrics |

Do **not** report headline metrics on a split of `brfss-smoteenn-resampled` —
see the [caveat below](#a-caveat-worth-knowing-before-you-report-numbers) for
why that would be misleading.

Open your own notebook — `notebooks/sadman/train_model_v1.ipynb` or
`notebooks/abrar/train_model_v1.ipynb` — and run the first cell. It does this:

```python
import wandb, pandas as pd
from pathlib import Path

run = wandb.init(project="diabetes-brfss-ml", job_type="train")

train_dir = Path(run.use_artifact("brfss-smoteenn-resampled:latest").download())
train_df = pd.read_csv(next(train_dir.glob("*.csv")))

holdout_dir = Path(run.use_artifact("brfss-holdout-test:latest").download())
holdout_df = pd.read_csv(next(holdout_dir.glob("*.csv")))
```

A few things worth knowing:

- `:latest` resolves to the newest version. Once you're producing results you
  want to be able to reproduce, pin an explicit version instead —
  `brfss-smoteenn-resampled:v0` / `brfss-holdout-test:v0` — so a re-run of the
  resampling step doesn't silently change the data under your model. The two
  artifacts are versioned together (produced by the same run), so keep their
  version numbers matched.
- If W&B can't find an artifact, it's almost certainly the entity: qualify it
  fully as `"<entity>/diabetes-brfss-ml/brfss-holdout-test:latest"`.
- The download is cached in `artifacts/` (gitignored), so it only pulls once.

---

## 5. Working agreement

**Each person works only inside their own `notebooks/<name>/` folder** — that
way three people committing notebooks never touch the same file, and we avoid
merge conflicts on notebook JSON.

Shared code belongs in `src/`; if you need to change something there, mention it
first so we're not editing it in parallel.

### A caveat worth knowing before you report numbers

SMOTE-ENN used to be applied to the **whole dataset** before any train/test
split. If you split *that* combined data, your test set would contain
synthetic rows and would have had borderline real rows removed by ENN —
metrics on it look good and don't reflect real-world performance.

The pipeline now avoids this itself: `scripts/run_resample.py` splits off a
stratified holdout slice from the **raw** data first (`data.test_size` in the
config, default 20%), and only runs SMOTE-ENN on the remaining training rows.
The holdout never contains synthetic rows and never has real rows dropped.

**Always evaluate headline numbers on the `brfss-holdout-test` artifact**, and
train on `brfss-smoteenn-resampled`. Since it's the same shared script for all
three of us, the evaluation protocol is automatically consistent — no need to
separately agree on a split.

---

## Layout

```text
diabetes-brfss-ml/
├── configs/
│   └── resample_smoteenn.yaml   # sampling params, seed, paths, W&B + artifact names
├── data/
│   ├── raw/                     # Kaggle CSV goes here (gitignored)
│   └── processed/               # resampled + holdout output (gitignored)
├── src/
│   ├── data/
│   │   ├── load.py              # CSV loading + schema/dtype/target validation
│   │   └── resample.py          # config-driven holdout split + SMOTE-ENN, W&B logging + artifacts
│   └── utils/
│       └── seed.py              # set_seed() for reproducibility
├── scripts/
│   └── run_resample.py          # entrypoint, owns wandb.init()/finish()
└── notebooks/
    ├── mahdi/train_model_v1.ipynb
    ├── sadman/train_model_v1.ipynb
    └── abrar/train_model_v1.ipynb
```

All resampling behaviour is config-driven — `configs/resample_smoteenn.yaml` is
the place to change the seed, `sampling_strategy`, SMOTE's `k_neighbors`, ENN's
`n_neighbors`/`kind_sel`, `test_size`, paths, and the W&B project/entity/artifact
names.
