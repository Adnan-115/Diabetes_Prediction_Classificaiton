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

---

## 1. Setup

Requires **Python 3.13**. From the repo root:

```bash
python3.13 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
```

`requirements.txt` holds the direct dependencies at exact pinned versions.
`requirements.lock.txt` is the full `pip freeze` (every transitive dependency) —
use it if you need a bit-for-bit identical environment:

```bash
pip install -r requirements.lock.txt
```

### Weights & Biases

```bash
wandb login
```

Paste the key from https://wandb.ai/authorize. Alternatively copy `.env.example`
to `.env` and set `WANDB_API_KEY` there — `.env` is gitignored, so never commit
a real key.

Set `wandb.entity` in `configs/resample_smoteenn.yaml` to our shared team name
once we've created one; leaving it `null` uses your personal default entity, and
then the artifact won't be visible to the others.

### nbstripout (everyone must do this once)

```bash
nbstripout --install --attributes .gitattributes
```

This installs a git filter that strips notebook **output cells** before they're
committed, so notebook diffs stay readable and the repo stays small with three
people committing notebooks independently.

It lives in `.git/config`, which is **not** part of the repo — cloning does not
set it up for you. Run it once after you create your venv. Check it took with:

```bash
nbstripout --status
```

---

## 2. Get the raw data

Download `diabetes_binary_health_indicators_BRFSS2015.csv` from the Kaggle
dataset and put it in `data/raw/`:

```
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
3. Apply SMOTE-ENN — SMOTE oversamples the minority class, then Edited Nearest
   Neighbours drops samples whose neighbours disagree with them.
4. Log the full config, before/after class distributions (W&B table + bar chart),
   and row counts.
5. Write `data/processed/diabetes_binary_smoteenn.csv`.
6. Upload that CSV as a W&B artifact named `brfss-smoteenn-resampled`.

Useful flags:

```bash
python scripts/run_resample.py --config configs/resample_smoteenn.yaml --wandb-mode disabled
```

`--wandb-mode` overrides the config (`online` / `offline` / `disabled`) — handy
for testing the pipeline without creating runs.

**On runtime:** SMOTE-ENN on the full 253,680 rows is the slow part. SMOTE first
grows the data to ~440k rows, then ENN runs a k-nearest-neighbour search across
all of it. Expect this to take a while (minutes to tens of minutes depending on
your machine) and to hold a few GB of RAM. For a quick end-to-end check first,
set `data.subsample_n` in the config to something like `20000` — it takes a
stratified subsample of the raw rows. Set it back to `null` for the real run.

---

## 4. Pull the resampled dataset (Sadman & Abrar)

**You do not need to run the resampling step, and you don't need the raw CSV.**
Open your own notebook — `notebooks/sadman/train_model_v1.ipynb` or
`notebooks/abrar/train_model_v1.ipynb` — and run the first cell. It does this:

```python
import wandb, pandas as pd
from pathlib import Path

run = wandb.init(project="diabetes-brfss-ml", job_type="train")
artifact = run.use_artifact("brfss-smoteenn-resampled:latest")
artifact_dir = Path(artifact.download())

df = pd.read_csv(next(artifact_dir.glob("*.csv")))
```

`:latest` resolves to the newest version. Once you're producing results you want
to be able to reproduce, pin an explicit version instead —
`brfss-smoteenn-resampled:v0` — so a re-run of the resampling step doesn't
silently change the data under your model.

If W&B can't find the artifact, it's almost certainly the entity: qualify it
fully as `"<entity>/diabetes-brfss-ml/brfss-smoteenn-resampled:latest"`.

The download is cached in `artifacts/` (gitignored), so it only pulls once.

---

## 5. Working agreement

**Each person works only inside their own `notebooks/<name>/` folder** — that
way three people committing notebooks never touch the same file, and we avoid
merge conflicts on notebook JSON.

Shared code belongs in `src/`; if you need to change something there, mention it
first so we're not editing it in parallel.

### A caveat worth knowing before you report numbers

SMOTE-ENN is applied to the **whole dataset** before any train/test split. That
means if you split the resampled data, your test set contains synthetic rows and
has had borderline real rows removed by ENN — metrics on it will look good and
won't reflect real-world performance.

For headline numbers, evaluate on a held-out slice of the **original** raw data.
Worth us agreeing on one shared evaluation protocol early so the three sets of
results are actually comparable.

---

## Layout

```
diabetes-brfss-ml/
├── configs/
│   └── resample_smoteenn.yaml   # sampling params, seed, paths, W&B + artifact names
├── data/
│   ├── raw/                     # Kaggle CSV goes here (gitignored)
│   └── processed/               # resampled output (gitignored)
├── src/
│   ├── data/
│   │   ├── load.py              # CSV loading + schema/dtype/target validation
│   │   └── resample.py          # config-driven SMOTE-ENN, W&B logging + artifact
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
`n_neighbors`/`kind_sel`, paths, and the W&B project/entity/artifact names.
