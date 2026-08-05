"""Config-driven SMOTE-ENN resampling, with W&B logging and artifact upload.

The entrypoint is `run_resampling(config, run)`, called by
`scripts/run_resample.py`. That script owns the `wandb.init()` /
`wandb.finish()` lifecycle; this module only *uses* the run it is handed, so it
stays importable from a notebook without side effects.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
import yaml
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import train_test_split

from src.data.load import (
    class_distribution,
    load_raw,
    split_features_target,
)
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Read the YAML config. Raises if it is missing or not a mapping."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return config


def build_smoteenn(config: dict[str, Any], seed: int) -> SMOTEENN:
    """Construct SMOTEENN from config.

    Note: when `smote=` / `enn=` are passed explicitly, imbalanced-learn clones
    them as-is and SMOTEENN's own `sampling_strategy` and `random_state` are
    *not* propagated into them. So we set those on the sub-estimators here --
    otherwise the config's sampling_strategy and seed would silently do nothing.
    """
    cfg = config.get("smoteenn", {}) or {}
    smote_cfg = cfg.get("smote", {}) or {}
    enn_cfg = cfg.get("enn", {}) or {}

    sampling_strategy = cfg.get("sampling_strategy", "auto")
    n_jobs = cfg.get("n_jobs", -1)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=seed,
        k_neighbors=smote_cfg.get("k_neighbors", 5),
    )
    enn = EditedNearestNeighbours(
        # "all" matches what SMOTEENN uses for its internal default ENN.
        sampling_strategy=enn_cfg.get("sampling_strategy", "all"),
        n_neighbors=enn_cfg.get("n_neighbors", 3),
        kind_sel=enn_cfg.get("kind_sel", "all"),
        n_jobs=n_jobs,
    )
    return SMOTEENN(
        sampling_strategy=sampling_strategy,
        random_state=seed,
        smote=smote,
        enn=enn,
    )


def _maybe_subsample(
    df: pd.DataFrame, target_column: str, n: int | None, seed: int
) -> pd.DataFrame:
    """Stratified subsample for smoke tests. `n` of None/0 means 'use all rows'."""
    if not n or n >= len(df):
        return df
    frac = n / len(df)
    out = (
        df.groupby(target_column, group_keys=False)
        .sample(frac=frac, random_state=seed)
        .reset_index(drop=True)
    )
    logger.warning(
        "SUBSAMPLED to %d of %d rows (subsample_n set in config) -- "
        "this is a smoke-test run, not the real dataset.",
        len(out),
        len(df),
    )
    return out


def _log_distributions(
    run: Any,
    before: pd.DataFrame,
    after: pd.DataFrame,
    target_column: str,
) -> None:
    """Log class balance before/after as a W&B table, a bar chart, and scalars."""
    if run is None:
        return

    rows = []
    for stage, dist in (("before", before), ("after", after)):
        for rec in dist.to_dict(orient="records"):
            rows.append(
                {
                    "stage": stage,
                    "class": rec["class"],
                    "count": rec["count"],
                    "percent": rec["percent"],
                    "stage_class": f"{stage}/class_{rec['class']}",
                }
            )

    table = wandb.Table(
        columns=["stage", "class", "count", "percent", "stage_class"],
        data=[
            [r["stage"], r["class"], r["count"], r["percent"], r["stage_class"]]
            for r in rows
        ],
    )

    payload: dict[str, Any] = {
        "class_distribution/table": table,
        "class_distribution/bar": wandb.plot.bar(
            table, "stage_class", "count", title="Class distribution: before vs after"
        ),
    }

    # Flat scalars too, so they show up in the run summary and are easy to
    # compare across runs in the W&B table view.
    for stage, dist in (("before", before), ("after", after)):
        total = int(dist["count"].sum())
        payload[f"{stage}/n_rows"] = total
        counts = dict(zip(dist["class"], dist["count"]))
        for cls, count in counts.items():
            payload[f"{stage}/class_{cls}_count"] = int(count)
            payload[f"{stage}/class_{cls}_percent"] = round(100 * count / total, 4)
        if len(counts) == 2:
            lo, hi = sorted(int(v) for v in counts.values())
            payload[f"{stage}/imbalance_ratio"] = round(hi / lo, 4) if lo else None

    run.log(payload)
    logger.info("Logged class distributions to W&B.")


def _log_artifact(
    run: Any,
    art_cfg: dict[str, Any],
    default_name: str,
    output_path: Path,
    dist: pd.DataFrame,
    n_rows: int,
    n_features: int,
    config: dict[str, Any],
) -> None:
    """Upload a CSV (resampled train split or raw holdout) as a W&B artifact."""
    if run is None:
        logger.info("W&B run is disabled -- skipping artifact upload.")
        return

    name = art_cfg.get("name", default_name)

    artifact = wandb.Artifact(
        name=name,
        type=art_cfg.get("type", "dataset"),
        description=art_cfg.get("description"),
        metadata={
            "rows": n_rows,
            "n_features": n_features,
            "class_counts": {
                int(r["class"]): int(r["count"])
                for r in dist.to_dict(orient="records")
            },
            "source_config": config,
        },
    )
    artifact.add_file(str(output_path), name=output_path.name)
    run.log_artifact(artifact, aliases=["latest"])
    logger.info("Logged artifact %r (file: %s).", name, output_path.name)


def _split_holdout(
    df: pd.DataFrame, target_column: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/holdout split of the RAW data, before any resampling.

    This has to happen before SMOTE-ENN touches anything: SMOTE fabricates
    synthetic minority rows and ENN deletes real borderline rows, so a split
    taken *after* resampling would leak synthetic data into the test set and
    strip out the hard real examples that make it a fair test. The holdout
    here is untouched raw data and is what headline metrics should be
    computed on.
    """
    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_column],
    )
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def run_resampling(config: dict[str, Any], run: Any = None) -> dict[str, Any]:
    """Load -> validate -> hold out a raw test slice -> SMOTE-ENN the rest ->
    save both CSVs -> log to W&B.

    Args:
        config: parsed YAML config.
        run: an active `wandb.Run`, or None to skip all W&B logging.

    Returns a summary dict (paths, row counts, class distributions, duration).
    """
    seed = set_seed(int(config.get("random_seed", 42)))
    logger.info("Seed set to %d.", seed)

    data_cfg = config.get("data", {}) or {}
    target_column = data_cfg.get("target_column", "diabetes_binary")
    input_path = data_cfg.get("input_path")
    output_path = Path(data_cfg.get("output_path", "data/processed/resampled.csv"))
    holdout_output_path = Path(
        data_cfg.get("holdout_output_path", "data/processed/holdout_test.csv")
    )
    test_size = float(data_cfg.get("test_size", 0.2))

    if not input_path:
        raise ValueError("config['data']['input_path'] is required.")

    df = load_raw(
        input_path,
        target_column=target_column,
        normalize_columns=bool(data_cfg.get("normalize_columns", True)),
        strict_schema=bool(data_cfg.get("strict_schema", True)),
    )
    logger.info("Loaded %d rows x %d columns from %s.", *df.shape, input_path)

    df = _maybe_subsample(df, target_column, data_cfg.get("subsample_n"), seed)

    # Split BEFORE any resampling touches the data, so the holdout set stays
    # 100% real, untouched raw rows -- see `_split_holdout`.
    train_df, holdout_df = _split_holdout(df, target_column, test_size, seed)
    logger.info(
        "Split off a raw holdout set: %d train rows / %d holdout rows (test_size=%.2f).",
        len(train_df),
        len(holdout_df),
        test_size,
    )

    holdout_dist = class_distribution(holdout_df[target_column], target_column)
    logger.info(
        "Holdout class distribution (untouched raw data):\n%s",
        holdout_dist.to_string(index=False),
    )
    holdout_output_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(holdout_output_path, index=False)
    logger.info(
        "Wrote %d raw holdout rows to %s.", len(holdout_df), holdout_output_path
    )

    X, y = split_features_target(train_df, target_column)
    before = class_distribution(y, target_column)
    logger.info(
        "Train-split class distribution BEFORE resampling:\n%s",
        before.to_string(index=False),
    )

    sampler = build_smoteenn(config, seed)
    logger.info(
        "Fitting SMOTE-ENN on %d training rows -- this is the slow step, "
        "the ENN pass does a kNN search over the whole oversampled set. "
        "The holdout set above is excluded from this entirely.",
        len(X),
    )
    started = time.perf_counter()
    X_res, y_res = sampler.fit_resample(X, y)
    elapsed = time.perf_counter() - started
    logger.info("SMOTE-ENN finished in %.1fs.", elapsed)

    # fit_resample returns numpy arrays (imbalanced-learn runs check_X_y
    # internally), so rebuild the frame with the original column names.
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=target_column).astype(int)

    after = class_distribution(y_res, target_column)
    logger.info("Train-split class distribution AFTER resampling:\n%s", after.to_string(index=False))

    resampled = pd.concat(
        [y_res.reset_index(drop=True), X_res.reset_index(drop=True)], axis=1
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_csv(output_path, index=False)
    logger.info("Wrote %d resampled training rows to %s.", len(resampled), output_path)

    _log_distributions(run, before, after, target_column)
    if run is not None:
        run.log(
            {
                "holdout/n_rows": int(holdout_dist["count"].sum()),
                **{
                    f"holdout/class_{int(r['class'])}_count": int(r["count"])
                    for r in holdout_dist.to_dict(orient="records")
                },
            }
        )

    art_cfg = config.get("artifact", {}) or {}
    holdout_art_cfg = config.get("holdout_artifact", {}) or {}

    _log_artifact(
        run,
        art_cfg,
        "brfss-smoteenn-resampled",
        output_path,
        after,
        n_rows=len(resampled),
        n_features=X_res.shape[1],
        config=config,
    )
    _log_artifact(
        run,
        holdout_art_cfg,
        "brfss-holdout-test",
        holdout_output_path,
        holdout_dist,
        n_rows=len(holdout_df),
        n_features=X_res.shape[1],
        config=config,
    )

    if run is not None:
        run.summary["resample/duration_seconds"] = round(elapsed, 2)
        run.summary["resample/output_path"] = str(output_path)
        run.summary["resample/holdout_output_path"] = str(holdout_output_path)

    return {
        "seed": seed,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "holdout_output_path": str(holdout_output_path),
        "target_column": target_column,
        "test_size": test_size,
        "n_rows_before": int(before["count"].sum()),
        "n_rows_after": int(after["count"].sum()),
        "n_holdout_rows": int(holdout_dist["count"].sum()),
        "n_features": int(X_res.shape[1]),
        "class_distribution_before": before.to_dict(orient="records"),
        "class_distribution_after": after.to_dict(orient="records"),
        "holdout_class_distribution": holdout_dist.to_dict(orient="records"),
        "duration_seconds": round(elapsed, 2),
    }
