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
    config: dict[str, Any],
    output_path: Path,
    after: pd.DataFrame,
    n_rows: int,
    n_features: int,
) -> None:
    """Upload the resampled CSV as a versioned W&B dataset artifact."""
    if run is None:
        logger.info("W&B run is disabled -- skipping artifact upload.")
        return

    art_cfg = config.get("artifact", {}) or {}
    name = art_cfg.get("name", "brfss-smoteenn-resampled")

    artifact = wandb.Artifact(
        name=name,
        type=art_cfg.get("type", "dataset"),
        description=art_cfg.get("description"),
        metadata={
            "rows": n_rows,
            "n_features": n_features,
            "class_counts": {
                int(r["class"]): int(r["count"])
                for r in after.to_dict(orient="records")
            },
            "source_config": config,
        },
    )
    artifact.add_file(str(output_path), name=output_path.name)
    run.log_artifact(artifact, aliases=["latest"])
    logger.info("Logged artifact %r (file: %s).", name, output_path.name)


def run_resampling(config: dict[str, Any], run: Any = None) -> dict[str, Any]:
    """Load -> validate -> SMOTE-ENN -> save CSV -> log to W&B.

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

    X, y = split_features_target(df, target_column)
    before = class_distribution(y, target_column)
    logger.info("Class distribution BEFORE:\n%s", before.to_string(index=False))

    sampler = build_smoteenn(config, seed)
    logger.info(
        "Fitting SMOTE-ENN on %d rows -- this is the slow step, "
        "the ENN pass does a kNN search over the whole oversampled set.",
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
    logger.info("Class distribution AFTER:\n%s", after.to_string(index=False))

    resampled = pd.concat(
        [y_res.reset_index(drop=True), X_res.reset_index(drop=True)], axis=1
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s.", len(resampled), output_path)

    _log_distributions(run, before, after, target_column)
    _log_artifact(
        run,
        config,
        output_path,
        after,
        n_rows=len(resampled),
        n_features=X_res.shape[1],
    )

    if run is not None:
        run.summary["resample/duration_seconds"] = round(elapsed, 2)
        run.summary["resample/output_path"] = str(output_path)

    return {
        "seed": seed,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "target_column": target_column,
        "n_rows_before": int(before["count"].sum()),
        "n_rows_after": int(after["count"].sum()),
        "n_features": int(X_res.shape[1]),
        "class_distribution_before": before.to_dict(orient="records"),
        "class_distribution_after": after.to_dict(orient="records"),
        "duration_seconds": round(elapsed, 2),
    }
