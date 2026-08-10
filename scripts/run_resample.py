#!/usr/bin/env python
"""Entrypoint for the SMOTENC + ENN resampling step.

    python scripts/run_resample.py --config configs/resample_smoteenn.yaml

Owns the W&B run lifecycle (init/finish); the actual work lives in
`src/data/resample.py`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `src` importable when this is run as a script from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

import wandb  # noqa: E402

from src.data.resample import load_config, run_resampling  # noqa: E402

logger = logging.getLogger("run_resample")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply SMOTENC + ENN resampling to the BRFSS 2015 diabetes dataset "
        "and publish the result as a W&B artifact."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML config, e.g. configs/resample_smoteenn.yaml",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="Override wandb.mode from the config (handy: --wandb-mode disabled).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Picks up WANDB_API_KEY / WANDB_ENTITY from a local .env if present.
    load_dotenv()

    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    wandb_cfg = config.get("wandb", {}) or {}
    mode = args.wandb_mode or wandb_cfg.get("mode", "online")

    run = None
    try:
        run = wandb.init(
            project=wandb_cfg.get("project", "diabetes-brfss-ml"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            job_type=wandb_cfg.get("job_type", "preprocess"),
            tags=wandb_cfg.get("tags"),
            mode=mode,
            # The full YAML lands in the run config, so any run is reproducible
            # from what W&B stored.
            config=config,
        )
        logger.info("W&B run: %s (mode=%s)", getattr(run, "url", "n/a"), mode)

        summary = run_resampling(config, run=run)

        logger.info(
            "Done. Train: %d rows -> %d resampled rows in %.1fs. Holdout (raw, "
            "untouched): %d rows. Output: %s | Holdout: %s",
            summary["n_rows_before"],
            summary["n_rows_after"],
            summary["duration_seconds"],
            summary["n_holdout_rows"],
            summary["output_path"],
            summary["holdout_output_path"],
        )
        artifact_name = (config.get("artifact", {}) or {}).get(
            "name", "brfss-smoteenn-resampled"
        )
        holdout_artifact_name = (config.get("holdout_artifact", {}) or {}).get(
            "name", "brfss-holdout-test"
        )
        if mode != "disabled":
            logger.info(
                'Teammates can now pull training data with: '
                'run.use_artifact("%s:latest").download()',
                artifact_name,
            )
            logger.info(
                'For headline metrics, evaluate on the RAW holdout set instead: '
                'run.use_artifact("%s:latest").download()',
                holdout_artifact_name,
            )
        return 0

    except Exception:
        logger.exception("Resampling run failed.")
        if run is not None:
            run.finish(exit_code=1)
            run = None
        return 1

    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
