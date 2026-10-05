#!/usr/bin/env python3
"""Strict one-method Qwen pilot CLI with no import-time model loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from qwen_lora_experiment.config import METHOD_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or resume one Qwen pilot method."
    )
    parser.add_argument(
        "--config", required=True, help="Fully explicit pilot JSON configuration."
    )
    parser.add_argument(
        "--run-dir", required=True, help="New (or --resume existing) run directory."
    )
    parser.add_argument(
        "--runs-root", help="Optional replacement for config.runs_root."
    )
    parser.add_argument(
        "--tuning-mode",
        choices=("lora",),
        help="Exact tuning-policy override; omitted keeps the config value (LoRA by default).",
    )
    parser.add_argument(
        "--method",
        choices=METHOD_NAMES,
        help="Explicit single-method override; all other profile fields remain fixed.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=(2048, 4096),
        help="Shared pilot sequence length override.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Require and strictly restore latest_resume.pt from --run-dir.",
    )
    parser.add_argument("--hd-manifest", help="HD cuBLAS compatibility manifest path.")
    parser.add_argument("--hd-probe", help="HD runtime probe path.")
    parser.add_argument(
        "--operator-root",
        default=Path(__file__).resolve().parents[1],
        help="OAL source root (defaults to this checkout).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from qwen_lora_experiment.workflows.config import load_pilot_config

    overrides = {
        "method": args.method,
        "sequence_length": args.sequence_length,
        "runs_root": args.runs_root,
    }
    if args.tuning_mode is not None:
        overrides["tuning_mode"] = args.tuning_mode
    config = load_pilot_config(args.config, **overrides)
    if getattr(config, "attention_backend", "legacy") == "hd_block_gemm":
        if args.operator_root is None:
            raise ValueError("HD training requires --operator-root")
        from qwen_lora_experiment.operator_imports import activate_operator_import_root

        activate_operator_import_root(args.operator_root)
    from qwen_lora_experiment.workflows.train import execute_pilot_training

    training_kwargs: dict[str, object] = {
        "run_dir": args.run_dir,
        "resume": args.resume,
    }
    if args.hd_manifest is not None:
        training_kwargs["hd_manifest_path"] = args.hd_manifest
    if args.hd_probe is not None:
        training_kwargs["hd_probe_path"] = args.hd_probe
    result = execute_pilot_training(config, **training_kwargs)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
