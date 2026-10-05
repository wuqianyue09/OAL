#!/usr/bin/env python3
"""Run a two-step OAL + LoRA smoke test."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--runs-root")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--hd-manifest")
    parser.add_argument("--hd-probe")
    return parser


def main():
    args = build_parser().parse_args()
    from qwen_lora_experiment.workflows.config import load_pilot_config
    from qwen_lora_experiment.workflows.smoke import run_smoke

    config = load_pilot_config(args.config, runs_root=args.runs_root)
    report = run_smoke(
        config,
        run_dir=args.run_dir,
        preflight_only=args.preflight_only,
        hd_manifest_path=args.hd_manifest,
        hd_probe_path=args.hd_probe,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
