#!/usr/bin/env python3
"""Create or verify the separate immutable HellaSwag evaluation bundle."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify the local Rowan/hellaswag validation bundle."
    )
    parser.add_argument(
        "--config", required=True, help="Fully explicit pilot JSON configuration."
    )
    parser.add_argument(
        "--hellaswag-revision",
        dest="requested_source_revision",
        help="Optional datasets revision selector recorded with the local bundle.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate the completed local bundle without importing or downloading datasets.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from qwen_lora_experiment.assets import data_manifest_path, validate_data_manifest
    from qwen_lora_experiment.hellaswag import (
        load_hellaswag_source,
        prepare_hellaswag_bundle,
    )
    from qwen_lora_experiment.workflows.config import load_pilot_config

    config = load_pilot_config(args.config)
    manifest = validate_data_manifest(
        data_manifest_path(config.data_root, config.sequence_length), include_piqa=False
    )
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise RuntimeError(
            "validated data manifest unexpectedly lacks tokenizer identity"
        )
    result = prepare_hellaswag_bundle(
        config.data_root,
        source_loader=None if args.verify_only else load_hellaswag_source,
        tokenizer_identity=tokenizer,
        requested_source_revision=args.requested_source_revision,
        verify_only=args.verify_only,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
