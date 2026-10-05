#!/usr/bin/env python3
"""Prepare immutable WikiText/PIQA pilot assets without import-time downloads."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build local, immutable WikiText and PIQA assets for the Qwen LoRA pilot."
    )
    parser.add_argument(
        "--config", required=True, help="Fully explicit pilot JSON configuration."
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=(2048, 4096),
        help="N-specific asset length; defaults to the validated config value.",
    )
    parser.add_argument(
        "--wikitext-revision",
        default="main",
        help="Explicit WikiText dataset revision recorded in the manifest.",
    )
    parser.add_argument(
        "--piqa-revision",
        default="main",
        help="Explicit PIQA dataset revision recorded in the manifest.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # argparse handles --help before importing Transformers or Datasets.
    from qwen_lora_experiment.data import (
        download_piqa_validation,
        download_wikitext_splits,
    )
    from qwen_lora_experiment.workflows.config import load_pilot_config
    from qwen_lora_experiment.workflows.prepare import prepare_pilot_data

    def download_pilot_datasets(
        *, wikitext_revision: str, piqa_revision: str
    ) -> tuple[object, object]:
        """The CLI's sole explicit network boundary for immutable source assets."""

        return (
            download_wikitext_splits(revision=wikitext_revision),
            download_piqa_validation(revision=piqa_revision),
        )

    config = load_pilot_config(args.config, sequence_length=args.sequence_length)
    result = prepare_pilot_data(
        config,
        wikitext_revision=args.wikitext_revision,
        piqa_revision=args.piqa_revision,
        dataset_downloader=download_pilot_datasets,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
