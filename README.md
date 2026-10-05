# OAL: Outlier-Aware Attention Linearization

Official implementation of **OAL**. OAL groups attention dimensions by their
contribution and learns a quadratic kernel with shared coefficients, enabling
linear complexity in sequence length for a fixed head dimension.

This repository includes the OAL operator and the Qwen2.5-0.5B LoRA experiment,
with precomputed dimension groups and initial coefficients.

## Installation

Requires Python 3.10+, PyTorch 2.6+ with CUDA, and an NVIDIA GPU with BF16 support.
Run from the repository root:

```bash
python -m pip install -e '.[cuda,data,experiment]'
```

The experiment uses Transformers **4.51.3**. The OAL kernels are included.

## Qwen experiment

Download [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), including its
tokenizer. Set `model_path` and `data_root` in
[configs/lora_qwen_n4096_template.json](configs/lora_qwen_n4096_template.json)
to your local directories. The released groups and coefficients are in
[configs/oal_qwen_n4096.json](configs/oal_qwen_n4096.json); no calibration is required.

The default configuration uses sequence length 4096, seed 42, 512 optimizer
steps, and LoRA rank 8. OAL replaces attention in zero-based layers 3–20.
Its configuration identifier is `grouped_quadratic`.

Prepare WikiText and PIQA, then run a two-step smoke test and train:

```bash
CONFIG=configs/lora_qwen_n4096_template.json
RUN_DIR=runs/oal_seed42

python scripts/prepare_data.py --config "$CONFIG"
python scripts/smoke_test.py --config "$CONFIG" --run-dir runs/oal_smoke
python scripts/train_pilot.py --config "$CONFIG" --run-dir "$RUN_DIR"
```

Use fresh run directories. To resume training, add `--resume` to the training
command. Evaluate the trained checkpoint using its saved configuration:

```bash
python scripts/evaluate_pilot.py \
  --config "$RUN_DIR/effective_config.json" --run-dir "$RUN_DIR" \
  --stage nll --evaluation-mode pilot
python scripts/evaluate_pilot.py \
  --config "$RUN_DIR/effective_config.json" --run-dir "$RUN_DIR" \
  --stage piqa --evaluation-mode pilot
```
