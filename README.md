# OAL: Outlier-Aware Attention Linearization

Official implementation of **OAL: Outlier-Aware Attention Linearization**.

OAL uses a Hadamard-product formulation to model dimension-wise contributions
and cross-dimension interactions in attention. Grouping dimensions by their
contributions allows kernel coefficients to be shared, yielding a compact
quadratic parameterization with linear complexity in sequence length for a
fixed head dimension.

This release includes the OAL kernels, precomputed dimension groups and initial
coefficients, and Qwen2.5-0.5B LoRA training and evaluation code.

## Installation

Requires Linux, Python 3.10+, PyTorch 2.6+ with CUDA, and an NVIDIA GPU with BF16
support. The experiments use Transformers **4.51.3**. Activate your Python
environment and run the following commands from the repository root:

```bash
python -m pip install -e '.[cuda,data,experiment]'
```

This installs OAL in editable mode together with Triton, Datasets, and
Transformers. Install a CUDA-enabled version of PyTorch before running it.

## Training

Download [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), including its
tokenizer. Set `model_path` and `data_root` in
[configs/lora_qwen_n4096_template.json](configs/lora_qwen_n4096_template.json)
to your local directories. Precomputed groups and coefficients are provided in
[configs/oal_qwen_n4096.json](configs/oal_qwen_n4096.json).

Prepare WikiText and PIQA, then train OAL with LoRA:

```bash
CONFIG=configs/lora_qwen_n4096_template.json
RUN_DIR=runs/oal_seed42

python scripts/prepare_data.py --config "$CONFIG"
python scripts/train_pilot.py --config "$CONFIG" --run-dir "$RUN_DIR"
```

Use a new run directory, or add `--resume` to continue an existing run.

## Evaluation

Evaluate the saved checkpoint on WikiText first, then PIQA:

```bash
RUN_DIR=runs/oal_seed42

python scripts/evaluate_pilot.py \
  --config "$RUN_DIR/effective_config.json" --run-dir "$RUN_DIR" \
  --stage nll
python scripts/evaluate_pilot.py \
  --config "$RUN_DIR/effective_config.json" --run-dir "$RUN_DIR" \
  --stage piqa
```
