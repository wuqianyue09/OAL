"""Prepare MMLU inputs, reference alignment and sidecar metadata."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
import time
from .. import data as data_module
from .. import run_artifacts
from .. import runtime_identity
from ..config import PilotConfig, from_json_file
from .evaluation_reference import GROUPED_MMLU_REFERENCE_MAX_SEQUENCE_LENGTH
from ..experiment_contract import BEST_ADAPTER_FILENAME
from ..paths import atomic_create_json, canonical_json, sha256_bytes, sha256_file
from ..runtime_execution import require_finite_tensor, model_logits
from ..workflows.errors import OrchestrationError
from ..workflows.prepare import _load_local_pilot_tokenizer, match_run_effective_config


def _preflight_pilot_mmlu_request(config: PilotConfig, *, run_dir: str | Path) -> None:
    """Bind a MMLU sidecar request to the exact pilot NLL ``best_adapter``."""
    destination = Path(run_dir)
    if not destination.is_dir():
        raise FileNotFoundError(
            f"MMLU evaluation run directory does not exist: {destination}"
        )
    stored_config = from_json_file(
        destination / run_artifacts.EFFECTIVE_CONFIG_FILENAME
    )
    config = match_run_effective_config(config, stored_config)
    nll_path = destination / "pilot_nll_eval.json"
    nll_record = run_artifacts._read_json_mapping(nll_path, "pilot NLL evaluation")
    if (
        nll_record.get("kind") != "qwen_lora_nll_evaluation"
        or nll_record.get("evaluation_scope") != "pilot_single_seed"
        or "formal_evidence" in nll_record
    ):
        raise OrchestrationError(
            "pilot MMLU evaluation requires a pilot-only NLL record"
        )
    best_checkpoint = nll_record.get("best_checkpoint")
    checkpoint_path = destination / BEST_ADAPTER_FILENAME
    if not isinstance(best_checkpoint, Mapping) or not checkpoint_path.is_file():
        raise OrchestrationError(
            "pilot MMLU evaluation requires a pilot NLL best_adapter source"
        )
    if best_checkpoint.get("path") != str(
        checkpoint_path.resolve()
    ) or best_checkpoint.get("sha256") != sha256_file(checkpoint_path):
        raise OrchestrationError(
            "pilot MMLU evaluation requires the same best_adapter as pilot NLL"
        )


def _load_pilot_mmlu_inputs(
    config: PilotConfig,
) -> tuple[
    object,
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, tuple[str, ...]],
    dict[str, object],
    dict[str, object],
]:
    """Validate MMLU's separate immutable bundle and its tokenizer identity."""
    from ..data import tokenizer_identity
    from ..mmlu import (
        MMLU_BUNDLE_FILENAMES,
        _read_jsonl,
        mmlu_bundle_path,
        validate_mmlu_bundle,
        validate_mmlu_bundle_directory,
    )

    tokenizer = _load_local_pilot_tokenizer(Path(config.model_path))
    identity = tokenizer_identity(
        tokenizer,
        vocab_file=data_module._local_tokenizer_file(
            config.model_path, "tokenizer.json"
        ),
        config_file=data_module._local_tokenizer_file(
            config.model_path, "tokenizer_config.json"
        ),
    )
    bundle_path = mmlu_bundle_path(config.data_root)
    try:
        manifest = validate_mmlu_bundle_directory(
            bundle_path, tokenizer_identity=identity
        )
        dev_path, test_path = (
            bundle_path / filename for filename in MMLU_BUNDLE_FILENAMES
        )
        normalized_dev, normalized_test, _ = validate_mmlu_bundle(
            _read_jsonl(dev_path), _read_jsonl(test_path)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OrchestrationError(
            "pilot MMLU input bundle/tokenizer admission failed"
        ) from exc
    token_work, _ = _mmlu_token_work(
        tokenizer,
        dev_rows=normalized_dev,
        test_rows=normalized_test,
        maximum_full_candidate_length=(
            GROUPED_MMLU_REFERENCE_MAX_SEQUENCE_LENGTH
            if config.uses_grouped_quadratic_base
            else None
        ),
    )
    expected_test_ids: dict[str, list[str]] = {}
    for row in normalized_test:
        subject = row["subject"]
        identifier = row["id"]
        assert isinstance(subject, str) and isinstance(identifier, str)
        expected_test_ids.setdefault(subject, []).append(identifier)
    return (
        tokenizer,
        tuple(normalized_dev),
        tuple(normalized_test),
        {
            subject: tuple(identifiers)
            for (subject, identifiers) in expected_test_ids.items()
        },
        {
            "bundle_sha256": manifest["bundle_sha256"],
            "tokenizer": dict(manifest["tokenizer"]),
        },
        token_work,
    )


def _grouped_mmlu_quality_execution(
    execution: Mapping[str, object],
    *,
    token_work: Mapping[str, object],
    smoke: Mapping[str, object],
    alignment: Mapping[str, object],
) -> dict[str, object]:
    """Render immutable MMLU-only Grouped quality disclosure metadata."""
    normalized = dict(execution)
    attention = normalized.get("attention")
    if (
        not isinstance(attention, Mapping)
        or attention.get("method") != "grouped_quadratic"
    ):
        raise OrchestrationError(
            "grouped MMLU reference quality requires grouped OAL training evidence"
        )
    if attention.get("execution") not in {"triton", "hd_block_gemm_causal"}:
        raise OrchestrationError(
            "grouped MMLU reference quality requires supported OAL training attention evidence"
        )
    observed_length = token_work.get("max_full_candidate_length")
    if type(observed_length) is not int or observed_length < 2:
        raise OrchestrationError(
            "grouped MMLU token work lacks observed candidate length"
        )
    normalized["grouped_mmlu_quality_execution"] = {
        "schema": "grouped_mmlu_quality_execution_v1",
        "execution": "reference",
        "public_callable": "oal_attention.oal_attention",
        "scope": "quality_only_no_triton_capability_or_performance_claim",
        "configured_max_sequence_length": GROUPED_MMLU_REFERENCE_MAX_SEQUENCE_LENGTH,
        "observed_max_sequence_length": observed_length,
        "longest_candidate_smoke": dict(smoke),
        "reference_alignment": dict(alignment),
    }
    return normalized


def _grouped_base_mmlu_quality_execution(
    config: PilotConfig,
    execution: Mapping[str, object],
    *,
    token_work: Mapping[str, object],
    smoke: Mapping[str, object],
    alignment: Mapping[str, object],
) -> dict[str, object]:
    return _grouped_mmlu_quality_execution(
        execution, token_work=token_work, smoke=smoke, alignment=alignment
    )


def _run_grouped_mmlu_reference_smoke_and_alignment(
    *,
    model: object,
    kernel_bank: object,
    tokenizer: object,
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    layer_ids: Sequence[int],
    device: object,
    seed: int,
) -> dict[str, object]:
    """Run one longest-prompt finite smoke and N=128 alignment per selected layer."""
    from ..attention.grouped import grouped_quadratic_reference_alignment

    return _run_mmlu_reference_smoke_and_alignment(
        model=model,
        kernel_bank=kernel_bank,
        tokenizer=tokenizer,
        dev_rows=dev_rows,
        test_rows=test_rows,
        layer_ids=layer_ids,
        device=device,
        seed=seed,
        alignment_runner=grouped_quadratic_reference_alignment,
        alignment_schema="grouped_reference_alignment_set_v1",
        smoke_label="grouped MMLU longest candidate logits",
    )


def _run_mmlu_reference_smoke_and_alignment(
    *,
    model: object,
    kernel_bank: object,
    tokenizer: object,
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    layer_ids: Sequence[int],
    device: object,
    seed: int,
    alignment_runner: Callable[..., Mapping[str, object]],
    alignment_schema: str,
    smoke_label: str,
) -> dict[str, object]:
    import torch

    candidate_ids = _mmlu_candidate_token_ids(
        tokenizer, dev_rows=dev_rows, test_rows=test_rows
    )
    longest = max(candidate_ids, key=len)
    if len(longest) > GROUPED_MMLU_REFERENCE_MAX_SEQUENCE_LENGTH:
        raise OrchestrationError(
            f"grouped-base MMLU reference quality evaluation exceeds the maximum supported logical length {GROUPED_MMLU_REFERENCE_MAX_SEQUENCE_LENGTH}"
        )
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(kernel_bank, "eval"):
        kernel_bank.eval()
    input_ids = torch.tensor(longest, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        require_finite_tensor(model_logits(model, input_ids), smoke_label)
    alignments = [
        alignment_runner(
            parameter_bank=kernel_bank, layer_id=layer_id, device=device, seed=seed
        )
        for layer_id in layer_ids
    ]
    return {
        "smoke": {
            "sequence_length": len(longest),
            "longest_candidate_length": len(longest),
            "finite": True,
            "grad_enabled": False,
        },
        "alignment": {
            "schema": alignment_schema,
            "sequence_length": 128,
            "seed": seed,
            "layers": alignments,
        },
    }


def _run_grouped_base_mmlu_reference_smoke_and_alignment(
    config: PilotConfig,
    *,
    model: object,
    kernel_bank: object,
    tokenizer: object,
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    layer_ids: Sequence[int],
    device: object,
    seed: int,
) -> dict[str, object]:
    runner = _run_grouped_mmlu_reference_smoke_and_alignment
    return runner(
        model=model,
        kernel_bank=kernel_bank,
        tokenizer=tokenizer,
        dev_rows=dev_rows,
        test_rows=test_rows,
        layer_ids=layer_ids,
        device=device,
        seed=seed,
    )


def _mmlu_candidate_token_ids(
    tokenizer: object,
    *,
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[int, ...], ...]:
    """Return the full scorer inputs, enforcing Grouped's 4096 safety cap."""
    _, candidates = _mmlu_token_work(
        tokenizer,
        dev_rows=dev_rows,
        test_rows=test_rows,
        maximum_full_candidate_length=GROUPED_MMLU_REFERENCE_MAX_SEQUENCE_LENGTH,
    )
    return candidates


def _mmlu_token_work(
    tokenizer: object,
    *,
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    maximum_full_candidate_length: int | None = None,
) -> tuple[dict[str, object], tuple[tuple[int, ...], ...]]:
    """Derive exact MMLU forward/token work from the frozen scorer inputs."""
    from ..mmlu import format_mmlu_prompt, mmlu_answer_candidates, validate_mmlu_rows

    if not callable(tokenizer):
        raise OrchestrationError(
            "grouped MMLU tokenizer is unavailable for reference admission"
        )
    normalized_dev = validate_mmlu_rows(dev_rows)
    normalized_test = validate_mmlu_rows(test_rows)
    demonstrations: dict[str, list[dict[str, object]]] = {}
    for row in normalized_dev:
        demonstrations.setdefault(str(row["subject"]), []).append(row)
    candidates: list[tuple[int, ...]] = []
    prompt_lengths: list[int] = []
    for row_index, row in enumerate(normalized_test):
        subject = str(row["subject"])
        prompt = format_mmlu_prompt(subject, demonstrations[subject], row)
        prompt_ids = _mmlu_token_ids(
            tokenizer, prompt, row_index=row_index, candidate_index=None
        )
        prompt_lengths.append(len(prompt_ids))
        for candidate_index, answer in enumerate(mmlu_answer_candidates()):
            candidates.append(
                _mmlu_token_ids(
                    tokenizer,
                    prompt + answer,
                    row_index=row_index,
                    candidate_index=candidate_index,
                )
            )
    if not candidates:
        raise OrchestrationError("MMLU bundle has no candidate inputs")
    maximum = max(map(len, candidates))
    if (
        maximum_full_candidate_length is not None
        and maximum > maximum_full_candidate_length
    ):
        raise OrchestrationError(
            f"grouped MMLU reference quality evaluation exceeds the maximum supported logical length {maximum_full_candidate_length}"
        )
    ordered_prompts = sorted(prompt_lengths)
    p95_index = max(0, math.ceil(0.95 * len(ordered_prompts)) - 1)
    return (
        {
            "forward_count": len(prompt_lengths),
            "max_prompt_tokens": max(prompt_lengths),
            "p95_prompt_tokens": ordered_prompts[p95_index],
            "total_prompt_tokens": sum(prompt_lengths),
            "max_full_candidate_length": maximum,
        },
        tuple(candidates),
    )


def _mmlu_token_ids(
    tokenizer: object, text: str, *, row_index: int, candidate_index: int | None
) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if (
        not isinstance(token_ids, Sequence)
        or isinstance(token_ids, (str, bytes))
        or len(token_ids) < 2
        or any((type(token_id) is not int for token_id in token_ids))
    ):
        label = "prompt" if candidate_index is None else f"candidate {candidate_index}"
        raise OrchestrationError(
            f"MMLU row {row_index} {label} has no valid runtime token IDs"
        )
    return tuple(token_ids)


def _mmlu_runtime_observation(
    started_at: float, *, device: object
) -> dict[str, object]:
    """Record available runtime facts as disclosure, never as a performance gate."""
    observation: dict[str, object] = {
        "wall_time_seconds": time.perf_counter() - started_at
    }
    try:
        import torch

        target = torch.device(device)
        if target.type == "cuda" and torch.cuda.is_available():
            observation["cuda_peak_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(target)
            )
            observation["cuda_peak_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(target)
            )
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    return observation


def _reset_mmlu_cuda_peak_memory(device: object) -> None:
    """Start MMLU's optional CUDA telemetry interval immediately before scoring."""
    try:
        import torch

        target = torch.device(device)
        if target.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(target)
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass


def _write_pilot_mmlu_orchestration_metadata(
    result: object,
    metadata: Mapping[str, object],
    *,
    runtime_observation: Mapping[str, object],
) -> None:
    """Persist stable evidence separately from a fresh, non-identity attempt record."""
    final_path = getattr(result, "final_path", None)
    if not isinstance(final_path, Path):
        return
    sidecar = final_path.parent
    if sidecar.name != "mmlu_sidecar":
        raise ValueError("pilot MMLU metadata requires the canonical sidecar directory")
    immutable_path = sidecar / "mmlu_orchestration_metadata.json"
    encoded = (canonical_json(dict(metadata)) + "\n").encode("utf-8")
    try:
        atomic_create_json(immutable_path, dict(metadata))
    except FileExistsError:
        if immutable_path.read_bytes() != encoded:
            raise ValueError(
                "pilot MMLU orchestration metadata already exists with different immutable contents"
            )
    immutable_digest = sha256_bytes(encoded)
    attempt_path = sidecar / "mmlu_orchestration_attempt_metadata.json"
    attempt_record = {
        "schema": "qwen_lora_pilot_mmlu_attempt_metadata_v1",
        "immutable_metadata_sha256": immutable_digest,
        "runtime_observation": dict(runtime_observation),
    }
    try:
        atomic_create_json(attempt_path, attempt_record)
    except FileExistsError:
        _validate_pilot_mmlu_attempt_metadata(
            attempt_path, expected_immutable_metadata_sha256=immutable_digest
        )


def _validate_pilot_mmlu_attempt_metadata(
    path: Path, *, expected_immutable_metadata_sha256: object
) -> None:
    """Accept a create-only attempt collision only for this immutable sidecar."""
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise ValueError("pilot MMLU attempt metadata is unreadable") from exc
    try:
        record = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pilot MMLU attempt metadata is unreadable") from exc
    if not isinstance(record, Mapping):
        raise ValueError(
            "pilot MMLU attempt metadata does not bind the current immutable sidecar"
        )
    if (
        set(record) != {"schema", "immutable_metadata_sha256", "runtime_observation"}
        or record.get("schema") != "qwen_lora_pilot_mmlu_attempt_metadata_v1"
        or record.get("immutable_metadata_sha256") != expected_immutable_metadata_sha256
        or (not isinstance(record.get("runtime_observation"), Mapping))
    ):
        raise ValueError(
            "pilot MMLU attempt metadata does not bind the current immutable sidecar"
        )
