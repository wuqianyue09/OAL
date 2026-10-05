"""Deterministic, injectable training primitives for the LoRA pilot.

The production entry point is deliberately independent of Transformers and
DataLoader.  Model setup supplies a forward function and a random-access block
source, allowing this module to enforce the run contract with small CPU fakes
as well as the real Qwen model.
"""

from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import random
import time
import traceback
from typing import Protocol, cast
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from . import training_diagnostics
from .training_diagnostics import EARLY_UPDATE_DIAGNOSTIC_STEPS, TrainingFailure
from .attention.common import (
    attention_validation_session,
    run_attention_validation_forward,
)
from .checkpointing import CheckpointContext, save_best_adapter, save_latest_resume
from .kernel_parameters import KernelParameterBank
from .lora import ADAPTER_MODULE_NAME, LoRAAdapter
from .protocol import (
    SeedDerivations,
    preserve_global_rng_state,
    seed_global_rng,
    validate_persisted_seed_derivations,
)
from .telemetry import JsonlWriter, RunStateSink, require_run_state_sink
from .training_measurement.runtime import (
    NOOP_TRAINING_MEASUREMENT_RUNTIME,
    TrainingMeasurementRuntime,
)

_PILOT_VALIDATION_BLOCKS = 32
_INTEGER_DTYPES = frozenset(
    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
)


class _TrainingConfig(Protocol):
    """The narrow configuration surface consumed by the training loop."""

    train_blocks: int
    validation_blocks: int
    validation_interval_steps: int
    diagnostic_interval_steps: int
    gradient_clip: float
    seed_derivations: SeedDerivations


ForwardFunction = Callable[[nn.Module, Tensor, KernelParameterBank], object]
DiagnosticCallback = Callable[[int, KernelParameterBank], None]
Clock = Callable[[], float]
TrainBlockSource = Sequence[object] | Callable[[int], object]


class ReiterableValidationBlocks(Protocol):
    """A sized container that returns a fresh iterator for each validation pass."""

    def __iter__(self) -> Iterator[object]: ...

    def __len__(self) -> int: ...


ValidationBlockSource = ReiterableValidationBlocks | Callable[[], Iterable[object]]


@dataclass(frozen=True)
class PreparedBatch:
    """A copied, device-resident input/label pair for causal language modelling."""

    input_ids: Tensor
    labels: Tensor


@dataclass(frozen=True)
class ValidationBlockNll:
    """JSON-safe next-token NLL evidence for one ordered validation block."""

    block_id: int
    total_nll: float
    token_count: int

    def __post_init__(self) -> None:
        if type(self.block_id) is not int or self.block_id < 0:
            raise ValueError("block_id must be a non-negative integer")
        if (
            isinstance(self.total_nll, bool)
            or not isinstance(self.total_nll, (int, float))
            or (not math.isfinite(float(self.total_nll)))
            or (float(self.total_nll) < 0.0)
        ):
            raise ValueError("total_nll must be finite and non-negative")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("token_count must be a positive integer")

    def as_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "total_nll": self.total_nll,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Exact aggregate NLL over the configured canonical validation blocks."""

    token_count: int
    total_nll: float
    mean_nll: float
    block_records: tuple[ValidationBlockNll, ...] = ()


@dataclass(frozen=True)
class TrainingResult:
    """Evidence from a completed run, including its explicit next block cursor."""

    steps_completed: int
    next_train_block: int
    initial_validation: ValidationResult | None
    final_validation: ValidationResult


@dataclass(frozen=True)
class CheckpointRuntime:
    """Injectable checkpoint side effects for CPU tests and orchestration."""

    save_best: Callable[..., object] = save_best_adapter
    save_latest: Callable[..., object] = save_latest_resume


def causal_next_token_nll(logits: Tensor, labels: Tensor) -> Tensor:
    """Return FP32 mean causal next-token NLL using the conventional shift."""
    shifted_logits, shifted_labels, _ = _validated_shifted_tokens(logits, labels)
    return _causal_nll_from_shifted(shifted_logits, shifted_labels, reduction="mean")


def prepare_training_batch(batch: object, device: torch.device | str) -> PreparedBatch:
    """Move input IDs before forward and independently copy causal labels.

    The loop never accepts model-provided labels or loss.  This makes the
    shifted NLL contract explicit and prevents a caller's CPU batch tensor
    from being aliased as a mutable label tensor on the target device.
    """
    target_device = torch.device(device)
    input_ids: object
    if isinstance(batch, Tensor):
        input_ids = batch
    elif isinstance(batch, Mapping):
        if "input_ids" not in batch:
            raise ValueError("training batch requires an input_ids field")
        input_ids = batch["input_ids"]
    else:
        raise TypeError("training batch must be an input-id Tensor or a mapping")
    if not isinstance(input_ids, Tensor):
        raise TypeError("training batch input_ids must be a torch.Tensor")
    if input_ids.ndim != 2 or input_ids.shape[0] <= 0 or input_ids.shape[1] < 2:
        raise ValueError(
            "training batch input_ids must be non-empty [batch, sequence>=2]"
        )
    if input_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("training batch input_ids must use an integer dtype")
    moved_input_ids = input_ids.to(device=target_device, dtype=torch.long).contiguous()
    return PreparedBatch(input_ids=moved_input_ids, labels=moved_input_ids.clone())


def validate(
    *,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    validation_blocks: Iterable[object],
    config: _TrainingConfig,
    device: torch.device | str,
    forward_fn: ForwardFunction | None = None,
) -> ValidationResult:
    """Evaluate exactly the pilot's 32 ordered validation blocks.

    This function is side-effect bounded: it restores module modes and the
    process RNG streams even if validation fails.  Optimizers are deliberately
    absent from the API, and autograd is disabled around every model call.
    """
    _require_validation_count(config)
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(kernel_bank, KernelParameterBank):
        raise TypeError("kernel_bank must be a KernelParameterBank")
    model_modes = _module_modes(model)
    bank_modes = _module_modes(kernel_bank)
    rng_snapshot = _capture_rng_state()
    token_count = 0
    total_nll = 0.0
    block_records: list[ValidationBlockNll] = []
    try:
        iterator = iter(validation_blocks)
        blocks: list[object] = []
        try:
            for _ in range(config.validation_blocks):
                blocks.append(next(iterator))
        except StopIteration as exc:
            raise ValueError(
                "validation must contain exactly 32 blocks; received too few"
            ) from exc
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError(
                "validation must contain exactly 32 blocks; received too many"
            )
        model.eval()
        kernel_bank.eval()
        with torch.no_grad():
            for block_id, block in enumerate(blocks):
                prepared = prepare_training_batch(block, device)
                logits = _forward_logits(
                    model, prepared.input_ids, kernel_bank, forward_fn
                )
                shifted_logits, shifted_labels, valid_tokens = (
                    _validated_shifted_tokens(logits, prepared.labels)
                )
                block_nll = _causal_nll_from_shifted(
                    shifted_logits, shifted_labels, reduction="sum"
                )
                block_total_nll = float(block_nll.item())
                if not math.isfinite(block_total_nll):
                    raise ValueError(
                        f"validation block {block_id} produced non-finite next-token NLL"
                    )
                block_records.append(
                    ValidationBlockNll(
                        block_id=block_id,
                        total_nll=block_total_nll,
                        token_count=valid_tokens,
                    )
                )
                total_nll += block_total_nll
                token_count += valid_tokens
    finally:
        _restore_module_modes(model_modes)
        _restore_module_modes(bank_modes)
        _restore_rng_state(rng_snapshot)
    if token_count <= 0 or not math.isfinite(total_nll):
        raise ValueError("validation produced no finite next-token NLL")
    return ValidationResult(
        token_count=token_count,
        total_nll=total_nll,
        mean_nll=total_nll / token_count,
        block_records=tuple(block_records),
    )


def run_training(
    *,
    config: _TrainingConfig,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    optimizer: Optimizer,
    train_blocks: TrainBlockSource,
    validation_blocks: ValidationBlockSource,
    run_dir: str | Path,
    checkpoint_context: CheckpointContext,
    checkpoint_runtime: CheckpointRuntime | None = None,
    metrics_writer: JsonlWriter,
    status_store: RunStateSink,
    device: torch.device | str,
    expected_lora_b_names: Sequence[str] | None = None,
    forward_fn: ForwardFunction | None = None,
    scheduler: object | None = None,
    diagnostic_callback: DiagnosticCallback | None = None,
    clock: Clock = time.perf_counter,
    resume_cursor: Mapping[str, object] | None = None,
    measurement_runtime: TrainingMeasurementRuntime = NOOP_TRAINING_MEASUREMENT_RUNTIME,
) -> TrainingResult:
    """Run training without leaking fresh or resumed global RNG state."""
    resolved_measurement_runtime = _require_measurement_runtime(measurement_runtime)
    try:
        with preserve_global_rng_state():
            if resume_cursor is None:
                global_seed = _fresh_training_global_seed(config, checkpoint_context)
                if global_seed is not None:
                    seed_global_rng(global_seed)
            return _run_training_impl(
                config=config,
                model=model,
                kernel_bank=kernel_bank,
                optimizer=optimizer,
                train_blocks=train_blocks,
                validation_blocks=validation_blocks,
                run_dir=run_dir,
                checkpoint_context=checkpoint_context,
                checkpoint_runtime=checkpoint_runtime,
                metrics_writer=metrics_writer,
                status_store=status_store,
                device=device,
                expected_lora_b_names=expected_lora_b_names,
                forward_fn=forward_fn,
                scheduler=scheduler,
                diagnostic_callback=diagnostic_callback,
                clock=clock,
                resume_cursor=resume_cursor,
                measurement_runtime=resolved_measurement_runtime,
            )
    finally:
        resolved_measurement_runtime.finish()


def _fresh_training_global_seed(
    config: _TrainingConfig, checkpoint_context: CheckpointContext
) -> int | None:
    """Return the only allowed fresh-run global seed before loop side effects.

    Formal checkpoint evidence makes the derivation a required part of the
    runtime contract.  The sole compatibility seam is an explicitly marked
    ``nonformal_test`` checkpoint context, which may omit derivations for
    narrow CPU fixtures without becoming reportable evidence.
    """
    context_kind = checkpoint_context.config_identity.get("checkpoint_context_kind")
    if context_kind not in {"formal", "nonformal_test"}:
        raise ValueError(
            "checkpoint context must declare a formal or nonformal_test kind"
        )
    derivations = getattr(config, "seed_derivations", None)
    if not isinstance(derivations, SeedDerivations):
        if context_kind == "formal":
            raise ValueError(
                "fresh formal training requires config.seed_derivations as SeedDerivations"
            )
        return None
    try:
        canonical = validate_persisted_seed_derivations(
            derivations.to_dict(), master_seed=derivations.master_seed
        )
    except ValueError as exc:
        raise ValueError(
            "config.seed_derivations do not match the formal seed protocol"
        ) from exc
    if context_kind == "formal":
        context_derivations = checkpoint_context.config_identity.get("seed_derivations")
        if context_derivations != canonical.to_dict():
            raise ValueError(
                "fresh formal training seed derivations do not match checkpoint evidence"
            )
    return canonical.training_global_rng_seed


def _require_measurement_runtime(value: object) -> TrainingMeasurementRuntime:
    """Validate the tensor-blind scope protocol once before the training hot path."""
    required_methods = ("prepare_step", "step", "scope", "external_scope", "finish")
    if any((not callable(getattr(value, name, None)) for name in required_methods)):
        raise TypeError(
            "measurement_runtime must provide prepare_step, step, scope, external_scope, and finish"
        )
    return value


def _run_training_impl(
    *,
    config: _TrainingConfig,
    model: nn.Module,
    kernel_bank: KernelParameterBank,
    optimizer: Optimizer,
    train_blocks: TrainBlockSource,
    validation_blocks: ValidationBlockSource,
    run_dir: str | Path,
    checkpoint_context: CheckpointContext,
    checkpoint_runtime: CheckpointRuntime | None = None,
    metrics_writer: JsonlWriter,
    status_store: RunStateSink,
    device: torch.device | str,
    expected_lora_b_names: Sequence[str] | None = None,
    forward_fn: ForwardFunction | None = None,
    scheduler: object | None = None,
    diagnostic_callback: DiagnosticCallback | None = None,
    clock: Clock = time.perf_counter,
    resume_cursor: Mapping[str, object] | None = None,
    measurement_runtime: TrainingMeasurementRuntime = NOOP_TRAINING_MEASUREMENT_RUNTIME,
) -> TrainingResult:
    """Run the bounded, cursor-addressed optimization mechanics.

    ``train_blocks`` must be indexed by the explicit block cursor (or be a
    function of it); this intentionally rules out DataLoader ordering and its
    hidden RNG state.  The sole maximum is ``config.train_blocks``.
    """
    current_phase = "preflight"
    current_step = 0
    try:
        if not isinstance(metrics_writer, JsonlWriter):
            raise TypeError("metrics_writer must be a JsonlWriter")
        status_store = require_run_state_sink(status_store)
        resolved_run_dir = Path(run_dir)
        resolved_run_dir.mkdir(parents=True, exist_ok=True)
        if not resolved_run_dir.is_dir():
            raise NotADirectoryError(f"run_dir is not a directory: {resolved_run_dir}")
        _start_run(status_store)
        total_steps = _require_train_blocks(config)
        _require_validation_count(config)
        _require_reiterable_validation_source(validation_blocks)
        _require_positive_finite(config.gradient_clip, "config.gradient_clip")
        _require_positive_interval(
            config.validation_interval_steps, "validation_interval_steps"
        )
        _require_positive_interval(
            config.diagnostic_interval_steps, "diagnostic_interval_steps"
        )
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(kernel_bank, KernelParameterBank):
            raise TypeError("kernel_bank must be a KernelParameterBank")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        if not isinstance(checkpoint_context, CheckpointContext):
            raise TypeError("checkpoint_context must be a CheckpointContext")
        resolved_checkpoint_runtime = (
            CheckpointRuntime() if checkpoint_runtime is None else checkpoint_runtime
        )
        if (
            not isinstance(resolved_checkpoint_runtime, CheckpointRuntime)
            or not callable(resolved_checkpoint_runtime.save_best)
            or (not callable(resolved_checkpoint_runtime.save_latest))
        ):
            raise TypeError(
                "checkpoint_runtime must provide callable save_best and save_latest"
            )
        if kernel_bank.method != checkpoint_context.method:
            raise ValueError("kernel_bank method must match checkpoint_context.method")
        if not callable(clock):
            raise TypeError("clock must be callable")
        start_cursor = _parse_resume_cursor(resume_cursor, total_steps)
        named_trainables = _collect_trainables(model, kernel_bank)
        _validate_optimizer_parameters(optimizer, named_trainables)
        early_gate_state = _prepare_lora_early_gate(
            model, named_trainables, expected_lora_b_names
        )
        initial_validation: ValidationResult | None = None
        last_interval_validation: tuple[int, ValidationResult] | None = None
        if start_cursor == 0:
            current_phase = "initial_validation"
            with measurement_runtime.external_scope("plan_b.external.validation"):
                initial_validation = validate(
                    model=model,
                    kernel_bank=kernel_bank,
                    validation_blocks=_validation_pass(validation_blocks),
                    config=config,
                    device=device,
                    forward_fn=forward_fn,
                )
            with measurement_runtime.external_scope("plan_b.external.save"):
                resolved_checkpoint_runtime.save_best(
                    run_dir,
                    model=model,
                    kernel_bank=kernel_bank,
                    context=checkpoint_context,
                    step=0,
                    validation_nll=initial_validation.mean_nll,
                )
            metrics_writer.write(
                {
                    "event": "validation",
                    "phase": "initial",
                    "step": 0,
                    **_validation_fields(initial_validation),
                }
            )
        else:
            metrics_writer.write(
                {
                    "event": "resume",
                    "step": start_cursor,
                    "next_train_block": start_cursor,
                }
            )
        if start_cursor < total_steps:
            _set_training_mode(model, kernel_bank)
        for block_index in range(start_cursor, total_steps):
            _set_training_mode(model, kernel_bank)
            current_step = block_index + 1
            current_phase = "train_batch"
            raw_block = _train_block_at(train_blocks, block_index)
            prepared = prepare_training_batch(raw_block, device)
            diagnostics: list[dict[str, object]] | None = None
            pre_clip_gradient_norms: dict[str, float] | None = None
            measurement_runtime.prepare_step(current_step)
            with measurement_runtime.step(current_step):
                with measurement_runtime.scope("plan_b.optimizer.zero_grad"):
                    optimizer.zero_grad(set_to_none=True)
                with measurement_runtime.scope("plan_b.diagnostics.before_snapshot"):
                    before = {
                        name: parameter.detach().clone(
                            memory_format=torch.preserve_format
                        )
                        for (name, parameter) in named_trainables.items()
                    }
                step_started_at = _clock_value(clock)
                current_phase = "forward"
                with measurement_runtime.scope("plan_b.model_forward"):
                    logits = _forward_logits(
                        model, prepared.input_ids, kernel_bank, forward_fn
                    )
                with measurement_runtime.scope("plan_b.diagnostics.forward_finite"):
                    if not bool(torch.isfinite(logits).all()):
                        raise TrainingFailure(
                            "model forward produced non-finite logits",
                            phase=current_phase,
                            step=current_step,
                        )
                current_phase = "loss"
                with measurement_runtime.scope("plan_b.loss"):
                    _validate_causal_nll_structure(logits, prepared.labels)
                    shifted_logits, shifted_labels, _ = _shifted_tokens_and_count(
                        logits, prepared.labels
                    )
                    loss = _causal_nll_from_shifted(
                        shifted_logits, shifted_labels, reduction="mean"
                    )
                with measurement_runtime.scope("plan_b.diagnostics.loss_finite"):
                    if not bool(torch.isfinite(loss)):
                        raise TrainingFailure(
                            "training loss must be finite",
                            phase=current_phase,
                            step=current_step,
                        )
                current_phase = "backward"
                with measurement_runtime.scope("plan_b.backward"):
                    _backward_with_attention_validation(loss)
                with measurement_runtime.scope("plan_b.diagnostics.gradient_finite"):
                    training_diagnostics.require_finite_gradients(
                        named_trainables, step=current_step
                    )
                    if current_step <= EARLY_UPDATE_DIAGNOSTIC_STEPS:
                        pre_clip_gradient_norms = (
                            training_diagnostics.pre_clip_gradient_norms(
                                named_trainables
                            )
                        )
                current_phase = "gradient_clip"
                with measurement_runtime.scope("plan_b.optimizer.gradient_clip"):
                    torch.nn.utils.clip_grad_norm_(
                        tuple(named_trainables.values()),
                        max_norm=float(config.gradient_clip),
                        error_if_nonfinite=True,
                    )
                current_phase = "optimizer_step"
                with measurement_runtime.scope("plan_b.optimizer.step"):
                    optimizer.step()
                with measurement_runtime.scope("plan_b.diagnostics.parameter_finite"):
                    training_diagnostics.require_finite_parameter_state_after_step(
                        named_trainables, before, step=current_step
                    )
                with measurement_runtime.scope("plan_b.optimizer.scheduler"):
                    if scheduler is not None:
                        scheduler.step()
                if current_step <= EARLY_UPDATE_DIAGNOSTIC_STEPS:
                    with measurement_runtime.scope(
                        "plan_b.diagnostics.parameter_delta_and_early_update"
                    ):
                        assert pre_clip_gradient_norms is not None
                        diagnostics = training_diagnostics.parameter_diagnostics(
                            named_trainables,
                            before,
                            pre_clip_gradient_norms=pre_clip_gradient_norms,
                        )
                        _enforce_lora_early_gate(
                            diagnostics, current_step, early_gate_state
                        )
            elapsed_seconds = _clock_value(clock) - step_started_at
            metrics_record: dict[str, object] = {
                "event": "train_step",
                "step": current_step,
                "loss": float(loss.item()),
                "lr": _learning_rate(optimizer),
                "tokens": int(prepared.input_ids.numel()),
                "elapsed_seconds": elapsed_seconds,
                "max_memory_bytes": _max_memory_bytes(torch.device(device)),
            }
            if diagnostics is not None:
                metrics_record["parameter_diagnostics"] = diagnostics
            metrics_writer.write(metrics_record)
            optimizer.zero_grad(set_to_none=True)
            if current_step % int(config.validation_interval_steps) == 0:
                current_phase = "interval_validation"
                with measurement_runtime.external_scope("plan_b.external.validation"):
                    interval_validation = validate(
                        model=model,
                        kernel_bank=kernel_bank,
                        validation_blocks=_validation_pass(validation_blocks),
                        config=config,
                        device=device,
                        forward_fn=forward_fn,
                    )
                last_interval_validation = (current_step, interval_validation)
                with measurement_runtime.external_scope("plan_b.external.save"):
                    resolved_checkpoint_runtime.save_best(
                        run_dir,
                        model=model,
                        kernel_bank=kernel_bank,
                        context=checkpoint_context,
                        step=current_step,
                        validation_nll=interval_validation.mean_nll,
                    )
                    resolved_checkpoint_runtime.save_latest(
                        run_dir,
                        model=model,
                        kernel_bank=kernel_bank,
                        context=checkpoint_context,
                        step=current_step,
                        validation_nll=interval_validation.mean_nll,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cursor={"next_train_block": current_step},
                    )
                metrics_writer.write(
                    {
                        "event": "validation",
                        "phase": "interval",
                        "step": current_step,
                        **_validation_fields(interval_validation),
                    }
                )
            if (
                diagnostic_callback is not None
                and current_step % int(config.diagnostic_interval_steps) == 0
                and any((True for _ in kernel_bank.parameters()))
            ):
                current_phase = "kernel_diagnostic"
                with measurement_runtime.external_scope(
                    "plan_b.external.kernel_diagnostic"
                ):
                    diagnostic_callback(current_step, kernel_bank)
        current_phase = "final_validation"
        if (
            diagnostic_callback is None
            and last_interval_validation is not None
            and (last_interval_validation[0] == total_steps)
        ):
            final_validation = last_interval_validation[1]
        else:
            with measurement_runtime.external_scope("plan_b.external.validation"):
                final_validation = validate(
                    model=model,
                    kernel_bank=kernel_bank,
                    validation_blocks=_validation_pass(validation_blocks),
                    config=config,
                    device=device,
                    forward_fn=forward_fn,
                )
        with measurement_runtime.external_scope("plan_b.external.save"):
            if start_cursor < total_steps:
                resolved_checkpoint_runtime.save_best(
                    run_dir,
                    model=model,
                    kernel_bank=kernel_bank,
                    context=checkpoint_context,
                    step=total_steps,
                    validation_nll=final_validation.mean_nll,
                )
            resolved_checkpoint_runtime.save_latest(
                run_dir,
                model=model,
                kernel_bank=kernel_bank,
                context=checkpoint_context,
                step=total_steps,
                validation_nll=final_validation.mean_nll,
                optimizer=optimizer,
                scheduler=scheduler,
                cursor={"next_train_block": total_steps},
            )
        metrics_writer.write(
            {
                "event": "validation",
                "phase": "final",
                "step": total_steps,
                **_validation_fields(final_validation),
            }
        )
        current_phase = "transition_trained"
        status_store.transition("trained", steps_completed=total_steps)
        return TrainingResult(
            steps_completed=total_steps,
            next_train_block=total_steps,
            initial_validation=initial_validation,
            final_validation=final_validation,
        )
    except KeyboardInterrupt:
        _persist_interruption(
            status_store=status_store,
            metrics_writer=metrics_writer,
            phase=current_phase,
            step=current_step,
        )
        raise
    except BaseException as exc:
        failure = (
            exc
            if isinstance(exc, TrainingFailure)
            else TrainingFailure(str(exc), phase=current_phase, step=current_step)
        )
        _persist_failure(
            status_store=status_store,
            metrics_writer=metrics_writer,
            failure=failure,
            original_exception=exc,
        )
        raise failure from exc


def _validated_shifted_tokens(
    logits: Tensor, labels: Tensor
) -> tuple[Tensor, Tensor, int]:
    _validate_causal_nll_structure(logits, labels)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    return _shifted_tokens_and_count(logits, labels)


def _validate_causal_nll_structure(logits: Tensor, labels: Tensor) -> None:
    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("logits and labels must be torch.Tensors")
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "logits must be [batch, sequence, vocabulary] and labels [batch, sequence]"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits batch/sequence dimensions must match labels")
    if logits.shape[-1] <= 0 or logits.shape[1] < 2:
        raise ValueError(
            "causal NLL requires a non-empty vocabulary and at least two tokens"
        )
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating dtype")
    if labels.dtype is not torch.long:
        raise TypeError("labels must use torch.long")
    if logits.device != labels.device:
        raise ValueError("logits and labels must be on the same device")


def _shifted_tokens_and_count(
    logits: Tensor, labels: Tensor
) -> tuple[Tensor, Tensor, int]:
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    if not bool(valid.any()):
        raise ValueError("causal NLL has no valid next-token labels")
    if bool(((shifted_labels < -100) | (shifted_labels >= logits.shape[-1])).any()):
        raise ValueError("labels must be vocabulary indices or -100")
    return (logits[:, :-1, :], shifted_labels, int(valid.sum().item()))


def _causal_nll_from_shifted(
    shifted_logits: Tensor, shifted_labels: Tensor, *, reduction: str
) -> Tensor:
    return F.cross_entropy(
        shifted_logits.float().reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction=reduction,
    )


def _causal_next_token_nll_sum(logits: Tensor, labels: Tensor) -> Tensor:
    """Return the FP32 sum reduction used for durable per-block evidence."""
    shifted_logits, shifted_labels, _ = _validated_shifted_tokens(logits, labels)
    return _causal_nll_from_shifted(shifted_logits, shifted_labels, reduction="sum")


def _forward_logits(
    model: nn.Module,
    input_ids: Tensor,
    kernel_bank: KernelParameterBank,
    forward_fn: ForwardFunction | None,
) -> Tensor:
    output = (
        run_attention_validation_forward(model, input_ids=input_ids)
        if forward_fn is None
        else run_attention_validation_forward(forward_fn, model, input_ids, kernel_bank)
    )
    if isinstance(output, Tensor):
        logits = output
    elif isinstance(output, Mapping):
        logits = output.get("logits")
    else:
        logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError(
            "model forward must return logits as a Tensor or a logits field"
        )
    return logits


def _backward_with_attention_validation(loss: Tensor) -> None:
    """Run one backward chain in a fresh normal attention-validation session."""
    with attention_validation_session():
        loss.backward()


def _require_validation_count(config: _TrainingConfig) -> None:
    value = getattr(config, "validation_blocks", None)
    if type(value) is not int or value != _PILOT_VALIDATION_BLOCKS:
        raise ValueError("pilot validation_blocks must be exactly 32")


def _require_train_blocks(config: _TrainingConfig) -> int:
    value = getattr(config, "train_blocks", None)
    if type(value) is not int or value < 0:
        raise ValueError("config.train_blocks must be a non-negative integer")
    return value


def _require_positive_interval(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_positive_finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


def _validation_pass(source: ValidationBlockSource) -> Iterable[object]:
    return source() if callable(source) else source


def _require_reiterable_validation_source(source: object) -> None:
    if callable(source):
        return
    if isinstance(source, (str, bytes)) or isinstance(source, Iterator):
        raise TypeError(
            "validation_blocks must be a re-iterable container or factory returning an iterable"
        )
    try:
        length = len(source)
        first_iterator = iter(source)
        second_iterator = iter(source)
    except TypeError as exc:
        raise TypeError(
            "validation_blocks must be a re-iterable container or factory returning an iterable"
        ) from exc
    if type(length) is not int or length < 0 or first_iterator is second_iterator:
        raise TypeError(
            "validation_blocks must be a re-iterable container or factory returning an iterable"
        )


def _set_training_mode(model: nn.Module, kernel_bank: KernelParameterBank) -> None:
    """Make every optimization forward explicit about module training mode."""
    model.train()
    kernel_bank.train()


def _train_block_at(source: TrainBlockSource, index: int) -> object:
    if callable(source):
        return source(index)
    if index >= len(source):
        raise ValueError(f"train block source ended before required cursor {index}")
    return source[index]


def _parse_resume_cursor(cursor: Mapping[str, object] | None, total_steps: int) -> int:
    if cursor is None:
        return 0
    if not isinstance(cursor, Mapping):
        raise TypeError("resume_cursor must be a mapping")
    value = cursor.get("next_train_block")
    if type(value) is not int or not 0 <= value <= total_steps:
        raise ValueError(
            "resume_cursor.next_train_block must be in the configured train-block range"
        )
    return value


def _collect_trainables(
    model: nn.Module, kernel_bank: KernelParameterBank
) -> dict[str, nn.Parameter]:
    result: dict[str, nn.Parameter] = {}
    model_modules = dict(model.named_modules())
    kernel_parameter_ids = {id(parameter) for parameter in kernel_bank.parameters()}
    for name, parameter in model.named_parameters():
        if id(parameter) in kernel_parameter_ids:
            continue
        if not parameter.requires_grad:
            continue
        module_name, separator, parameter_name = name.rpartition(".")
        owner = model_modules.get(module_name) if separator else None
        if (
            not isinstance(owner, LoRAAdapter)
            or parameter_name not in {"A", "B"}
            or getattr(owner, parameter_name) is not parameter
        ):
            raise ValueError(
                f"model trainables must contain only registered LoRA A/B parameters; found {name}"
            )
        result[name] = parameter
    for name, parameter in kernel_bank.named_parameters():
        if parameter.requires_grad:
            full_name = f"kernel_bank.{name}"
            if full_name in result:
                raise ValueError(f"duplicate trainable parameter name: {full_name}")
            result[full_name] = parameter
    if not result:
        raise ValueError("training requires at least one trainable parameter")
    return dict(sorted(result.items()))


def _validate_optimizer_parameters(
    optimizer: Optimizer,
    trainables: Mapping[str, nn.Parameter],
    *,
    error_message: str = "optimizer parameters must exactly match allowed LoRA and kernel trainables",
) -> None:
    expected = {id(parameter) for parameter in trainables.values()}
    actual: list[int] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise ValueError(
                f"optimizer parameter group {group_index} has no params list"
            )
        actual.extend((id(parameter) for parameter in parameters))
    if len(actual) != len(set(actual)):
        raise ValueError("optimizer contains duplicate trainable parameters")
    if set(actual) != expected:
        raise ValueError(error_message)


def _validate_expected_lora_b_names(
    model: nn.Module,
    trainables: Mapping[str, nn.Parameter],
    expected_names: Sequence[str] | None,
) -> tuple[str, ...]:
    actual = tuple(
        (name for name in trainables if name.endswith(f".{ADAPTER_MODULE_NAME}.B"))
    )
    if not actual:
        raise ValueError("training requires registered LoRA B trainables")
    modules = dict(model.named_modules())
    for name in actual:
        module_name = name.rsplit(".", 1)[0]
        module = modules.get(module_name)
        if not isinstance(module, LoRAAdapter):
            raise ValueError(f"LoRA B trainable is not owned by a LoRAAdapter: {name}")
        if module.B is not trainables[name]:
            raise ValueError(
                f"LoRA B trainable does not match its adapter parameter: {name}"
            )
        paired_a_name = f"{module_name}.A"
        if paired_a_name not in trainables or module.A is not trainables[paired_a_name]:
            raise ValueError(
                f"LoRA B trainable is missing its paired adapter A: {name}"
            )
    if expected_names is None:
        return actual
    if any((not isinstance(name, str) or not name for name in expected_names)):
        raise TypeError("expected_lora_b_names must contain non-empty strings")
    expected = tuple(expected_names)
    if len(expected) != len(set(expected)) or set(expected) != set(actual):
        raise ValueError(
            "expected_lora_b_names must exactly name every trainable LoRA B"
        )
    return expected


def _prepare_lora_early_gate(
    model: nn.Module,
    trainables: Mapping[str, nn.Parameter],
    expected_names: Sequence[str] | None,
) -> object:
    return _validate_expected_lora_b_names(model, trainables, expected_names)


def _enforce_lora_early_gate(
    diagnostics: Sequence[Mapping[str, object]], step: int, state: object
) -> None:
    training_diagnostics.enforce_early_update_contract(
        diagnostics, step=step, expected_lora_b_names=cast(Sequence[str], state)
    )


def _learning_rate(optimizer: Optimizer) -> float:
    values = [group.get("lr") for group in optimizer.param_groups]
    if not values or not isinstance(values[0], (int, float)):
        raise ValueError("optimizer must expose a numeric learning rate")
    return float(values[0])


def _max_memory_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def _clock_value(clock: Clock) -> float:
    value = clock()
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("clock must return a finite numeric value")
    return float(value)


def _validation_fields(result: ValidationResult) -> dict[str, object]:
    return {
        "token_count": result.token_count,
        "total_nll": result.total_nll,
        "mean_nll": result.mean_nll,
        "blocks": [record.as_dict() for record in result.block_records],
    }


def _module_modes(module: nn.Module) -> tuple[tuple[nn.Module, bool], ...]:
    """Snapshot each module's mode, including intentionally mixed subtrees."""
    return tuple(((child, child.training) for child in module.modules()))


def _restore_module_modes(snapshot: Sequence[tuple[nn.Module, bool]]) -> None:
    for module, training in snapshot:
        module.training = training


def _start_run(store: RunStateSink) -> None:
    try:
        status = store.read()
    except FileNotFoundError:
        store.create()
        status = store.read()
    state = status["state"]
    if state == "created":
        store.transition("running")
    elif state != "running":
        raise ValueError(
            f"training can start only from created or running status, got {state!r}"
        )


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = None
    return state


def _restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_cpu = state["torch_cpu"]
    if not isinstance(torch_cpu, Tensor):
        raise TypeError("captured torch CPU RNG state is invalid")
    torch.set_rng_state(torch_cpu)
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not isinstance(cuda_state, list) or not all(
            (isinstance(item, Tensor) for item in cuda_state)
        ):
            raise TypeError("captured CUDA RNG state is invalid")
        torch.cuda.set_rng_state_all(cuda_state)


def _persist_failure(
    *,
    status_store: RunStateSink,
    metrics_writer: JsonlWriter,
    failure: TrainingFailure,
    original_exception: BaseException,
) -> None:
    details: dict[str, object] = {
        "exception_type": type(original_exception).__name__,
        "exception_message": str(failure),
        "phase": failure.phase,
        "step": failure.step,
        "traceback": traceback.format_exc(),
    }
    if failure.parameter is not None:
        details["parameter"] = failure.parameter
    if failure.value is not None:
        details["value"] = repr(failure.value)
    try:
        metrics_writer.write({"event": "failure", **details})
    finally:
        try:
            status_store.transition("failed", failure=details)
        except (FileNotFoundError, ValueError):
            pass


def _persist_interruption(
    *, status_store: RunStateSink, metrics_writer: JsonlWriter, phase: str, step: int
) -> None:
    details = {"event": "interrupted", "phase": phase, "step": step}
    try:
        metrics_writer.write(details)
    finally:
        try:
            status_store.transition("interrupted", phase=phase, step=step)
        except (FileNotFoundError, ValueError):
            pass
