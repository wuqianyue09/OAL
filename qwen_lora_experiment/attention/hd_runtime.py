"""Narrow experiment-side bridge to the operator package HD attention families."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
import importlib
from pathlib import Path
import torch
from ..config import PilotConfig

FamilyCallable = Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class HDAttentionRuntime:
    """Initialized public family callables plus one canonical option mapping."""

    grouped_callable: FamilyCallable
    options: dict[str, object]
    requested_options: dict[str, object]
    prepare_grouped_callable: Callable[..., object]
    run_prepared_grouped_callable: FamilyCallable
    _grouped_prepared: dict[object, list[tuple[tuple[object, ...], object]]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def _options_for(self, q: torch.Tensor) -> dict[str, object]:
        options = dict(self.options)
        block_count = (q.shape[2] + int(options["token_block"]) - 1) // int(
            options["token_block"]
        )
        options["feature_wave_blocks"] = min(
            int(options["feature_wave_blocks"]), block_count
        )
        return options

    def grouped(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dim_groups: torch.Tensor,
        factor: torch.Tensor,
        cache_key: object | None = None,
        **family: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        options = self._options_for(q)
        if cache_key is None:
            return self.grouped_callable(
                q, k, v, dim_groups, factor, **family, **options
            )
        signature = (
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            str(q.dtype),
            str(q.device),
            torch.is_grad_enabled(),
            q.requires_grad,
            k.requires_grad,
            v.requires_grad,
            tuple(factor.shape),
            str(factor.dtype),
            str(factor.device),
            factor.requires_grad,
            tuple(dim_groups.shape),
            str(dim_groups.dtype),
            tuple(sorted((*options.items(), *family.items()))),
        )
        entries = self._grouped_prepared.setdefault(cache_key, [])
        prepared = next(
            (
                candidate
                for (candidate_signature, candidate) in entries
                if candidate_signature == signature
            ),
            None,
        )
        if prepared is None:
            prepared = self.prepare_grouped_callable(
                q, k, v, dim_groups, factor, **family, **options
            )
            entries.append((signature, prepared))
            if len(entries) > 2:
                del entries[0]
        return self.run_prepared_grouped_callable(prepared, q, k, v, factor)

    def execution_options(self) -> dict[str, object]:
        return dict(self.options)

    def requested_execution_options(self) -> dict[str, object]:
        return dict(self.requested_options)


_PLAN_OPTION_FIELDS = (
    "token_block",
    "feature_wave_blocks",
    "memory_budget_bytes",
    "precision",
    "key_feature_impl",
    "key_fold_impl",
    "query_feature_impl",
    "query_feature_token_tile",
    "query_fold_impl",
    "query_fold_token_tile",
    "query_fold_input",
    "query_gradient_flow",
    "backward_schedule",
    "gradient_staging",
    "forward_normalize_impl",
    "backward_normalize_impl",
    "kv_cross_impl",
    "save_local_score",
    "key_retention",
    "feature_padding",
)


def _canonical_training_options(
    config: PilotConfig,
    target_device: torch.device,
    options: dict[str, object],
    plan_builder: Callable[..., object],
) -> dict[str, object]:
    """Validate the exact training geometry with the operator package's HD planner."""
    planning_device = (
        target_device
        if target_device.type != "cuda" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    dtype = torch.bfloat16
    geometry = config.geometry
    shape_q = (1, geometry.num_query_heads, config.sequence_length, geometry.head_dim)
    shape_kv = (1, geometry.num_kv_heads, config.sequence_length, geometry.head_dim)
    q = torch.empty(shape_q, dtype=dtype, device=planning_device)
    k = torch.empty(shape_kv, dtype=dtype, device=planning_device)
    v = torch.empty(shape_kv, dtype=dtype, device=planning_device)
    requested = dict(options)
    block_count = (config.sequence_length + int(requested["token_block"]) - 1) // int(
        requested["token_block"]
    )
    requested["feature_wave_blocks"] = min(
        int(requested["feature_wave_blocks"]), block_count
    )
    coefficient_grad = True
    plan = plan_builder(
        q,
        k,
        v,
        **requested,
        output_dtype=dtype,
        requested_gradient_mask=(
            True,
            True,
            True,
            coefficient_grad,
            coefficient_grad,
            coefficient_grad,
        ),
    )
    try:
        return {name: getattr(plan, name) for name in _PLAN_OPTION_FIELDS}
    except AttributeError as exc:
        raise TypeError("OAL HD planner returned incomplete canonical options") from exc


def initialize_hd_runtime(
    config: PilotConfig,
    device: torch.device | str,
    *,
    manifest_path: str | Path | None = None,
    runtime_probe_path: str | Path | None = None,
    importer: Callable[[str], object] = importlib.import_module,
) -> HDAttentionRuntime | None:
    """Initialize HD once at setup; legacy never imports the operator HD path."""
    config.validate()
    if config.attention_backend == "legacy":
        return None
    target_device = torch.device(device)
    requested_options = config.resolved_hd_options.to_dict()
    if requested_options["precision"] == "bf16_tensorcore":
        if (
            target_device.type != "cuda"
            or manifest_path is None
            or runtime_probe_path is None
        ):
            raise ValueError(
                "bf16_tensorcore requires a CUDA device plus explicit manifest_path and runtime_probe_path"
            )
        compat = importer("oal_attention.hd_cublas_compat")
        loader = getattr(compat, "load_hd_cublas_compat", None)
        if not callable(loader):
            raise TypeError("oal_attention.hd_cublas_compat loader is not callable")
        loader(
            Path(manifest_path),
            Path(runtime_probe_path),
            target_device,
            identity_policy="semantic_compat",
        )
    family_module = importer("oal_attention.hd_attention")
    adapter_module = importer("oal_attention.hd_block_gemm_adapters")
    grouped = getattr(family_module, "grouped_causal_attention", None)
    prepare_grouped = getattr(adapter_module, "prepare_grouped_hd_block_gemm", None)
    run_prepared_grouped = getattr(
        adapter_module, "run_prepared_grouped_hd_block_gemm", None
    )
    plan_builder = getattr(adapter_module, "build_hd_block_gemm_plan", None)
    if not all(
        (
            callable(value)
            for value in (grouped, prepare_grouped, run_prepared_grouped, plan_builder)
        )
    ):
        raise TypeError("oal_attention.hd_attention family entry points are incomplete")
    options = _canonical_training_options(
        config, target_device, requested_options, plan_builder
    )
    return HDAttentionRuntime(
        grouped_callable=grouped,
        prepare_grouped_callable=prepare_grouped,
        run_prepared_grouped_callable=run_prepared_grouped,
        options=options,
        requested_options=requested_options,
    )


__all__ = ("HDAttentionRuntime", "initialize_hd_runtime")
