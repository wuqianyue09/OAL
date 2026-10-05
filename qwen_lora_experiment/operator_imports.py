"""Repository-local OAL imports and historical callable-name reconciliation.

The uploaded GPU host does not need a Git checkout, but live runs must still
load the OAL implementation named by the operator rather than an unrelated
editable or site-packages installation.
"""

from __future__ import annotations
import importlib
import os
from pathlib import Path
import sys
from collections.abc import Mapping


class OperatorImportRootError(ValueError):
    """The requested OAL source tree cannot be used."""


OPERATOR_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_HISTORICAL_OPERATOR_CALLABLES = {
    "flash_taylor_attn.grouped_quadratic": "oal_attention.oal_attention",
    "flash_taylor_attn.hd_block_gemm_adapters.run_prepared_grouped_hd_block_gemm": "oal_attention.hd_block_gemm_adapters.run_prepared_grouped_hd_block_gemm",
}


def canonical_operator_callable(value: object) -> object:
    """Reconcile only the public callable names used by existing experiments."""
    if isinstance(value, str):
        return _HISTORICAL_OPERATOR_CALLABLES.get(value, value)
    return value


def normalize_operator_identity(value: Mapping[str, object]) -> dict[str, object]:
    """Read renamed callable/fixture fields without rewriting frozen artifacts."""
    result = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            item = normalize_operator_identity(item)
        elif key in {"callable", "public_callable", "base_public_callable"}:
            item = canonical_operator_callable(item)
        result[key] = item
    if "flash_taylor_attn" in result and "oal_attention" not in result:
        result["oal_attention"] = result.pop("flash_taylor_attn")
    return result


def resolve_operator_import_root(value: str | Path = OPERATOR_SOURCE_ROOT) -> Path:
    """Resolve a source tree containing the public OAL package."""
    if not isinstance(value, (str, Path)) or not str(value):
        raise OperatorImportRootError(
            "operator_root must be a non-empty source-tree path"
        )
    root = Path(value).expanduser().resolve()
    package_init = root / "oal_attention" / "__init__.py"
    if not root.is_dir() or not package_init.is_file():
        raise OperatorImportRootError(
            f"operator_root must contain oal_attention/__init__.py: {root}"
        )
    return root


def operator_import_environment(
    operator_root: str | Path, *, environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return a child environment whose first OAL lookup is ``operator_root``."""
    root = str(resolve_operator_import_root(operator_root))
    result = dict(os.environ if environment is None else environment)
    inherited = result.get("PYTHONPATH", "")
    entries = [
        entry for entry in inherited.split(os.pathsep) if entry and entry != root
    ]
    result["PYTHONPATH"] = os.pathsep.join((root, *entries))
    return result


def activate_operator_import_root(operator_root: str | Path) -> Path:
    """Pin subsequent OAL imports in this process to ``operator_root``.

    Refuse a preloaded package from another location instead of silently
    allowing one process to combine two operator source trees.
    """
    root = resolve_operator_import_root(operator_root)
    existing = sys.modules.get("oal_attention")
    existing_path = getattr(existing, "__file__", None)
    if isinstance(existing_path, str):
        try:
            Path(existing_path).resolve().relative_to(root)
        except ValueError as exc:
            raise OperatorImportRootError(
                f"oal_attention was already imported outside requested operator_root: {existing_path}"
            ) from exc
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    else:
        sys.path.remove(root_text)
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    return root
