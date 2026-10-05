#!/usr/bin/env python3
"""Explicitly build and admit the host-only HD cuBLAS compatibility library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch
from torch.utils import cpp_extension


load = cpp_extension.load


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "oal_attention" / "csrc" / "hd_cublas_compat.cpp"
_SCHEMA = "hd_cublas_compat_artifact_v1"


def _canonical_device(device: torch.device | str) -> torch.device:
    """Resolve ``cuda`` to the scheduler-selected current visible device."""
    try:
        canonical = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("build device is invalid") from error
    if canonical.type == "cuda" and canonical.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return canonical


def _stable_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checked(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def _compiler_identity() -> dict[str, str]:
    compiler = os.environ.get("CXX") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("a C++ compiler is required")
    resolved = str(Path(compiler).resolve())
    return {
        "path": resolved,
        "version": _run_checked((resolved, "--version")).splitlines()[0],
    }


def _loaded_cublas_paths() -> tuple[Path, ...]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("cuBLAS provider discovery requires Linux /proc/self/maps")
    paths: set[Path] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        candidate = line.rsplit(maxsplit=1)[-1]
        if "libcublas.so" not in candidate or "libcublasLt" in candidate:
            continue
        candidate = candidate.removesuffix(" (deleted)")
        path = Path(candidate)
        if path.is_file():
            paths.add(path.resolve())
    return tuple(sorted(paths))


def _loaded_cuda_runtime_paths() -> tuple[Path, ...]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError(
            "CUDA runtime provider discovery requires Linux /proc/self/maps"
        )
    paths: set[Path] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        candidate = line.rsplit(maxsplit=1)[-1].removesuffix(" (deleted)")
        if "libcudart.so" not in candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            paths.add(path.resolve())
    return tuple(sorted(paths))


def _resolve_torch_cublas_provider(device: torch.device) -> Path:
    torch.cuda.set_device(device)
    left = torch.ones((1, 2, 2), dtype=torch.float32, device=device)
    right = torch.ones((1, 2, 2), dtype=torch.float32, device=device)
    torch.bmm(left, right)
    torch.cuda.synchronize(device)
    del left, right
    providers = _loaded_cublas_paths()
    if len(providers) != 1:
        raise RuntimeError(
            f"expected exactly one loaded Torch cuBLAS provider, found {providers}"
        )
    return providers[0]


def _resolve_torch_cuda_runtime_provider(device: torch.device) -> Path:
    torch.cuda.set_device(device)
    providers = _loaded_cuda_runtime_paths()
    if len(providers) != 1:
        raise RuntimeError(
            f"expected exactly one loaded Torch CUDA runtime provider, found {providers}"
        )
    return providers[0]


def _readelf_dynamic(library: Path) -> str:
    readelf = shutil.which("readelf")
    if readelf is None:
        raise RuntimeError("readelf is required for post-build ELF admission")
    return _run_checked((readelf, "-d", str(library)))


def _elf_soname(dynamic_section: str, *, fallback: str) -> str:
    for line in dynamic_section.splitlines():
        if "(SONAME)" in line and "[" in line and "]" in line:
            return line.split("[", 1)[1].split("]", 1)[0]
    return fallback


def _extension_load_kwargs(
    *,
    source: Path,
    build_directory: Path,
    provider: Path,
    cuda_runtime_provider: Path,
    cublas_header_directory: Path,
    cuda_header_directory: Path,
    cuda_crt_include_directory: Path,
    extension_name: str,
) -> dict[str, object]:
    source = source.resolve()
    build_directory = build_directory.resolve()
    provider = provider.resolve()
    cuda_runtime_provider = cuda_runtime_provider.resolve()
    cublas_header_directory = cublas_header_directory.resolve()
    cuda_header_directory = cuda_header_directory.resolve()
    cuda_crt_include_directory = cuda_crt_include_directory.resolve()
    include_paths = list(
        dict.fromkeys(
            str(path)
            for path in (
                cublas_header_directory,
                cuda_header_directory,
                cuda_crt_include_directory,
            )
        )
    )
    return {
        "name": extension_name,
        "sources": [str(source)],
        "extra_cflags": ["-O3", "-std=c++17"],
        "extra_ldflags": [
            f"-L{cpp_extension.TORCH_LIB_PATH}",
            "-lc10_cuda",
            "-ltorch_cuda",
            str(provider),
            "-Wl,--no-as-needed",
            str(cuda_runtime_provider),
            "-Wl,--as-needed",
            f"-Wl,-rpath,{provider.parent}",
            f"-Wl,-rpath,{cuda_runtime_provider.parent}",
            "-ldl",
        ],
        "extra_include_paths": include_paths,
        "build_directory": str(build_directory),
        "with_cuda": False,
        "is_python_module": False,
        "verbose": True,
    }


def _validate_postbuild_evidence(
    *,
    dynamic_section: str,
    provider: Path,
    cuda_runtime_provider: Path,
    cuda_runtime_soname: str,
    child_metadata: Mapping[str, object],
) -> None:
    provider = provider.resolve()
    cuda_runtime_provider = cuda_runtime_provider.resolve()
    soname_match = re.match(r"(libcublas\.so\.\d+)", provider.name)
    soname = soname_match.group(1) if soname_match is not None else provider.name
    if "(NEEDED)" not in dynamic_section or soname not in dynamic_section:
        raise RuntimeError("artifact DT_NEEDED does not bind the selected cuBLAS soname")
    needed_sonames = set(
        re.findall(r"\(NEEDED\).*?Shared library: \[([^]]+)\]", dynamic_section)
    )
    cuda_runtime_dependencies = {
        name for name in needed_sonames if name.startswith("libcudart.so")
    }
    if cuda_runtime_dependencies != {cuda_runtime_soname}:
        raise RuntimeError(
            "artifact CUDA runtime DT_NEEDED does not bind the selected soname"
        )
    if (
        "(RUNPATH)" not in dynamic_section
        and "(RPATH)" not in dynamic_section
    ) or any(
        str(directory) not in dynamic_section
        for directory in (provider.parent, cuda_runtime_provider.parent)
    ):
        raise RuntimeError(
            "artifact RPATH/RUNPATH does not contain the selected provider directory"
        )
    observed = child_metadata.get("provider_realpath")
    if not isinstance(observed, str) or Path(observed).resolve() != provider:
        raise RuntimeError("post-load cuBLAS provider does not match the build provider")
    version = child_metadata.get("runtime_cublas_version")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise RuntimeError("post-load cuBLAS runtime version is invalid")


def _postbuild_child(
    *,
    device_name: str,
    library: Path,
    output: Path,
) -> int:
    device = _canonical_device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("post-build validation requires CUDA")
    provider = _resolve_torch_cublas_provider(device)
    cuda_runtime_provider = _resolve_torch_cuda_runtime_provider(device)
    torch.ops.load_library(str(library.resolve()))
    metadata = json.loads(torch.ops.hd_cublas_compat.runtime_metadata())
    metadata["torch_provider_before_load"] = str(provider)
    metadata["torch_cuda_runtime_before_load"] = str(cuda_runtime_provider)
    output.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return 0


def _run_postbuild_child(
    *,
    device: torch.device,
    library: Path,
    build_directory: Path,
) -> dict[str, object]:
    output = build_directory / "postbuild_metadata.json"
    if output.exists():
        output.unlink()
    subprocess.run(
        (
            sys.executable,
            str(Path(__file__).resolve()),
            "--postbuild-child",
            "--device",
            str(device),
            "--library",
            str(library),
            "--child-output",
            str(output),
        ),
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("post-build child metadata is invalid")
    return payload


def _is_cuda_toolkit_home(path: Path) -> bool:
    include = path / "include"
    return (include / "cuda.h").is_file() and (include / "cuda_runtime_api.h").is_file()


def _cuda_toolkit_candidates(
    provider: Path | None = None,
    *,
    cuda_runtime_provider: Path | None = None,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if cuda_runtime_provider is not None:
        candidates.append(cuda_runtime_provider.resolve().parent.parent)
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        configured = os.environ.get(variable)
        if configured:
            candidates.append(Path(configured))
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)
    configured = cpp_extension.CUDA_HOME
    if configured:
        candidates.append(Path(configured))
    if provider is not None:
        candidates.append(provider.resolve().parent.parent.parent / "cuda_runtime")
    if torch.version.cuda:
        candidates.append(Path(f"/usr/local/cuda-{torch.version.cuda}"))
    candidates.append(Path("/usr/local/cuda"))

    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _resolve_cuda_toolkit_home(
    provider: Path | None = None,
    *,
    cuda_runtime_provider: Path | None = None,
) -> Path:
    candidates = _cuda_toolkit_candidates(
        provider,
        cuda_runtime_provider=cuda_runtime_provider,
    )
    for candidate in candidates:
        if _is_cuda_toolkit_home(candidate):
            # torch's extension loader consults this module global, while an
            # activated environment is not required to export CUDA_HOME.
            cpp_extension.CUDA_HOME = str(candidate)
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "CUDA runtime headers (include/cuda.h and cuda_runtime_api.h) are required "
        "for the explicit build "
        f"step; searched: {searched or '<none>'}"
    )


def _resolve_cublas_header_directory(provider: Path, cuda_home: Path) -> Path:
    candidates = (provider.parent.parent / "include", cuda_home / "include")
    for candidate in candidates:
        if (candidate / "cublas_v2.h").is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "cuBLAS headers (include/cublas_v2.h) are required for the explicit build "
        f"step; searched: {searched}"
    )


def _resolve_cuda_crt_include_directory(
    provider: Path,
    cuda_home: Path,
    *,
    cuda_runtime_provider: Path | None = None,
) -> Path:
    nvidia_root = provider.resolve().parent.parent.parent
    candidates = [cuda_home / "include", Path(sys.prefix) / "include"]
    if nvidia_root.is_dir():
        candidates.extend(sorted(nvidia_root.glob("*/include")))
    candidates.extend(
        toolkit / "include"
        for toolkit in _cuda_toolkit_candidates(
            provider,
            cuda_runtime_provider=cuda_runtime_provider,
        )
    )
    for candidate in candidates:
        if (candidate / "crt" / "host_defines.h").is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "CUDA CRT header crt/host_defines.h is required for the explicit build "
        f"step; searched: {searched}"
    )


def _cuda_header_version(cuda_home: Path) -> str:
    version_file = cuda_home / "version.json"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip()
    version_text = cuda_home / "version.txt"
    if version_text.is_file():
        return version_text.read_text(encoding="utf-8").strip()
    for header_name, macro in (
        ("cuda_version.h", "CUDA_VERSION"),
        ("cuda.h", "CUDA_VERSION"),
        ("cuda_runtime_api.h", "CUDART_VERSION"),
    ):
        header = cuda_home / "include" / header_name
        if not header.is_file():
            continue
        match = re.search(
            rf"^\s*#\s*define\s+{macro}\s+(\d+)\s*$",
            header.read_text(encoding="utf-8", errors="replace"),
            flags=re.MULTILINE,
        )
        if match is not None:
            return f"{macro}={match.group(1)}"
    nvcc = cuda_home / "bin" / "nvcc"
    if nvcc.is_file():
        return _run_checked((str(nvcc), "--version"))
    raise RuntimeError("CUDA toolkit header version could not be determined")


def _find_built_library(build_directory: Path, extension_name: str) -> Path:
    candidates = tuple(build_directory.glob(f"{extension_name}*.so"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one built shared object, found {candidates}")
    return candidates[0].resolve()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build(*, device: torch.device, output_dir: Path) -> Path:
    device = _canonical_device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("HD cuBLAS compatibility build requires CUDA")
    if not _SOURCE.is_file():
        raise RuntimeError("HD cuBLAS compatibility C++ source is missing")
    if shutil.which("ninja") is None:
        raise RuntimeError("ninja is required for the explicit build step")
    provider = _resolve_torch_cublas_provider(device)
    cuda_runtime_provider = _resolve_torch_cuda_runtime_provider(device)
    cuda_home = _resolve_cuda_toolkit_home(
        provider,
        cuda_runtime_provider=cuda_runtime_provider,
    )
    cublas_header_directory = _resolve_cublas_header_directory(provider, cuda_home)
    cuda_crt_include_directory = _resolve_cuda_crt_include_directory(
        provider,
        cuda_home,
        cuda_runtime_provider=cuda_runtime_provider,
    )
    provider_dynamic = _readelf_dynamic(provider)
    cuda_runtime_dynamic = _readelf_dynamic(cuda_runtime_provider)
    compiler = _compiler_identity()
    builder_path = Path(__file__).resolve()
    prebuild_payload: dict[str, object] = {
        "schema": "hd_cublas_compat_prebuild_v1",
        "source_sha256": _file_sha256(_SOURCE),
        "builder_sha256": _file_sha256(builder_path),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_cache_tag": sys.implementation.cache_tag,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler": compiler,
        "cuda_header_version": _cuda_header_version(cuda_home),
        "provider_realpath": str(provider),
        "provider_soname": _elf_soname(provider_dynamic, fallback=provider.name),
        "provider_sha256": _file_sha256(provider),
        "cuda_runtime_realpath": str(cuda_runtime_provider),
        "cuda_runtime_soname": _elf_soname(
            cuda_runtime_dynamic,
            fallback=cuda_runtime_provider.name,
        ),
        "cuda_runtime_sha256": _file_sha256(cuda_runtime_provider),
        "operator_schema": (
            "bmm_fp32(Tensor left, Tensor right, *, Tensor(a!) out) -> ()"
        ),
        "cflags": ["-O3", "-std=c++17"],
    }
    prebuild_key = _payload_sha256(prebuild_payload)
    build_directory = (output_dir / prebuild_key).resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    extension_name = f"hd_cublas_compat_{prebuild_key[:16]}"
    kwargs = _extension_load_kwargs(
        source=_SOURCE,
        build_directory=build_directory,
        provider=provider,
        cuda_runtime_provider=cuda_runtime_provider,
        cublas_header_directory=cublas_header_directory,
        cuda_header_directory=cuda_home / "include",
        cuda_crt_include_directory=cuda_crt_include_directory,
        extension_name=extension_name,
    )
    load(**kwargs)
    library = _find_built_library(build_directory, extension_name)
    try:
        dynamic_section = _readelf_dynamic(library)
        child_metadata = _run_postbuild_child(
            device=device,
            library=library,
            build_directory=build_directory,
        )
        _validate_postbuild_evidence(
            dynamic_section=dynamic_section,
            provider=provider,
            cuda_runtime_provider=cuda_runtime_provider,
            cuda_runtime_soname=str(prebuild_payload["cuda_runtime_soname"]),
            child_metadata=child_metadata,
        )
    except BaseException:
        library.unlink(missing_ok=True)
        (build_directory / "manifest.json.tmp").unlink(missing_ok=True)
        raise

    shared_object_sha256 = _file_sha256(library)
    artifact_identity = _payload_sha256(
        {
            "prebuild_cache_key": prebuild_key,
            "shared_object_sha256": shared_object_sha256,
        }
    )
    manifest: dict[str, object] = {
        "schema_version": _SCHEMA,
        "prebuild_cache_key": prebuild_key,
        "artifact_identity_sha256": artifact_identity,
        "library_path": str(library),
        "shared_object_sha256": shared_object_sha256,
        "build_provenance": prebuild_payload,
        "load_compatibility": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "operator_schema": prebuild_payload["operator_schema"],
            "provider_realpath": str(provider),
            "provider_soname": prebuild_payload["provider_soname"],
            "provider_sha256": prebuild_payload["provider_sha256"],
            "cuda_runtime_realpath": prebuild_payload["cuda_runtime_realpath"],
            "cuda_runtime_soname": prebuild_payload["cuda_runtime_soname"],
            "cuda_runtime_sha256": prebuild_payload["cuda_runtime_sha256"],
            "runtime_cublas_version": child_metadata["runtime_cublas_version"],
        },
        "postbuild_metadata": child_metadata,
    }
    manifest["manifest_identity_sha256"] = _payload_sha256(manifest)
    manifest_path = build_directory / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    current = output_dir.resolve() / "current" / "manifest.json"
    _atomic_write_json(current, manifest)
    return current


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--postbuild-child", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--child-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.postbuild_child:
        if args.library is None or args.child_output is None:
            raise SystemExit("post-build child requires --library and --child-output")
        return _postbuild_child(
            device_name=args.device,
            library=args.library,
            output=args.child_output,
        )
    if args.output_dir is None:
        raise SystemExit("build requires --output-dir")
    manifest = _build(device=_canonical_device(args.device), output_dir=args.output_dir)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
