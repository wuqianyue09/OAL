"""Dependency-free contracts for the read-only cais/mmlu evaluation endpoint."""

from __future__ import annotations
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile

MMLU_PROTOCOL = "cais_mmlu_5shot_answer_letter_v1"
ANSWER_LETTERS = ("A", "B", "C", "D")
MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)
MMLU_DEV_EXAMPLES_PER_SUBJECT = 5
MMLU_TOTAL_TEST_COUNT = 14042
MMLU_BUNDLE_SCHEMA = "cais_mmlu_5shot_bundle_v1"
MMLU_BUNDLE_DIRECTORY = "cais_mmlu_5shot_v1"
MMLU_DATASET_NAME = "cais/mmlu"
MMLU_BUNDLE_FILENAMES = ("mmlu_dev.jsonl", "mmlu_test.jsonl")
MMLU_ROW_FIELDS = ("id", "subject", "test_index", "question", "choices", "answer")
MMLU_SUBJECTS_SHA256 = (
    "00b25f22871494628f70e593b6d9f60ad7c7d51629606f88febe8697c2a7ad35"
)
MMLU_PROMPT_TEMPLATE = "The following are multiple choice questions (with answers) about {subject}.\n\n{demonstrations}\n\n{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
MMLU_PROMPT_TEMPLATE_SHA256 = (
    "c286d91bcd33166d0086f6205bb2ed3b5ef1932811a7985be3336a029a93d16a"
)
MMLU_PROMPT_GOLDEN_SHA256 = (
    "e26d0c8d1008ddfdc91ef0597f1ce94f29eef34a14c55bb17d74cd3dc57bd168"
)
MMLU_SUBJECT_SPLIT_COUNTS = {
    "abstract_algebra": (5, 100),
    "anatomy": (5, 135),
    "astronomy": (5, 152),
    "business_ethics": (5, 100),
    "clinical_knowledge": (5, 265),
    "college_biology": (5, 144),
    "college_chemistry": (5, 100),
    "college_computer_science": (5, 100),
    "college_mathematics": (5, 100),
    "college_medicine": (5, 173),
    "college_physics": (5, 102),
    "computer_security": (5, 100),
    "conceptual_physics": (5, 235),
    "econometrics": (5, 114),
    "electrical_engineering": (5, 145),
    "elementary_mathematics": (5, 378),
    "formal_logic": (5, 126),
    "global_facts": (5, 100),
    "high_school_biology": (5, 310),
    "high_school_chemistry": (5, 203),
    "high_school_computer_science": (5, 100),
    "high_school_european_history": (5, 165),
    "high_school_geography": (5, 198),
    "high_school_government_and_politics": (5, 193),
    "high_school_macroeconomics": (5, 390),
    "high_school_mathematics": (5, 270),
    "high_school_microeconomics": (5, 238),
    "high_school_physics": (5, 151),
    "high_school_psychology": (5, 545),
    "high_school_statistics": (5, 216),
    "high_school_us_history": (5, 204),
    "high_school_world_history": (5, 237),
    "human_aging": (5, 223),
    "human_sexuality": (5, 131),
    "international_law": (5, 121),
    "jurisprudence": (5, 108),
    "logical_fallacies": (5, 163),
    "machine_learning": (5, 112),
    "management": (5, 103),
    "marketing": (5, 234),
    "medical_genetics": (5, 100),
    "miscellaneous": (5, 783),
    "moral_disputes": (5, 346),
    "moral_scenarios": (5, 895),
    "nutrition": (5, 306),
    "philosophy": (5, 311),
    "prehistory": (5, 324),
    "professional_accounting": (5, 282),
    "professional_law": (5, 1534),
    "professional_medicine": (5, 272),
    "professional_psychology": (5, 612),
    "public_relations": (5, 110),
    "security_studies": (5, 245),
    "sociology": (5, 201),
    "us_foreign_policy": (5, 100),
    "virology": (5, 166),
    "world_religions": (5, 171),
}


def mmlu_answer_candidates() -> tuple[str, str, str, str]:
    """Return Qwen's four exact, one-token candidate continuations."""
    return tuple((f" {letter}" for letter in ANSWER_LETTERS))


def validate_mmlu_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Normalize MMLU source rows without modifying source-field strings.

    Prepared rows gain the immutable ``subject:test_index`` identifier when it
    is not supplied.  A supplied ID must agree with that deterministic value.
    """
    normalized: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"MMLU row {row_index} must be a mapping")
        unknown = [key for key in row if key not in MMLU_ROW_FIELDS]
        if unknown:
            rendered = ", ".join((repr(key) for key in unknown))
            raise ValueError(f"MMLU row {row_index} has unknown field(s): {rendered}")
        missing = [
            field for field in MMLU_ROW_FIELDS if field != "id" and field not in row
        ]
        if missing:
            raise ValueError(
                f"MMLU row {row_index} is missing required field(s): {', '.join(missing)}"
            )
        subject = row["subject"]
        if not isinstance(subject, str) or subject not in MMLU_SUBJECTS:
            raise ValueError(
                f"MMLU row {row_index}.subject must be one of the registered configs"
            )
        test_index = row["test_index"]
        if type(test_index) is not int or test_index < 0:
            raise ValueError(
                f"MMLU row {row_index}.test_index must be a non-negative integer"
            )
        expected_id = f"{subject}:{test_index}"
        identifier = row.get("id", expected_id)
        if not isinstance(identifier, str) or identifier != expected_id:
            raise ValueError(f"MMLU row {row_index}.id must equal {expected_id!r}")
        if identifier in seen_ids:
            raise ValueError(f"MMLU row {row_index} has duplicate id {identifier!r}")
        seen_ids.add(identifier)
        question = row["question"]
        if not isinstance(question, str):
            raise ValueError(f"MMLU row {row_index}.question must be a string")
        choices = row["choices"]
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise ValueError(
                f"MMLU row {row_index}.choices must be a sequence of four strings"
            )
        if len(choices) != len(ANSWER_LETTERS):
            raise ValueError(
                f"MMLU row {row_index}.choices must contain exactly four choices"
            )
        normalized_choices: list[str] = []
        for choice_index, choice in enumerate(choices):
            if not isinstance(choice, str):
                raise ValueError(
                    f"MMLU row {row_index}.choices[{choice_index}] must be a string"
                )
            if not choice:
                raise ValueError(
                    f"MMLU row {row_index}.choices[{choice_index}] must be non-empty"
                )
            normalized_choices.append(choice)
        answer = row["answer"]
        if type(answer) is not int or answer not in range(len(ANSWER_LETTERS)):
            raise ValueError(
                f"MMLU row {row_index}.answer must be an integer from 0 through 3"
            )
        normalized.append(
            {
                "id": identifier,
                "subject": subject,
                "test_index": test_index,
                "question": question,
                "choices": normalized_choices,
                "answer": answer,
            }
        )
    return normalized


def validate_mmlu_bundle(
    dev_rows: Iterable[Mapping[str, object]], test_rows: Iterable[Mapping[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    """Validate the immutable 57-subject, 5-shot MMLU bundle contract."""
    normalized_dev = validate_mmlu_rows(dev_rows)
    _require_exact_subject_registry(normalized_dev, split="development")
    dev_counts = Counter((str(row["subject"]) for row in normalized_dev))
    wrong_dev_counts = {
        subject: dev_counts[subject]
        for subject in MMLU_SUBJECTS
        if dev_counts[subject] != MMLU_DEV_EXAMPLES_PER_SUBJECT
    }
    if wrong_dev_counts:
        raise ValueError(
            f"MMLU development split requires exactly five examples per subject; observed {wrong_dev_counts}"
        )
    _require_canonical_subject_index_order(normalized_dev, split="development")
    normalized_test = validate_mmlu_rows(test_rows)
    _require_exact_subject_registry(normalized_test, split="test")
    if len(normalized_test) != MMLU_TOTAL_TEST_COUNT:
        raise ValueError(
            f"MMLU test split requires exactly {MMLU_TOTAL_TEST_COUNT} questions; observed {len(normalized_test)}"
        )
    observed_counts = {subject: 0 for subject in MMLU_SUBJECTS}
    for row in normalized_test:
        subject = row["subject"]
        assert isinstance(subject, str)
        observed_counts[subject] += 1
    _require_canonical_subject_index_order(normalized_test, split="test")
    return (normalized_dev, normalized_test, observed_counts)


def format_mmlu_prompt(
    subject: str,
    demonstrations: Iterable[Mapping[str, object]],
    question: Mapping[str, object],
) -> str:
    """Render the locked five-shot prompt with exact LF separators."""
    if not isinstance(subject, str) or subject not in MMLU_SUBJECTS:
        raise ValueError("MMLU prompt subject must be one of the registered configs")
    normalized_demonstrations = validate_mmlu_rows(demonstrations)
    if len(normalized_demonstrations) != MMLU_DEV_EXAMPLES_PER_SUBJECT:
        raise ValueError("MMLU prompt requires exactly five demonstrations")
    normalized_question = validate_mmlu_rows([question])
    for row in (*normalized_demonstrations, *normalized_question):
        if row["subject"] != subject:
            raise ValueError("MMLU prompt rows must match the requested subject")
    demonstrations_text = "\n\n".join(
        (_format_answered_question(row) for row in normalized_demonstrations)
    )
    row = normalized_question[0]
    return MMLU_PROMPT_TEMPLATE.format(
        subject=subject.replace("_", " "),
        demonstrations=demonstrations_text,
        question=row["question"],
        choices=row["choices"],
    )


def _require_exact_subject_registry(
    rows: Sequence[Mapping[str, object]], *, split: str
) -> None:
    observed = {row["subject"] for row in rows}
    missing = [subject for subject in MMLU_SUBJECTS if subject not in observed]
    if missing:
        raise ValueError(
            f"MMLU {split} split omits expected config(s): {', '.join(missing)}"
        )
    unexpected = observed.difference(MMLU_SUBJECTS)
    if unexpected:
        raise ValueError(
            f"MMLU {split} split has unknown config(s): {sorted(unexpected)}"
        )


def _require_canonical_subject_index_order(
    rows: Sequence[Mapping[str, object]], *, split: str
) -> None:
    previous_subject_index = -1
    next_test_index = 0
    for row in rows:
        subject = str(row["subject"])
        subject_index = MMLU_SUBJECTS.index(subject)
        if (
            subject_index < previous_subject_index
            or subject_index > previous_subject_index + 1
        ):
            raise ValueError(
                f"MMLU {split} rows must use canonical subject/index order"
            )
        if subject_index != previous_subject_index:
            previous_subject_index = subject_index
            next_test_index = 0
        if row["test_index"] != next_test_index:
            raise ValueError(
                f"MMLU {split} rows must use canonical subject/index order"
            )
        next_test_index += 1


def _format_answered_question(row: Mapping[str, object]) -> str:
    choices = row["choices"]
    answer = row["answer"]
    assert isinstance(choices, list) and type(answer) is int
    return f"{row['question']}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {ANSWER_LETTERS[answer]}"


MmluSourceLoader = Callable[[str, str, str | None], Iterable[Mapping[str, object]]]


def mmlu_bundle_path(data_root: str | Path) -> Path:
    """Return the independent, versioned local MMLU bundle directory."""
    return Path(data_root) / "mmlu" / MMLU_BUNDLE_DIRECTORY


def load_cais_mmlu_source(
    subject: str, split: str, requested_source_revision: str | None
) -> Iterable[Mapping[str, object]]:
    """Load one source split using an unverified requested revision selector."""
    if subject not in MMLU_SUBJECTS:
        raise ValueError(f"unknown MMLU subject: {subject!r}")
    if split not in ("dev", "test"):
        raise ValueError("MMLU source split must be dev or test")
    try:
        datasets = importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MMLU preparation requires the optional dependency; install it with python -m pip install --no-deps 'datasets>=3,<4'"
        ) from exc
    load_dataset = getattr(datasets, "load_dataset", None)
    if not callable(load_dataset):
        raise RuntimeError("installed datasets package does not provide load_dataset")
    kwargs: dict[str, object] = {}
    if requested_source_revision is not None:
        kwargs["revision"] = requested_source_revision
    return load_dataset(MMLU_DATASET_NAME, subject, split=split, **kwargs)


def installed_datasets_version() -> str:
    """Return the installed optional package version when a bundle is created."""
    try:
        datasets = importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MMLU preparation requires the optional dependency; install it with python -m pip install --no-deps 'datasets>=3,<4'"
        ) from exc
    version = getattr(datasets, "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError("installed datasets package does not expose a version")
    return version


def prepare_mmlu_bundle(
    data_root: str | Path,
    *,
    source_loader: MmluSourceLoader | None,
    tokenizer_identity: Mapping[str, object],
    datasets_version: str | None = None,
    requested_source_revision: str | None = None,
    verify_only: bool = False,
) -> dict[str, object]:
    """Create once or verify the local, canonical MMLU evaluation bundle.

    A completed directory is always validated and never reloaded from a remote
    source.  ``requested_source_revision`` is passed to ``datasets`` as a
    remote selector (for example, a branch, tag, or commit).  It is recorded
    as requested, never verified or represented as an immutable remote pin.
    """
    _validate_requested_source_revision(requested_source_revision)
    tokenizer = _json_object(tokenizer_identity, "tokenizer_identity")
    destination = mmlu_bundle_path(data_root)
    if destination.exists():
        manifest = validate_mmlu_bundle_directory(
            destination,
            tokenizer_identity=tokenizer,
            requested_source_revision=requested_source_revision,
            datasets_version=datasets_version,
        )
        return _bundle_result(destination, manifest, reused=True)
    if verify_only:
        raise FileNotFoundError(
            f"MMLU bundle does not exist for --verify-only: {destination}"
        )
    if source_loader is None:
        raise ValueError("source_loader is required to create a missing MMLU bundle")
    if datasets_version is None:
        datasets_version = installed_datasets_version()
    if not isinstance(datasets_version, str) or not datasets_version:
        raise ValueError(
            "datasets_version must be a non-empty string when creating a bundle"
        )
    dev_rows, test_rows, fingerprints = _load_canonical_source_rows(
        source_loader, requested_source_revision=requested_source_revision
    )
    manifest = _build_manifest(
        dev_rows,
        test_rows,
        tokenizer_identity=tokenizer,
        datasets_version=datasets_version,
        requested_source_revision=requested_source_revision,
        source_fingerprints=fingerprints,
    )
    try:
        _publish_bundle(destination, dev_rows, test_rows, manifest)
    except FileExistsError:
        manifest = validate_mmlu_bundle_directory(
            destination,
            tokenizer_identity=tokenizer,
            requested_source_revision=requested_source_revision,
            datasets_version=datasets_version,
        )
        return _bundle_result(destination, manifest, reused=True)
    verified = validate_mmlu_bundle_directory(
        destination,
        tokenizer_identity=tokenizer,
        requested_source_revision=requested_source_revision,
        datasets_version=datasets_version,
    )
    return _bundle_result(destination, verified, reused=False)


def validate_mmlu_bundle_directory(
    bundle_path: str | Path,
    *,
    tokenizer_identity: Mapping[str, object] | None = None,
    requested_source_revision: str | None = None,
    datasets_version: str | None = None,
) -> dict[str, object]:
    """Validate all bytes and immutable facts in one completed local bundle."""
    _validate_requested_source_revision(requested_source_revision)
    root = Path(bundle_path)
    if not root.is_dir():
        raise FileNotFoundError(f"MMLU bundle directory does not exist: {root}")
    paths = {
        name: root / name for name in (*MMLU_BUNDLE_FILENAMES, "mmlu_manifest.json")
    }
    missing = [name for (name, path) in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"MMLU bundle is incomplete; missing {', '.join(missing)}")
    try:
        raw_manifest = json.loads(
            paths["mmlu_manifest.json"].read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"MMLU manifest is invalid JSON: {exc.msg}") from exc
    manifest = _validate_manifest_structure(raw_manifest)
    if tokenizer_identity is not None and manifest["tokenizer"] != _json_object(
        tokenizer_identity, "tokenizer_identity"
    ):
        raise ValueError("MMLU manifest tokenizer identity does not match the request")
    if (
        requested_source_revision is not None
        and manifest["requested_source_revision"] != requested_source_revision
    ):
        raise ValueError(
            "MMLU manifest requested_source_revision does not match the request"
        )
    if (
        datasets_version is not None
        and manifest["datasets_version"] != datasets_version
    ):
        raise ValueError("MMLU manifest datasets_version does not match the request")
    for filename in MMLU_BUNDLE_FILENAMES:
        payload = paths[filename].read_bytes()
        record = manifest["files"][filename]
        assert isinstance(record, Mapping)
        if len(payload) != record["bytes"]:
            raise ValueError(f"MMLU {filename} byte count does not match its manifest")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"MMLU {filename} sha256 does not match its manifest")
    dev_rows = _read_jsonl(paths["mmlu_dev.jsonl"])
    test_rows = _read_jsonl(paths["mmlu_test.jsonl"])
    normalized_dev, normalized_test, observed_test = validate_mmlu_bundle(
        dev_rows, test_rows
    )
    if _jsonl_bytes(normalized_dev) != paths["mmlu_dev.jsonl"].read_bytes():
        raise ValueError("MMLU development rows are not canonical JSONL")
    if _jsonl_bytes(normalized_test) != paths["mmlu_test.jsonl"].read_bytes():
        raise ValueError("MMLU test rows are not canonical JSONL")
    observed_counts = _observed_counts(normalized_dev, observed_test)
    if manifest["observed_counts"] != observed_counts:
        raise ValueError("MMLU manifest observed_counts do not match local rows")
    if (
        _bundle_sha256(
            paths["mmlu_dev.jsonl"].read_bytes(), paths["mmlu_test.jsonl"].read_bytes()
        )
        != manifest["bundle_sha256"]
    ):
        raise ValueError("MMLU bundle_sha256 does not match local rows")
    return manifest


def _load_canonical_source_rows(
    source_loader: MmluSourceLoader, *, requested_source_revision: str | None
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, str | None]]
]:
    dev_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    fingerprints: dict[str, dict[str, str | None]] = {}
    for subject in MMLU_SUBJECTS:
        fingerprints[subject] = {}
        for split, target in (("dev", dev_rows), ("test", test_rows)):
            source = source_loader(subject, split, requested_source_revision)
            fingerprint = getattr(source, "_fingerprint", None)
            fingerprints[subject][split] = (
                fingerprint if isinstance(fingerprint, str) else None
            )
            for index, raw in enumerate(source):
                if not isinstance(raw, Mapping):
                    raise TypeError(
                        f"MMLU {subject}/{split} row {index} must be a mapping"
                    )
                target.append(
                    {
                        "id": f"{subject}:{index}",
                        "subject": subject,
                        "test_index": index,
                        "question": raw.get("question"),
                        "choices": raw.get("choices"),
                        "answer": raw.get("answer"),
                    }
                )
    normalized_dev, normalized_test, _ = validate_mmlu_bundle(dev_rows, test_rows)
    return (normalized_dev, normalized_test, fingerprints)


def _build_manifest(
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    *,
    tokenizer_identity: Mapping[str, object],
    datasets_version: str,
    requested_source_revision: str | None,
    source_fingerprints: Mapping[str, Mapping[str, str | None]],
) -> dict[str, object]:
    dev_bytes = _jsonl_bytes(dev_rows)
    test_bytes = _jsonl_bytes(test_rows)
    _, _, observed_test = validate_mmlu_bundle(dev_rows, test_rows)
    return {
        "schema": MMLU_BUNDLE_SCHEMA,
        "dataset_name": MMLU_DATASET_NAME,
        "requested_source_revision": requested_source_revision,
        "requested_source_revision_kind": "unverified_remote_selector",
        "source_fingerprints": _json_object(source_fingerprints, "source_fingerprints"),
        "datasets_version": datasets_version,
        "protocol": MMLU_PROTOCOL,
        "registry": list(MMLU_SUBJECTS),
        "observed_counts": _observed_counts(dev_rows, observed_test),
        "tokenizer": _json_object(tokenizer_identity, "tokenizer_identity"),
        "prompt_template_sha256": MMLU_PROMPT_TEMPLATE_SHA256,
        "files": {
            "mmlu_dev.jsonl": _file_record(dev_bytes),
            "mmlu_test.jsonl": _file_record(test_bytes),
        },
        "bundle_sha256": _bundle_sha256(dev_bytes, test_bytes),
    }


def _publish_bundle(
    destination: Path,
    dev_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> None:
    """Atomically create the public bundle symlink without replacing a winner.

    A same-parent staging directory is fully fsynced before ``symlink`` creates
    the public name.  The final path creation is a single create-only system
    call, so a concurrent publisher cannot be overwritten between an existence
    check and a directory rename.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    published = False
    try:
        (temporary / "mmlu_dev.jsonl").write_bytes(_jsonl_bytes(dev_rows))
        (temporary / "mmlu_test.jsonl").write_bytes(_jsonl_bytes(test_rows))
        manifest_bytes = _canonical_json_bytes(manifest)
        (temporary / "mmlu_manifest.json").write_bytes(manifest_bytes)
        for path in temporary.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.symlink(temporary.name, destination, target_is_directory=True)
        published = True
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def _validate_manifest_structure(raw: object) -> dict[str, object]:
    manifest = _json_object(raw, "MMLU manifest")
    required = {
        "schema",
        "dataset_name",
        "requested_source_revision",
        "requested_source_revision_kind",
        "source_fingerprints",
        "datasets_version",
        "protocol",
        "registry",
        "observed_counts",
        "tokenizer",
        "prompt_template_sha256",
        "files",
        "bundle_sha256",
    }
    if set(manifest) != required:
        raise ValueError(
            "MMLU manifest fields do not match the immutable bundle schema"
        )
    if (
        manifest["schema"] != MMLU_BUNDLE_SCHEMA
        or manifest["dataset_name"] != MMLU_DATASET_NAME
    ):
        raise ValueError("MMLU manifest schema or dataset name is invalid")
    _validate_requested_source_revision(manifest["requested_source_revision"])
    if manifest["requested_source_revision_kind"] != "unverified_remote_selector":
        raise ValueError("MMLU requested_source_revision_kind is invalid")
    if (
        not isinstance(manifest["datasets_version"], str)
        or not manifest["datasets_version"]
    ):
        raise ValueError("MMLU manifest datasets_version is invalid")
    if manifest["protocol"] != MMLU_PROTOCOL or manifest["registry"] != list(
        MMLU_SUBJECTS
    ):
        raise ValueError("MMLU manifest protocol or registry is invalid")
    manifest["observed_counts"] = _validate_count_map(
        manifest["observed_counts"], "observed_counts"
    )
    manifest["source_fingerprints"] = _json_object(
        manifest["source_fingerprints"], "source_fingerprints"
    )
    if set(manifest["source_fingerprints"]) != set(MMLU_SUBJECTS):
        raise ValueError("MMLU manifest source_fingerprints must cover every subject")
    for subject in MMLU_SUBJECTS:
        fingerprints = _json_object(
            manifest["source_fingerprints"][subject], f"source_fingerprints.{subject}"
        )
        if set(fingerprints) != {"dev", "test"} or any(
            (
                value is not None and (not isinstance(value, str))
                for value in fingerprints.values()
            )
        ):
            raise ValueError(f"MMLU manifest source_fingerprints.{subject} is invalid")
        manifest["source_fingerprints"][subject] = fingerprints
    manifest["tokenizer"] = _json_object(manifest["tokenizer"], "tokenizer")
    if not manifest["tokenizer"]:
        raise ValueError("MMLU manifest tokenizer identity is empty")
    if (
        not isinstance(manifest["prompt_template_sha256"], str)
        or manifest["prompt_template_sha256"] != MMLU_PROMPT_TEMPLATE_SHA256
    ):
        raise ValueError("MMLU manifest prompt_template_sha256 is invalid")
    files = _json_object(manifest["files"], "files")
    if set(files) != set(MMLU_BUNDLE_FILENAMES):
        raise ValueError("MMLU manifest files are invalid")
    for filename, record in files.items():
        record = _json_object(record, f"files.{filename}")
        if (
            set(record) != {"bytes", "sha256"}
            or type(record["bytes"]) is not int
            or record["bytes"] < 0
        ):
            raise ValueError(f"MMLU manifest files.{filename} is invalid")
        if not _is_sha256(record["sha256"]):
            raise ValueError(f"MMLU manifest files.{filename}.sha256 is invalid")
        files[filename] = record
    manifest["files"] = files
    if not _is_sha256(manifest["bundle_sha256"]):
        raise ValueError("MMLU manifest bundle_sha256 is invalid")
    return manifest


def _observed_counts(
    dev_rows: Iterable[Mapping[str, object]], observed_test: Mapping[str, int]
) -> dict[str, dict[str, int]]:
    dev_counts = Counter((str(row["subject"]) for row in dev_rows))
    return {
        subject: {"dev": dev_counts[subject], "test": observed_test[subject]}
        for subject in MMLU_SUBJECTS
    }


def _validate_count_map(value: object, label: str) -> dict[str, dict[str, int]]:
    counts = _json_object(value, label)
    if set(counts) != set(MMLU_SUBJECTS):
        raise ValueError(f"MMLU manifest {label} must cover every subject")
    normalized: dict[str, dict[str, int]] = {}
    for subject in MMLU_SUBJECTS:
        pair = _json_object(counts[subject], f"{label}.{subject}")
        if set(pair) != {"dev", "test"} or any(
            (type(pair[key]) is not int or pair[key] < 0 for key in pair)
        ):
            raise ValueError(f"MMLU manifest {label}.{subject} is invalid")
        normalized[subject] = {"dev": pair["dev"], "test": pair["test"]}
    return normalized


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Parse LF-delimited JSONL without treating Unicode line separators as records.

    ``str.splitlines()`` also splits on U+2028/U+2029/U+0085.  Canonical MMLU
    JSON is written with ``ensure_ascii=False``, so those characters are legal
    inside a question string and must stay inside one record.
    """
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    if not text:
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"MMLU {path.name} line {line_number} is invalid JSON"
            ) from exc
        rows.append(_json_object(row, f"MMLU {path.name} line {line_number}"))
    return rows


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join((_canonical_json_bytes(row) + b"\n" for row in rows))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_record(payload: bytes) -> dict[str, object]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _bundle_sha256(dev_bytes: bytes, test_bytes: bytes) -> str:
    digest = hashlib.sha256()
    for filename, payload in zip(MMLU_BUNDLE_FILENAMES, (dev_bytes, test_bytes)):
        digest.update(filename.encode("ascii"))
        digest.update(b"\x00")
        digest.update(payload)
    return digest.hexdigest()


def _bundle_result(
    destination: Path, manifest: Mapping[str, object], *, reused: bool
) -> dict[str, object]:
    return {
        "bundle_path": str(destination),
        "bundle_sha256": manifest["bundle_sha256"],
        "test_count": MMLU_TOTAL_TEST_COUNT,
        "reused": reused,
    }


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    try:
        copied = json.loads(_canonical_json_bytes(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable") from exc
    if not isinstance(copied, dict):
        raise AssertionError("canonical JSON object unexpectedly changed type")
    return copied


def _validate_requested_source_revision(value: object) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or not value.strip()
        or any((ord(character) < 32 or ord(character) == 127 for character in value))
    ):
        raise ValueError(
            "requested_source_revision must be None or a non-empty string without control bytes"
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all((character in "0123456789abcdef" for character in value))
    )
