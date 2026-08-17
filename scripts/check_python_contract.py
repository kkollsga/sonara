#!/usr/bin/env python3
"""Validate Sonara's runtime, stub, abi3, documentation, and CI contracts."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
FLOOR_RE = re.compile(r"^>=(\d+)\.(\d+)$")
ABI_FEATURE_RE = re.compile(r'"abi3-py(\d)(\d+)"')
CLASSIFIER_RE = re.compile(r"^Programming Language :: Python :: (\d+)\.(\d+)$")
FUSED_ANALYZER_CONTRACT = {
    "analyze_file": (
        "path",
        {
            "sr",
            "mode",
            "features",
            "bpm_min",
            "bpm_max",
            "genre_model",
            "vocalness_model",
        },
        "AnalysisResult",
    ),
    "analyze_signal": (
        "y",
        {
            "sr",
            "mode",
            "features",
            "bpm_min",
            "bpm_max",
            "genre_model",
            "vocalness_model",
        },
        "AnalysisResult",
    ),
    "analyze_batch": (
        "paths",
        {
            "sr",
            "mode",
            "features",
            "bpm_min",
            "bpm_max",
            "progress",
            "genre_model",
            "vocalness_model",
        },
        "List[AnalysisResult]",
    ),
    # --- augment lane --- (positional spec may be a tuple of names)
    "augment_analysis": (
        ("cached", "features"),
        {
            "audio_path",
            "bpm_min",
            "bpm_max",
            "genre_model",
            "vocalness_model",
        },
        "AnalysisResult",
    ),
    "can_augment": (("cached", "feature"), set(), "bool"),
    "augment_blocker": (("cached", "feature"), set(), "Optional[str]"),
    "feature_dependencies": ((), set(), "List[Dict[str, Union[str, bool, List[str]]]]"),
}


def parse_runtime_floor(value: str) -> tuple[int, int]:
    match = FLOOR_RE.fullmatch(value)
    if not match:
        raise ValueError("requires-python must be one exact >=MAJOR.MINOR floor")
    return int(match.group(1)), int(match.group(2))


def project_data(root: Path = ROOT) -> dict:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def runtime_floor(root: Path = ROOT) -> tuple[int, int]:
    return parse_runtime_floor(project_data(root)["requires-python"])


def runtime_floor_text(root: Path = ROOT) -> str:
    major, minor = runtime_floor(root)
    return f"{major}.{minor}"


def abi_tag(root: Path = ROOT) -> str:
    major, minor = runtime_floor(root)
    return f"cp{major}{minor}-abi3"


def check_fused_analyzer_stub(root: Path = ROOT) -> None:
    stub_path = root / "python" / "sonara" / "__init__.pyi"
    tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(node.name, []).append(node)

    duplicates = {
        name: len(nodes)
        for name, nodes in definitions.items()
        if len(nodes) > 1
    }
    if duplicates:
        raise AssertionError(f"duplicate top-level stub functions: {duplicates}")

    for name, contract in FUSED_ANALYZER_CONTRACT.items():
        positional, required_keywords, expected_return = contract
        # A str is a single positional; a tuple lists them in order (possibly
        # empty, e.g. feature_dependencies()).
        expected_positional = [positional] if isinstance(positional, str) else list(positional)
        nodes = definitions.get(name, [])
        if len(nodes) != 1:
            raise AssertionError(f"stub must declare {name} exactly once, got {len(nodes)}")
        node = nodes[0]
        positional_names = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
        if positional_names != expected_positional:
            raise AssertionError(
                f"{name} positional parameters changed: expected {expected_positional}, got {positional_names}"
            )
        keyword_names = {arg.arg for arg in node.args.kwonlyargs}
        missing = required_keywords - keyword_names
        if missing:
            raise AssertionError(f"{name} stub is missing keyword parameters: {sorted(missing)}")
        required_without_defaults = {
            arg.arg
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
            if arg.arg in required_keywords and default is None
        }
        if required_without_defaults:
            raise AssertionError(
                f"{name} keywords unexpectedly became required: {sorted(required_without_defaults)}"
            )
        actual_return = ast.unparse(node.returns) if node.returns is not None else None
        if actual_return != expected_return:
            raise AssertionError(
                f"{name} return changed: expected {expected_return}, got {actual_return}"
            )


def check_contract(root: Path = ROOT) -> None:
    check_fused_analyzer_stub(root)
    project = project_data(root)
    floor = runtime_floor(root)
    floor_text = f"{floor[0]}.{floor[1]}"

    cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
    abi_features = ABI_FEATURE_RE.findall(cargo)
    expected_feature = (str(floor[0]), str(floor[1]))
    if abi_features != [expected_feature]:
        raise AssertionError(
            f"Cargo must contain exactly abi3-py{floor[0]}{floor[1]}, got {abi_features}"
        )

    minors = []
    for classifier in project.get("classifiers", []):
        match = CLASSIFIER_RE.fullmatch(classifier)
        if match:
            minors.append((int(match.group(1)), int(match.group(2))))
    if floor not in minors or any(version < floor for version in minors):
        raise AssertionError(f"Python classifiers do not start at {floor_text}: {minors}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    if f"Requires Python {floor_text}+" not in readme:
        raise AssertionError("README runtime floor does not match pyproject.toml")
    contributing = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    if "Python 3.11+ for repository tooling" not in contributing:
        raise AssertionError("CONTRIBUTING.md must state the Python 3.11+ tooling floor")
    if f"package itself supports Python {floor_text}+" not in contributing:
        raise AssertionError("CONTRIBUTING.md runtime floor does not match pyproject.toml")

    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    required = (
        'label: linux-x86_64\n            os: ubuntu-latest\n            python-version: "3.10"',
        "python-version: ${{ matrix.python-version }}",
    )
    missing = [value for value in required if value not in workflow]
    if missing:
        raise AssertionError(f"minimum-runtime wheel smoke is not aligned: {missing}")


def self_test() -> None:
    if parse_runtime_floor(">=3.10") != (3, 10):
        raise AssertionError("valid floor parsing failed")
    for invalid in ("3.10", ">3.10", ">=3", ">=3.10,<4", ">= 3.10"):
        try:
            parse_runtime_floor(invalid)
        except ValueError:
            continue
        raise AssertionError(f"invalid floor was accepted: {invalid}")

    with tempfile.TemporaryDirectory(prefix="sonara-python-contract-") as raw_root:
        root = Path(raw_root)
        (root / ".github" / "workflows").mkdir(parents=True)

        def write_fixture(
            *,
            requirement: str = ">=3.10",
            abi_feature: str = "abi3-py310",
            classifiers: tuple[str, ...] = ("3.10", "3.11"),
            readme_floor: str = "3.10",
            workflow_floor: str = "3.10",
            stub_text: str | None = None,
        ) -> None:
            (root / "python" / "sonara").mkdir(parents=True, exist_ok=True)
            classifier_text = "\n".join(
                f'  "Programming Language :: Python :: {value}",' for value in classifiers
            )
            (root / "pyproject.toml").write_text(
                f'[project]\nrequires-python = "{requirement}"\nclassifiers = [\n{classifier_text}\n]\n',
                encoding="utf-8",
            )
            (root / "Cargo.toml").write_text(
                f'features = ["{abi_feature}", "extension-module"]\n', encoding="utf-8"
            )
            (root / "README.md").write_text(
                f"Requires Python {readme_floor}+.\n", encoding="utf-8"
            )
            (root / "CONTRIBUTING.md").write_text(
                "Python 3.11+ for repository tooling; the package itself supports "
                f"Python {readme_floor}+.\n",
                encoding="utf-8",
            )
            (root / ".github" / "workflows" / "ci.yml").write_text(
                "          - label: linux-x86_64\n"
                "            os: ubuntu-latest\n"
                f'            python-version: "{workflow_floor}"\n'
                "      - uses: actions/setup-python@v6\n"
                "        with:\n"
                "          python-version: ${{ matrix.python-version }}\n",
                encoding="utf-8",
            )
            (root / "python" / "sonara" / "__init__.pyi").write_text(
                stub_text
                or """
def analyze_file(path: str, *, sr: int = 22050, mode: str = \"compact\", features=None, bpm_min=None, bpm_max=None, genre_model=None, vocalness_model=None) -> AnalysisResult: ...
def analyze_signal(y: AudioArray, *, sr: int = 22050, mode: str = \"compact\", features=None, bpm_min=None, bpm_max=None, genre_model=None, vocalness_model=None) -> AnalysisResult: ...
def analyze_batch(paths: list[str], *, sr: int = 22050, mode: str = \"compact\", features=None, bpm_min=None, bpm_max=None, progress=None, genre_model=None, vocalness_model=None) -> List[AnalysisResult]: ...
def augment_analysis(cached: Dict, features=None, *, audio_path=None, bpm_min=None, bpm_max=None, genre_model=None, vocalness_model=None) -> AnalysisResult: ...
def can_augment(cached: Dict, feature: str) -> bool: ...
def augment_blocker(cached: Dict, feature: str) -> Optional[str]: ...
def feature_dependencies() -> List[Dict[str, Union[str, bool, List[str]]]]: ...
""".lstrip(),
                encoding="utf-8",
            )

        def expect_failure(**kwargs: object) -> None:
            write_fixture(**kwargs)
            try:
                check_contract(root)
            except (AssertionError, ValueError):
                return
            raise AssertionError(f"mismatched fixture was accepted: {kwargs}")

        write_fixture()
        check_contract(root)
        expect_failure(requirement=">=3.10,<4")
        expect_failure(abi_feature="abi3-py39")
        expect_failure(classifiers=("3.9", "3.10"))
        expect_failure(readme_floor="3.9")
        expect_failure(workflow_floor="3.12")
        valid_stub = (root / "python" / "sonara" / "__init__.pyi").read_text(
            encoding="utf-8"
        )
        expect_failure(stub_text=valid_stub + valid_stub.splitlines()[0] + "\n")
        expect_failure(stub_text=valid_stub.replace(", features=None", ""))
        expect_failure(
            stub_text=valid_stub.replace(
                ") -> AnalysisResult: ...", ") -> dict: ...", 1
            )
        )
        # Tuple-positional contracts must also pin their positional lists.
        expect_failure(
            stub_text=valid_stub.replace(
                "def can_augment(cached: Dict, feature: str)",
                "def can_augment(cached: Dict)",
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--runtime-floor", action="store_true")
    actions.add_argument("--abi-tag", action="store_true")
    args = parser.parse_args()
    try:
        if args.check:
            check_contract()
            print("Python runtime/stub/abi3 contract: PASS")
        elif args.self_test:
            self_test()
            print("Python runtime/stub/abi3 self-test: PASS")
        elif args.runtime_floor:
            print(runtime_floor_text())
        else:
            print(abi_tag())
    except (AssertionError, KeyError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
