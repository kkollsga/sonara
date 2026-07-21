#!/usr/bin/env python3
"""Validate Sonara's runtime, abi3, documentation, and CI Python floor."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
FLOOR_RE = re.compile(r"^>=(\d+)\.(\d+)$")
ABI_FEATURE_RE = re.compile(r'"abi3-py(\d)(\d+)"')
CLASSIFIER_RE = re.compile(r"^Programming Language :: Python :: (\d+)\.(\d+)$")


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


def check_contract(root: Path = ROOT) -> None:
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
        ) -> None:
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
            print("Python runtime/abi3 contract: PASS")
        elif args.self_test:
            self_test()
            print("Python runtime/abi3 self-test: PASS")
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
