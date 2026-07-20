#!/usr/bin/env python3
"""Run Sonara's canonical standard Python API suite on every platform."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STANDARD_TESTS = (
    "test_api",
    "test_batch_errors",
    "test_camelot",
    "test_beatgrid",
    "test_structure",
    "test_similarity",
    "test_fingerprint",
    "test_loudness",
    "test_misc_features",
    "test_genre",
    "test_vocalness_model",
)
FIDELITY_TESTS = {
    "test_tonal_batch",
    "test_vocalness_frozen",
    "test_vocalness_real",
}
CONTRACT_FILES = (
    ".github/workflows/ci.yml",
    "CONTRIBUTING.md",
)
RUNNER_PATH = "scripts/run_python_tests.py"


def check_contract() -> None:
    discovered = {
        path.stem for path in (ROOT / "tests" / "python").glob("test_*.py")
    } - FIDELITY_TESTS
    expected = set(STANDARD_TESTS)
    if discovered != expected:
        raise AssertionError(
            "standard Python suite drift: "
            f"missing={sorted(discovered - expected)}, "
            f"stale={sorted(expected - discovered)}"
        )
    for relative in CONTRACT_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        if RUNNER_PATH not in text:
            raise AssertionError(f"{relative} does not use {RUNNER_PATH!r}")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_release_artifacts.py"),
         "--check-workflow"],
        cwd=ROOT,
        check=True,
    )

    # Local agent rules/workflow skills are gitignored and may not exist in a
    # clean clone, but when present they must consume the same runner.
    for relative in (
        "AGENTS.md",
        "CLAUDE.md",
        ".agents/skills/release/SKILL.md",
        ".agents/skills/phased-plan/SKILL.md",
    ):
        path = ROOT / relative
        if path.exists() and RUNNER_PATH not in path.read_text(encoding="utf-8"):
            raise AssertionError(f"{relative} does not use {RUNNER_PATH!r}")


def run_suite() -> None:
    for name in STANDARD_TESTS:
        path = ROOT / "tests" / "python" / f"{name}.py"
        print(f"=== {name} ===", flush=True)
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print the test names")
    parser.add_argument(
        "--check-contract",
        action="store_true",
        help="validate suite discovery and workflow consumers without running tests",
    )
    args = parser.parse_args()
    if args.list:
        print("\n".join(STANDARD_TESTS))
        return 0
    check_contract()
    if not args.check_contract:
        run_suite()
    return 0


if __name__ == "__main__":
    sys.exit(main())
