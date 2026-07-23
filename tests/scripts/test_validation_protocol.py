#!/usr/bin/env python3
"""Run the opt-in Rust validation contract gate."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


subprocess.run(
    [
        "cargo",
        "test",
        "-p",
        "sonara",
        "--features",
        "validation-cli",
        "--test",
        "validation_contracts",
        "--test",
        "validation_custody",
        "--test",
        "validation_runner",
        "--test",
        "validation_cli",
    ],
    cwd=ROOT,
    check=True,
)
print("validation protocol contracts: PASS")
