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
        "validation",
        "--test",
        "validation_contracts",
    ],
    cwd=ROOT,
    check=True,
)
print("validation protocol contracts: PASS")
