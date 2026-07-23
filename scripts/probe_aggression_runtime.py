#!/usr/bin/env python3
"""Emit a content attestation for the exact Python/native aggression runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import sonara


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    # Force model decode and the full wrapper/native dependency path to load.
    sample_rate = 22_050
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    signal = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    result = sonara.analyze_aggression_signal(signal, sr=sample_rate)
    if result["aggression_model_id"] != sonara.AGGRESSION_MODEL_ID:
        raise RuntimeError("runtime model identity mismatch")

    modules = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == "sonara" or name.startswith("sonara.") or name == "numpy" or name.startswith("numpy.")):
            continue
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        if path.is_file():
            modules[name] = {"path": str(path), "sha256": sha256(path)}
    native = Path(sonara._sonara.__file__).resolve()
    payload = {
        "format": "sonara.aggression-runtime-probe.v1",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_sha256": sha256(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "model_id": sonara.AGGRESSION_MODEL_ID,
        "tie_band": sonara.AGGRESSION_TIE_BAND,
        "native_path": str(native),
        "native_sha256": sha256(native),
        "modules": modules,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
