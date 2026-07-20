#!/usr/bin/env python3
"""Build and smoke-install Sonara's wheel and sdist in clean environments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def smoke(artifact: Path, work: Path) -> None:
    kind = "sdist" if artifact.name.endswith(".tar.gz") else "wheel"
    venv = work / f"venv-{kind}"
    run([sys.executable, "-m", "venv", str(venv)], cwd=work)
    python = venv_python(venv)
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(artifact),
        ],
        cwd=work,
    )

    check = r"""
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import sonara

metadata = importlib.metadata.metadata("sonara")
requirements = metadata.get_all("Requires-Dist") or []
if not any(requirement.lower().startswith("numpy") for requirement in requirements):
    raise AssertionError(f"installed metadata has no NumPy requirement: {requirements}")
package_path = Path(sonara.__file__).resolve()
if "site-packages" not in package_path.parts:
    raise AssertionError(f"Sonara was not imported from the clean environment: {package_path}")
signal = np.zeros(4096, dtype=np.float32)
result = sonara.analyze_signal(signal, sr=22050)
if not np.isfinite(result["duration_sec"]):
    raise AssertionError(result)
print(json.dumps({
    "python": sys.version.split()[0],
    "sonara": sonara.__version__,
    "numpy": np.__version__,
    "package_path": str(package_path),
    "requirements": requirements,
}, sort_keys=True))
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    run([str(python), "-I", "-c", check], cwd=work, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--kind", choices=("wheel", "sdist"))
    args = parser.parse_args()
    if (args.artifact_dir is None) != (args.kind is None):
        parser.error("--artifact-dir and --kind must be supplied together")

    with tempfile.TemporaryDirectory(prefix="sonara-artifact-smoke-") as temp:
        work = Path(temp)
        if args.artifact_dir is not None:
            pattern = "*.whl" if args.kind == "wheel" else "*.tar.gz"
            artifacts = sorted(args.artifact_dir.resolve().glob(pattern))
            if len(artifacts) != 1:
                raise AssertionError(
                    f"expected one {args.kind} in {args.artifact_dir}, got {artifacts}"
                )
            smoke(artifacts[0], work)
            print(f"{args.kind} smoke install passed")
            return

        dist = work / "dist"
        dist.mkdir()
        build_env = os.environ.copy()
        build_env["RUSTC_WRAPPER"] = ""
        run(
            [
                "maturin",
                "build",
                "--release",
                "--interpreter",
                sys.executable,
                "--out",
                str(dist),
                "-m",
                "sonara-python/Cargo.toml",
            ],
            env=build_env,
        )
        run(
            [
                "maturin",
                "sdist",
                "--out",
                str(dist),
                "-m",
                "sonara-python/Cargo.toml",
            ],
            env=build_env,
        )
        wheels = sorted(dist.glob("*.whl"))
        sdists = sorted(dist.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise AssertionError(f"expected one wheel and one sdist, got {list(dist.iterdir())}")
        smoke(wheels[0], work)
        smoke(sdists[0], work)
        print("wheel and sdist smoke installs passed")


if __name__ == "__main__":
    main()
