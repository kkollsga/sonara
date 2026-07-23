#!/usr/bin/env python3
"""Build and attest the exact frozen aggression candidate in a clean directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = ROOT / "tests/reference_data/aggression_v2_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*command: str, cwd: Path = ROOT, env=None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def tree_manifest(root: Path) -> dict[str, str]:
    manifest = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if any("__pycache__" in Path(path).parts or Path(path).suffix == ".pyc" for path in manifest):
        raise ValueError("candidate runtime contains forbidden Python bytecode")
    return manifest


def verify_candidate(freeze: dict) -> None:
    candidate = freeze["candidate"]
    if run("git", "rev-parse", f"{candidate['commit']}^{{tree}}") != candidate["tree"]:
        raise ValueError("candidate tree mismatch")
    if run("git", "rev-parse", f"{candidate['commit']}^") != candidate["base_commit"]:
        raise ValueError("candidate base mismatch")
    diff = subprocess.run(
        ["git", "diff", "--binary", f"{candidate['base_commit']}..{candidate['commit']}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if hashlib.sha256(diff).hexdigest() != candidate["binary_diff_sha256"]:
        raise ValueError("candidate binary diff mismatch")
    for relative, expected in candidate["files"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"candidate file mismatch: {relative}")
    for relative, expected in freeze["evaluation_protocol"].get("immutable_dependencies", {}).items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"immutable build dependency mismatch: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    verify_candidate(freeze)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    wheel_dir = args.output_dir / "wheel"
    site_dir = args.output_dir / "site"
    wheel_dir.mkdir()
    site_dir.mkdir()

    hasher_manifest = ROOT / "tools/aggression-content-hash/Cargo.toml"
    run("cargo", "build", "--release", "--locked", "--manifest-path", str(hasher_manifest))
    executable = "sonara-aggression-content-hash.exe" if os.name == "nt" else "sonara-aggression-content-hash"
    hasher = ROOT / "tools/aggression-content-hash/target/release" / executable

    with tempfile.TemporaryDirectory(prefix="sonara-aggression-candidate-") as raw_source:
        source_dir = Path(raw_source)
        run("git", "worktree", "add", "--detach", str(source_dir), freeze["candidate"]["commit"])
        try:
            source_lock_sha256 = sha256(source_dir / "Cargo.lock")
            run(
                "maturin", "build", "--release", "--locked",
                "-m", str(source_dir / "sonara-python/Cargo.toml"),
                "--out", str(wheel_dir.resolve()), cwd=source_dir,
            )
        finally:
            run("git", "worktree", "remove", "--force", str(source_dir))
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected one wheel, found {len(wheels)}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(site_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    probe = json.loads(run(sys.executable, str(ROOT / "scripts/probe_aggression_runtime.py"), env=env))
    if not Path(probe["native_path"]).is_relative_to(site_dir.resolve()):
        raise ValueError("runtime probe imported a native module outside the sealed site")
    if probe["model_id"] != freeze["model"]["model_id"]:
        raise ValueError("runtime probe model mismatch")

    runtime_config = {
        "build": ["maturin", "build", "--release", "--locked"],
        "content_hasher": ["cargo", "build", "--release", "--locked"],
        "python_no_user_site": True,
        "source_commit": freeze["candidate"]["commit"],
    }
    runtime_config_sha256 = hashlib.sha256(
        json.dumps(runtime_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    attestation = {
        "format": "sonara.aggression-runtime-attestation.v1",
        "freeze_identity_sha256": freeze["freeze_identity_sha256"],
        "candidate_commit": freeze["candidate"]["commit"],
        "candidate_tree": freeze["candidate"]["tree"],
        "wheel_name": wheel.name,
        "wheel_sha256": sha256(wheel),
        "site_manifest": tree_manifest(site_dir),
        "content_hasher_sha256": sha256(hasher),
        "content_hasher_source_sha256": sha256(ROOT / "tools/aggression-content-hash/src/main.rs"),
        "content_hasher_lock_sha256": sha256(ROOT / "tools/aggression-content-hash/Cargo.lock"),
        "candidate_cargo_lock_sha256": source_lock_sha256,
        "probe_script_sha256": sha256(ROOT / "scripts/probe_aggression_runtime.py"),
        "runtime_builder_sha256": sha256(Path(__file__)),
        "runtime_config": runtime_config,
        "runtime_config_sha256": runtime_config_sha256,
        "probe": probe,
        "toolchain": {
            "rustc": run("rustc", "--version"),
            "cargo": run("cargo", "--version"),
            "maturin": run("maturin", "--version"),
        },
    }
    args.attestation.parent.mkdir(parents=True, exist_ok=True)
    with args.attestation.open("x", encoding="utf-8") as handle:
        json.dump(attestation, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"runtime attested: {args.attestation} sha256={sha256(args.attestation)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"candidate build rejected: {error}", file=sys.stderr)
        raise SystemExit(1)
