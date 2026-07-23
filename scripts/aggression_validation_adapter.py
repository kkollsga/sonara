#!/usr/bin/env python3
"""Build generic validation artifacts for the frozen aggression audit.

The public capsule contains only logical resource identities and content
digests. Filesystem paths remain in the local-only binding manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CAPSULE_FORMAT = "sonara.validation-capsule.v1"
BINDINGS_FORMAT = "sonara.validation-bindings.v1"
MANIFEST_FORMAT = "sonara.aggression-validation-manifest.v1"
SECTIONS = {"runtime", "evaluator", "control", "tooling", "artifact"}


def canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def digest_bytes(content: bytes) -> dict[str, str]:
    return {"algorithm": "sha256", "value": hashlib.sha256(content).hexdigest()}


def command_digest(command: dict) -> dict[str, str]:
    content = b"sonara-validation-command-v1\0" + canonical_json(command).encode()
    return digest_bytes(content)


def artifact(identity: str, role: str, path: Path) -> dict:
    content = path.read_bytes()
    return {
        "digest": digest_bytes(content),
        "id": identity,
        "role": role,
        "size_bytes": len(content),
    }


def decimal_text(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("threshold values must be decimal numbers or strings")
    text = str(value)
    if "e" in text.lower() or text.startswith("+"):
        raise ValueError("threshold values must use plain decimal notation")
    return text


def build_artifacts(freeze_path: Path, manifest_path: Path) -> tuple[dict, dict, dict]:
    freeze = read_object(freeze_path)
    manifest = read_object(manifest_path)
    if freeze.get("format") != "sonara.aggression-freeze.v1":
        raise ValueError("invalid aggression freeze")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("invalid aggression validation manifest")

    rows = list(manifest.get("resources", []))
    rows.append({
        "id": "aggression-freeze.json",
        "path": str(freeze_path),
        "role": "suite-freeze",
        "section": "artifact",
    })
    ids: set[str] = set()
    grouped: dict[str, list[dict]] = {section: [] for section in SECTIONS}
    bindings = []
    for row in rows:
        if not isinstance(row, dict) or row.get("section") not in SECTIONS:
            raise ValueError("resource section is invalid")
        identity = row.get("id")
        role = row.get("role")
        path = Path(row.get("path", ""))
        if not isinstance(identity, str) or not isinstance(role, str) or identity in ids:
            raise ValueError("resource identities must be unique strings")
        if not path.is_file():
            raise ValueError(f"missing validation resource: {identity}")
        ids.add(identity)
        grouped[row["section"]].append(artifact(identity, role, path))
        bindings.append({"id": identity, "path": str(path.resolve())})

    evaluator = manifest.get("evaluator", {})
    kind = evaluator.get("kind")
    if kind not in {"executable", "signed_human_panel"}:
        raise ValueError("evaluator kind is invalid")
    output_path = Path(evaluator.get("output_path", ""))
    attestation_path = Path(evaluator.get("attestation_path", ""))
    if not output_path.is_file() or not attestation_path.is_file():
        raise ValueError("evaluator output and attestation must exist")

    candidate = freeze["candidate"]
    model = freeze["model"]
    thresholds = freeze["evaluation_protocol"]["required_thresholds"]
    command = manifest.get("command")
    if not isinstance(command, dict) or command.get("executable_id") not in ids:
        raise ValueError("command must name a bound executable resource")
    arguments = command.get("arguments")
    if not isinstance(arguments, list):
        raise ValueError("command arguments must be a list")
    for argument in arguments:
        if not isinstance(argument, dict) or argument.get("kind") not in {"literal", "resource"}:
            raise ValueError("command argument is invalid")
        if argument["kind"] == "resource" and argument.get("value") not in ids:
            raise ValueError("command argument names an unknown resource")

    capsule = {
        "artifacts": grouped["artifact"],
        "candidate": {
            "binary_diff": {"algorithm": "sha256", "value": candidate["binary_diff_sha256"]},
            "commit": candidate["commit"],
            "tree": candidate["tree"],
        },
        "command_digest": command_digest(command),
        "feature": "aggression-ranking",
        "format": CAPSULE_FORMAT,
        "model_id": model["model_id"],
        "runtime": {
            "artifacts": grouped["runtime"],
            "model_id": model["model_id"],
            "schema_version": model["analysis_schema_version"],
        },
        "suite": {
            "controls": grouped["control"],
            "evaluator": {
                "artifacts": grouped["evaluator"],
                "attestation_digest": digest_bytes(attestation_path.read_bytes()),
                "id": evaluator.get("id", "aggression-independent-evaluator"),
                "kind": kind,
                "output_digest": digest_bytes(output_path.read_bytes()),
            },
            "kind": "aggression-locked-ranking",
            "schema_version": 1,
            "thresholds": [
                {"name": name.replace("_", "-"), "value": decimal_text(value)}
                for name, value in sorted(thresholds.items())
            ],
            "tooling": grouped["tooling"],
        },
    }
    return capsule, {"format": BINDINGS_FORMAT, "resources": bindings}, command


def write_exclusive(path: Path, value: dict, *, canonical: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json(value) if canonical else json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capsule-output", type=Path, required=True)
    parser.add_argument("--bindings-output", type=Path, required=True)
    parser.add_argument("--command-output", type=Path, required=True)
    args = parser.parse_args()
    capsule, bindings, command = build_artifacts(args.freeze, args.manifest)
    write_exclusive(args.capsule_output, capsule, canonical=True)
    write_exclusive(args.bindings_output, bindings, canonical=False)
    write_exclusive(args.command_output, command, canonical=False)
    print(f"aggression validation capsule prepared: {args.capsule_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"aggression validation capsule rejected: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
