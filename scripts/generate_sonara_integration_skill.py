#!/usr/bin/env python3
"""Generate the distributable Sonara integration skill and package mirrors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "sonara" / "contracts" / "feature-catalog.v1.json"
CONTRACTS = ROOT / "sonara" / "contracts" / "validation" / "v1"
SOURCE = ROOT / "agent-skills" / "sonara-integration"
PACKAGED = ROOT / "python" / "sonara" / "skills" / "sonara-integration"
PACKAGED_CONTRACTS = ROOT / "python" / "sonara" / "contracts"
ANALYZE = ROOT / "sonara" / "src" / "analyze.rs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_catalog() -> dict:
    value = json.loads(CATALOG.read_text(encoding="utf-8"))
    if set(value) != {"format", "modes", "feature_groups", "interpretation"}:
        raise ValueError("feature catalog shape changed")
    if value["format"] != "sonara.feature-catalog.v1":
        raise ValueError("unsupported feature catalog")
    catalog_names = [name for group in value["feature_groups"] for name in group["features"]]
    if len(catalog_names) != len(set(catalog_names)):
        raise ValueError("duplicate catalog feature")
    source_names = re.findall(r'feature\("([a-z_]+)"\s*,', ANALYZE.read_text(encoding="utf-8"))
    if set(catalog_names) != set(source_names):
        raise ValueError(f"feature catalog drift: catalog={catalog_names}, Rust={source_names}")
    return value


def skill_bytes() -> bytes:
    catalog = load_catalog()
    schema_rows = "\n".join(
        f"- `{path.name}` — SHA-256 `{digest(path)}`"
        for path in sorted(CONTRACTS.glob("*.json"))
    )
    groups = "\n".join(
        f"- **{group['name']}**: {group['meaning']} "
        f"Request with `features=[{', '.join(repr(name) for name in group['features'])}]` as needed."
        for group in catalog["feature_groups"]
    )
    modes = "\n".join(
        f"- `{name}`: {meaning}" for name, meaning in catalog["modes"].items()
    )
    policy = catalog["interpretation"]
    body = f"""---
name: sonara-integration
description: Analyze music with Sonara's Python API, choose efficient feature modes, interpret confidence and abstention, process libraries safely in batches, and verify content-addressed validation receipts. Use when an agent needs to integrate Sonara audio analysis or assess a Sonara validation result.
---

# Sonara integration

## Analyze audio

1. Use `sonara.analyze_file(path)` for one file, `analyze_signal(y, sr=...)` for decoded mono samples, or `analyze_batch(paths)` for a library.
2. Start with `mode="compact"`; select `playlist` only when organization features are needed. Supplying `features=[...]` replaces mode selection rather than adding to it, so list exactly the outputs required; internal dependencies are resolved automatically.
3. Preserve the returned `provenance`, model IDs, schema versions, and error records with persisted results.

{modes}

{groups}

## Interpret results

- {policy['confidence']}
- {policy['uncertainty']}
- {policy['abstention']}
- {policy['batch']}

## Verify validation evidence

1. Obtain the receipt, custody proof, and trust root through independent channels.
2. Call `sonara.validation.verify(receipt, proof, trust_root)` or `sonara validate verify --receipt ... --proof ... --trust-root ...`.
3. Accept the outcome only if verification succeeds and the receipt's evaluation digest names the intended candidate and suite.
4. Never tune from sealed inputs, infer private membership from aggregate receipts, or trust a root merely because it is embedded in the proof.

{policy['validation']}

Packaged validation contract identities:

{schema_rows}
"""
    return body.encode("utf-8")


def expected_files() -> dict[Path, bytes]:
    expected = {
        SOURCE / "SKILL.md": skill_bytes(),
        PACKAGED / "SKILL.md": skill_bytes(),
        PACKAGED / "agents" / "openai.yaml": (SOURCE / "agents" / "openai.yaml").read_bytes(),
        PACKAGED_CONTRACTS / "feature-catalog.v1.json": CATALOG.read_bytes(),
    }
    for contract in sorted(CONTRACTS.glob("*.json")):
        expected[PACKAGED_CONTRACTS / "validation" / "v1" / contract.name] = contract.read_bytes()
    return expected


def generate() -> None:
    for path, content in expected_files().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def stale_files(expected: dict[Path, bytes]) -> list[Path]:
    return [path for path, content in expected.items() if not path.exists() or path.read_bytes() != content]


def check() -> None:
    stale = [str(path.relative_to(ROOT)) for path in stale_files(expected_files())]
    if stale:
        raise AssertionError(f"generated Sonara integration artifacts are stale: {stale}")
    source_files = {path.relative_to(SOURCE) for path in SOURCE.rglob("*") if path.is_file()}
    packaged_files = {path.relative_to(PACKAGED) for path in PACKAGED.rglob("*") if path.is_file()}
    if source_files != packaged_files:
        raise AssertionError(f"skill mirror file drift: source={source_files}, packaged={packaged_files}")
    for relative in source_files:
        if (SOURCE / relative).read_bytes() != (PACKAGED / relative).read_bytes():
            raise AssertionError(f"skill mirror byte drift: {relative}")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="sonara-skill-generator-") as temp:
        artifact = Path(temp) / "artifact"
        expected = {artifact: b"expected"}
        if stale_files(expected) != [artifact]:
            raise AssertionError("generator self-test missed an absent artifact")
        artifact.write_bytes(b"corrupt")
        if stale_files(expected) != [artifact]:
            raise AssertionError("generator self-test missed byte drift")
        artifact.write_bytes(b"expected")
        if stale_files(expected):
            raise AssertionError("generator self-test rejected exact bytes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    elif args.self_test:
        self_test()
    else:
        generate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
