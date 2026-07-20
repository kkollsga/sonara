#!/usr/bin/env python3
"""Synchronize local workflow-skill mirrors from committed canonical sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "workflow" / "skills"
INVENTORY_PATH = SOURCE_ROOT / "inventory.json"
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ContractError(RuntimeError):
    """A canonical source or installed mirror violated the workflow contract."""


def load_inventory() -> tuple[tuple[str, ...], tuple[Path, ...]]:
    data = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if set(data) != {"version", "skills", "mirrors"} or data["version"] != 1:
        raise ContractError("inventory must contain version 1, skills, and mirrors")

    skills = tuple(data["skills"])
    if not skills or len(skills) != len(set(skills)) or tuple(sorted(skills)) != skills:
        raise ContractError("skill inventory must be non-empty, unique, and sorted")
    if any(not isinstance(name, str) or not SLUG_RE.fullmatch(name) for name in skills):
        raise ContractError("skill inventory contains an unsafe name")

    mirrors = tuple(Path(value) for value in data["mirrors"])
    if len(mirrors) != 2 or len(set(mirrors)) != 2:
        raise ContractError("inventory must define exactly two unique mirrors")
    for mirror in mirrors:
        if mirror.is_absolute() or ".." in mirror.parts or mirror.name != "skills":
            raise ContractError(f"unsafe mirror path: {mirror}")

    expected_sources = {SOURCE_ROOT / name / "SKILL.md" for name in skills}
    discovered_sources = set(SOURCE_ROOT.glob("*/SKILL.md"))
    if discovered_sources != expected_sources:
        missing = sorted(str(path.relative_to(ROOT)) for path in expected_sources - discovered_sources)
        extra = sorted(str(path.relative_to(ROOT)) for path in discovered_sources - expected_sources)
        raise ContractError(f"canonical skill source drift: missing={missing}, extra={extra}")
    for path in expected_sources:
        if not path.read_bytes().endswith(b"\n"):
            raise ContractError(f"canonical skill must end with newline: {path.relative_to(ROOT)}")
    return skills, mirrors


def source_bytes(skills: tuple[str, ...]) -> dict[str, bytes]:
    return {name: (SOURCE_ROOT / name / "SKILL.md").read_bytes() for name in skills}


def installed_errors(installed_root: Path) -> list[str]:
    skills, mirrors = load_inventory()
    sources = source_bytes(skills)
    errors: list[str] = []
    for mirror in mirrors:
        for name in skills:
            target = installed_root / mirror / name / "SKILL.md"
            if not target.is_file():
                errors.append(f"missing installed skill: {target}")
            elif target.read_bytes() != sources[name]:
                errors.append(f"installed skill differs from canonical source: {target}")
    return errors


def check_installed(installed_root: Path) -> None:
    errors = installed_errors(installed_root)
    if errors:
        raise ContractError("\n".join(errors))


def write_installed(installed_root: Path) -> None:
    skills, mirrors = load_inventory()
    sources = source_bytes(skills)
    for mirror in mirrors:
        for name in skills:
            target = installed_root / mirror / name / "SKILL.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_bytes() != sources[name]:
                target.write_bytes(sources[name])


def self_test() -> None:
    skills, mirrors = load_inventory()
    with tempfile.TemporaryDirectory(prefix="sonara-skill-sync-") as raw_root:
        installed_root = Path(raw_root)
        sentinel = installed_root / mirrors[0] / "unmanaged-plugin" / "SKILL.md"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text("unmanaged\n", encoding="utf-8")

        write_installed(installed_root)
        check_installed(installed_root)
        if sentinel.read_text(encoding="utf-8") != "unmanaged\n":
            raise ContractError("bootstrap modified an unmanaged skill")

        first = installed_root / mirrors[0] / skills[0] / "SKILL.md"
        first.write_text("corrupt\n", encoding="utf-8")
        if not installed_errors(installed_root):
            raise ContractError("corrupt installed skill was not detected")
        write_installed(installed_root)
        check_installed(installed_root)

        first.unlink()
        if not installed_errors(installed_root):
            raise ContractError("missing installed skill was not detected")
        write_installed(installed_root)
        check_installed(installed_root)
        if sentinel.read_text(encoding="utf-8") != "unmanaged\n":
            raise ContractError("repair modified an unmanaged skill")


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check-source", action="store_true")
    actions.add_argument("--check-installed", action="store_true")
    actions.add_argument("--write", action="store_true")
    actions.add_argument("--self-test", action="store_true")
    parser.add_argument("--installed-root", type=Path, default=ROOT)
    args = parser.parse_args()

    try:
        if args.check_source:
            load_inventory()
            print("workflow skill source contract: PASS")
        elif args.check_installed:
            check_installed(args.installed_root.resolve())
            print("installed workflow skill mirrors: PASS")
        elif args.write:
            write_installed(args.installed_root.resolve())
            check_installed(args.installed_root.resolve())
            print("installed workflow skill mirrors: synchronized")
        else:
            self_test()
            print("workflow skill synchronization self-test: PASS")
    except (ContractError, json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
