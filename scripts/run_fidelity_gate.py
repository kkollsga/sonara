#!/usr/bin/env python3
"""Route accuracy-bearing changes through exhaustive, fail-closed ownership."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "tests" / "fidelity_gates.json"
GLOB_META = frozenset("*?[")


def normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"unsafe repository path: {path!r}")
    return normalized


def matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def repository_files(root: Path) -> set[str]:
    commands = (
        ["git", "ls-files", "-z"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    )
    files: set[str] = set()
    for command in commands:
        output = subprocess.run(
            command, cwd=root, check=True, stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
        files.update(normalize_path(value) for value in output.split("\0") if value)
    return files


def protected_files(data: dict, root: Path) -> set[str]:
    patterns = data.get("protected_globs", [])
    return {path for path in repository_files(root) if matches(path, patterns)}


def validate_map(
    data: dict,
    files: set[str],
    root: Path = ROOT,
    *,
    check_commands: bool = True,
) -> None:
    if set(data) != {"version", "protected_globs", "domains", "ownership"}:
        raise ValueError("fidelity map keys do not match the v2 contract")
    if data["version"] != 2:
        raise ValueError("fidelity map version must be 2")

    protected = data["protected_globs"]
    if not isinstance(protected, list) or not protected or len(set(protected)) != len(protected):
        raise ValueError("protected_globs must be a non-empty unique list")
    for pattern in protected:
        if not isinstance(pattern, str) or normalize_path(pattern) != pattern:
            raise ValueError(f"invalid protected glob: {pattern!r}")
        if not any(fnmatch.fnmatchcase(path, pattern) for path in files):
            raise ValueError(f"protected glob matches no repository file: {pattern}")

    domains = data["domains"]
    if not isinstance(domains, dict) or not domains:
        raise ValueError("domains must be a non-empty object")
    for name, route in domains.items():
        if not isinstance(name, str) or not isinstance(route, dict):
            raise ValueError("invalid fidelity domain")
        if ("command" in route) == ("blocked" in route):
            raise ValueError(f"{name}: define exactly one of command or blocked")
        if "command" in route:
            command = route["command"]
            if not isinstance(command, list) or len(command) < 2 or not all(
                isinstance(value, str) and value for value in command
            ):
                raise ValueError(f"{name}: command must be a non-empty argv list")
            if check_commands:
                target = root / command[1]
                if not target.is_file():
                    raise ValueError(f"{name}: gate target is missing: {target}")
        elif not isinstance(route["blocked"], str) or not route["blocked"].strip():
            raise ValueError(f"{name}: blocked reason must be non-empty")

    ownership = data["ownership"]
    if not isinstance(ownership, list) or not ownership:
        raise ValueError("ownership must be a non-empty list")
    for index, owner in enumerate(ownership):
        if not isinstance(owner, dict) or "paths" not in owner:
            raise ValueError(f"ownership[{index}] is malformed")
        paths = owner["paths"]
        if not isinstance(paths, list) or not paths or len(set(paths)) != len(paths):
            raise ValueError(f"ownership[{index}].paths must be non-empty and unique")
        has_domains = "domains" in owner
        has_exemption = "exemption" in owner
        if has_domains == has_exemption:
            raise ValueError(f"ownership[{index}] needs domains or exemption")
        expected_keys = {"paths", "domains" if has_domains else "exemption"}
        if set(owner) != expected_keys:
            raise ValueError(f"ownership[{index}] has unexpected keys")
        if has_domains:
            names = owner["domains"]
            if not isinstance(names, list) or not names or len(set(names)) != len(names):
                raise ValueError(f"ownership[{index}].domains must be non-empty and unique")
            unknown = set(names) - set(domains)
            if unknown:
                raise ValueError(f"ownership[{index}] has undefined domains: {sorted(unknown)}")
        else:
            reason = owner["exemption"]
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"ownership[{index}] exemption reason must be non-empty")
            if any(any(char in path for char in GLOB_META) for path in paths):
                raise ValueError("reviewed exemptions must use exact paths")
        for pattern in paths:
            if not isinstance(pattern, str) or normalize_path(pattern) != pattern:
                raise ValueError(f"invalid ownership path: {pattern!r}")
            if not any(fnmatch.fnmatchcase(path, pattern) for path in files):
                raise ValueError(f"ownership path matches no protected file: {pattern}")

    for path in sorted(files):
        owners = [owner for owner in ownership if matches(path, owner["paths"])]
        if len(owners) != 1:
            raise ValueError(f"{path}: expected exactly one owner, found {len(owners)}")


def load_map(path: Path = MAP_PATH, root: Path = ROOT) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    files = protected_files(data, root)
    validate_map(data, files, root)
    return data


def path_owner(data: dict, path: str) -> dict | None:
    normalized = normalize_path(path)
    if not matches(normalized, data["protected_globs"]):
        return None
    owners = [owner for owner in data["ownership"] if matches(normalized, owner["paths"])]
    if len(owners) != 1:
        raise ValueError(f"{normalized}: expected exactly one owner, found {len(owners)}")
    return owners[0]


def domains_for_paths(data: dict, changed: list[str]) -> set[str]:
    resolved: set[str] = set()
    for path in changed:
        owner = path_owner(data, path)
        if owner is not None and "domains" in owner:
            resolved.update(owner["domains"])
    return resolved


def check_contract(data: dict) -> None:
    expected = {
        "sonara/models/vocalness_v2.json": {"vocalness"},
        "sonara/src/beat.rs": {"bpm"},
        "sonara/src/fingerprint.rs": {"fingerprint"},
        "sonara/src/genre.rs": {"genre"},
        "sonara/src/analyze.rs": {"cross_cutting"},
        "sonara/src/perceptual.rs": {"cross_cutting", "mood_aggression"},
    }
    for path, domains in expected.items():
        actual = domains_for_paths(data, [path])
        if actual != domains:
            raise AssertionError(f"{path}: expected {sorted(domains)}, got {sorted(actual)}")
    if domains_for_paths(data, ["README.md"]):
        raise AssertionError("non-accuracy documentation must not trigger a fidelity gate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", action="append", choices=None)
    parser.add_argument("--changed", nargs="+")
    parser.add_argument("--check-contract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = load_map()
    check_contract(data)
    if args.check_contract:
        print("fidelity routing contract: PASS")
        return 0

    domains = set(args.domain or [])
    unknown = domains - set(data["domains"])
    if unknown:
        raise ValueError(f"unknown fidelity domains: {sorted(unknown)}")
    domains.update(domains_for_paths(data, args.changed or []))
    if not domains:
        print("no changed accuracy domain requires a fidelity gate")
        return 0

    for name in sorted(domains):
        route = data["domains"][name]
        if "blocked" in route:
            print(f"BLOCKED {name}: {route['blocked']}", file=sys.stderr)
            return 2
        command = [sys.executable if value == "{python}" else value for value in route["command"]]
        print(f"=== fidelity:{name} ===", flush=True)
        if args.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
