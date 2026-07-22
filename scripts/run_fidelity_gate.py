#!/usr/bin/env python3
"""Route changed accuracy domains to mandatory, fail-closed fidelity gates."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "tests" / "fidelity_gates.json"
EXPECTED_DOMAINS = {"vocalness", "aggression", "tonal", "bpm", "beatgrid", "similarity"}


def load_map() -> dict:
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    if data.get("version") != 1 or set(data.get("domains", {})) != EXPECTED_DOMAINS:
        raise ValueError("fidelity map version/domains do not match the contract")
    for name, route in data["domains"].items():
        if not route.get("paths"):
            raise ValueError(f"{name}: paths must be non-empty")
        if ("command" in route) == ("blocked" in route):
            raise ValueError(f"{name}: define exactly one of command or blocked")
        if "command" in route:
            command = route["command"]
            if not isinstance(command, list) or len(command) < 2:
                raise ValueError(f"{name}: command must be an argv list")
            target = ROOT / command[1]
            if not target.is_file():
                raise ValueError(f"{name}: gate target is missing: {target}")
    return data


def domains_for_paths(data: dict, changed: list[str]) -> set[str]:
    resolved = set()
    for path in changed:
        normalized = path.replace("\\", "/").removeprefix("./")
        for name, route in data["domains"].items():
            if any(fnmatch.fnmatchcase(normalized, pattern) for pattern in route["paths"]):
                resolved.add(name)
    return resolved


def check_contract(data: dict) -> None:
    runnable = {
        name for name, route in data["domains"].items() if "command" in route
    }
    if runnable != {"vocalness", "aggression"}:
        raise AssertionError(
            "only the content-addressed vocalness and aggression domains have valid gates"
        )
    if domains_for_paths(data, ["sonara/models/vocalness_v2.json"]) != {"vocalness"}:
        raise AssertionError("bundled vocalness artifacts must route to the frozen gate")
    if domains_for_paths(data, ["sonara/src/beat.rs"]) != {"bpm"}:
        raise AssertionError("BPM changes must route to an explicit blocked domain")
    if domains_for_paths(data, ["sonara/src/aggression.rs"]) != {"aggression"}:
        raise AssertionError("aggression model changes must route to the frozen gate")
    if domains_for_paths(data, ["README.md"]):
        raise AssertionError("non-accuracy documentation must not trigger a fidelity gate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", action="append", choices=sorted(EXPECTED_DOMAINS))
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
    domains.update(domains_for_paths(data, args.changed or []))
    if not domains:
        print("no changed accuracy domain requires a fidelity gate")
        return 0

    for name in sorted(domains):
        route = data["domains"][name]
        if "blocked" in route:
            print(f"BLOCKED {name}: {route['blocked']}", file=sys.stderr)
            return 2
        command = [sys.executable if value == "{python}" else value
                   for value in route["command"]]
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
