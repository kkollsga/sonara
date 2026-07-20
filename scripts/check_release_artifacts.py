#!/usr/bin/env python3
"""Assert the exact verified artifact set and main-only publish contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
WHEEL_RE = re.compile(
    r"^sonara-(?P<version>\d+\.\d+\.\d+)-cp39-abi3-(?P<platform>.+)\.whl$"
)


def expected_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def check_artifacts(dist: Path) -> None:
    files = sorted(path for path in dist.iterdir() if path.is_file())
    wheels = [path for path in files if path.suffix == ".whl"]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    if len(files) != 5 or len(wheels) != 4 or len(sdists) != 1:
        raise AssertionError(
            f"expected exactly four wheels and one sdist; got {[p.name for p in files]}"
        )

    version = expected_version()
    expected_sdist = f"sonara-{version}.tar.gz"
    if sdists[0].name != expected_sdist:
        raise AssertionError(f"expected {expected_sdist}, got {sdists[0].name}")

    platforms = []
    for wheel in wheels:
        match = WHEEL_RE.fullmatch(wheel.name)
        if not match or match.group("version") != version:
            raise AssertionError(f"invalid abi3 wheel name/version: {wheel.name}")
        platforms.append(match.group("platform"))

    predicates = {
        "linux-x86_64": lambda tag: "manylinux" in tag and tag.endswith("x86_64"),
        "macos-x86_64": lambda tag: tag.startswith("macosx_") and tag.endswith("x86_64"),
        "macos-arm64": lambda tag: tag.startswith("macosx_") and tag.endswith("arm64"),
        "windows-x64": lambda tag: tag == "win_amd64",
    }
    for label, predicate in predicates.items():
        matches = [tag for tag in platforms if predicate(tag)]
        if len(matches) != 1:
            raise AssertionError(f"{label}: expected one wheel, got {matches}")
    print(f"release artifacts: PASS ({', '.join(path.name for path in files)})")


def check_workflow() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    required = (
        "build_wheels:",
        "build_sdist:",
        "collect_artifacts:",
        "needs: [test, collect_artifacts]",
        "if: github.event_name == 'push' && github.ref == 'refs/heads/main'",
        "pattern: release-*",
        "merge-multiple: true",
        "name: release-artifacts",
        "python scripts/check_release_artifacts.py dist",
        "packages-dir: dist/",
        "gh release create \"$VERSION\" dist/* --generate-notes",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise AssertionError(f"release workflow contract missing: {missing}")

    publish = text.split("\n  publish:\n", 1)
    if len(publish) != 2:
        raise AssertionError("publish job is missing")
    publish_body = publish[1]
    if "maturin-action" in publish_body or "command: build" in publish_body:
        raise AssertionError("publish must consume verified artifacts, not rebuild them")
    print("release workflow contract: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", nargs="?", type=Path)
    parser.add_argument("--check-workflow", action="store_true")
    args = parser.parse_args()
    if args.check_workflow:
        check_workflow()
    if args.dist is not None:
        check_artifacts(args.dist)
    if not args.check_workflow and args.dist is None:
        parser.error("provide an artifact directory and/or --check-workflow")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
