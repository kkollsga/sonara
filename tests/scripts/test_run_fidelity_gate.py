#!/usr/bin/env python3
"""Contract tests for exhaustive fidelity ownership."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_fidelity_gate", ROOT / "scripts" / "run_fidelity_gate.py"
)
assert SPEC is not None and SPEC.loader is not None
ROUTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTER)


def minimal_map() -> dict:
    return {
        "version": 2,
        "protected_globs": ["src/*.rs"],
        "domains": {"demo": {"blocked": "fixture"}},
        "ownership": [{"paths": ["src/a.rs"], "domains": ["demo"]}],
    }


class FidelityMapContractTests(unittest.TestCase):
    def test_live_map_is_exhaustive(self) -> None:
        data = ROUTER.load_map()
        ROUTER.check_contract(data)

    def test_uncovered_file_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            ROUTER.validate_map(minimal_map(), {"src/a.rs", "src/b.rs"}, check_commands=False)

    def test_duplicate_ownership_fails(self) -> None:
        data = minimal_map()
        data["ownership"].append({"paths": ["src/*.rs"], "domains": ["demo"]})
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            ROUTER.validate_map(data, {"src/a.rs"}, check_commands=False)

    def test_stale_ownership_path_fails(self) -> None:
        data = minimal_map()
        data["ownership"][0]["paths"].append("src/missing.rs")
        with self.assertRaisesRegex(ValueError, "matches no protected file"):
            ROUTER.validate_map(data, {"src/a.rs"}, check_commands=False)

    def test_exemptions_must_be_exact(self) -> None:
        data = minimal_map()
        data["ownership"] = [{"paths": ["src/*.rs"], "exemption": "fixture"}]
        with self.assertRaisesRegex(ValueError, "exact paths"):
            ROUTER.validate_map(data, {"src/a.rs"}, check_commands=False)

    def test_undefined_domain_fails(self) -> None:
        data = minimal_map()
        data["ownership"][0]["domains"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "undefined domains"):
            ROUTER.validate_map(data, {"src/a.rs"}, check_commands=False)

    def test_reported_probes_are_owned(self) -> None:
        data = ROUTER.load_map()
        expected = {
            "sonara/src/fingerprint.rs": {"fingerprint"},
            "sonara/src/genre.rs": {"genre"},
            "sonara/src/analyze.rs": {"cross_cutting"},
            "sonara/src/perceptual.rs": {"cross_cutting", "mood_aggression"},
        }
        for path, domains in expected.items():
            self.assertEqual(ROUTER.domains_for_paths(data, [path]), domains)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    sys.exit(not result.result.wasSuccessful())
