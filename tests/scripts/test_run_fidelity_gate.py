#!/usr/bin/env python3
"""Contract tests for exhaustive fidelity ownership."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
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
        "reviewed_transitions": [],
    }


class FidelityMapContractTests(unittest.TestCase):
    def test_reviewed_transition_hash_is_crlf_stable(self) -> None:
        self.assertEqual(
            ROUTER.canonical_text_sha256(b"first\nsecond\n"),
            ROUTER.canonical_text_sha256(b"first\r\nsecond\r\n"),
        )

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

    def test_changed_paths_cover_git_states_and_both_rename_sides(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sonara-fidelity-git-") as raw_root:
            root = Path(raw_root)

            def git(*args: str) -> str:
                return subprocess.run(
                    ["git", *args],
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                ).stdout.strip()

            git("init", "-q")
            git("config", "user.email", "contract@example.invalid")
            git("config", "user.name", "Contract Test")
            (root / "src").mkdir()
            (root / "src" / "a.rs").write_text("base\n", encoding="utf-8")
            (root / ".gitignore").write_text("src/generated.rs\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")

            (root / "src" / "a.rs").write_text("committed\n", encoding="utf-8")
            git("add", "src/a.rs")
            git("commit", "-qm", "change")
            git("mv", "src/a.rs", "src/renamed.rs")
            (root / "src" / "renamed.rs").write_text("dirty\n", encoding="utf-8")
            (root / "src" / "untracked.rs").write_text("new\n", encoding="utf-8")
            (root / "src" / "generated.rs").write_text("ignored\n", encoding="utf-8")

            merge_base, changed = ROUTER.derive_changed_paths(root, base, ["src/*.rs"])
            self.assertEqual(merge_base, base)
            self.assertTrue(
                {"src/a.rs", "src/renamed.rs", "src/untracked.rs"}.issubset(changed)
            )
            self.assertNotIn("src/generated.rs", changed)

    def test_invalid_base_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot resolve fidelity base"):
            ROUTER.derive_changed_paths(ROOT, "definitely-not-a-ref", ["sonara/src/*.rs"])

    def test_reviewed_bootstrap_transitions_match_exact_contents(self) -> None:
        data = ROUTER.load_map()
        for transition in data["reviewed_transitions"]:
            path = transition["path"]
            current = ROUTER.canonical_text_sha256((ROOT / path).read_bytes())
            self.assertEqual(current, transition["head_sha256"])

            base_hash = transition["base_sha256"]
            if base_hash is None:
                continue
            revisions = subprocess.run(
                ["git", "rev-list", "--all", "--", path],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.splitlines()
            self.assertTrue(
                any(
                    ROUTER.sha256_at_revision(ROOT, revision, path) == base_hash
                    for revision in revisions
                ),
                f"{path}: recorded transition base is absent from Git history",
            )


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    sys.exit(not result.result.wasSuccessful())
