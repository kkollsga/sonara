## Summary

What does this PR change, and why?

## Checklist

- [ ] `cargo test -p sonara` passes
- [ ] Python tests pass (`python tests/python/test_api.py`) if the change affects the bindings or pipeline
- [ ] Detection changes (BPM / key / chords) include accuracy evidence — dataset + size, octave-error rate, and median BPM error (see CONTRIBUTING.md)
- [ ] No version bump (no changes to `pyproject.toml` / `Cargo.toml` versions or the changelog)
- [ ] PR is small and focused on one logical change
