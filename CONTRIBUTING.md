# Contributing to sonara

Thanks for helping improve sonara. This guide covers setting up a dev environment, running the tests, and what a good pull request looks like.

## Dev setup

You need a stable Rust toolchain and Python 3.9+.

```bash
# Rust core builds with plain cargo
cargo build

# Python bindings are built with maturin into a virtualenv
python -m venv .venv
source .venv/bin/activate
pip install maturin numpy
maturin develop --release -m sonara-python/Cargo.toml
```

After `maturin develop`, `import sonara` works inside the active venv.

## Running tests

```bash
# Rust core tests (includes the accuracy test suite)
cargo test -p sonara

# On macOS you can add the Accelerate BLAS backend, matching CI:
cargo test -p sonara --features accelerate

# Python API tests (need the bindings built into the active venv first)
python tests/python/test_api.py
```

CI runs exactly these on Linux, macOS, and Windows — if they pass locally, they should pass in CI.

## Contributing accuracy improvements

Changes to BPM, key, or chord detection must come with evidence. A plausible-sounding tweak that regresses on real music is worse than no change, so we gate detection changes on measured results.

Include, on a labeled dataset or via the accuracy test suite:

- **Octave-error rate** for BPM (fraction of tracks detected at 2x or 0.5x the true tempo).
- **Median BPM error** (BPM, and/or percent).
- For key detection, accuracy on a labeled set, ideally with the correct / relative / parallel / fifth breakdown.

State the dataset and its size, and show before/after numbers for your change. A change that improves one metric while regressing another needs a clear justification.

## Pull request guidelines

- **Keep PRs small and focused** — one logical change per PR.
- **Tests are required** — add or update tests for behavior you change, and detection changes need accuracy evidence as above.
- **No version bumps in PRs** — leave `pyproject.toml` / `Cargo.toml` versions and the changelog alone; releases are handled separately by the maintainer.

## Forks vs. upstreaming

Forks are welcome — but upstreaming is preferred so the whole community benefits and your fix keeps getting maintained. The maintainer is responsive to issues, so if you have a fix or an idea, open an issue or a PR rather than diverging quietly. We would rather review your change than have it live only on a fork.
