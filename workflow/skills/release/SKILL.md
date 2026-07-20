---
name: release
description: Cut a sonara release — goal-check against the phased-plan, run the full local gate (cargo test + maturin build + Python API scripts, plus mandatory routed fidelity gates), bump the version in ALL THREE fields (sonara/Cargo.toml, sonara-python/Cargo.toml, pyproject.toml — kept in lockstep) and add the CHANGELOG.md entry, commit, and (with explicit approval) ff-merge to main + push, which triggers the auto-version-gated verified wheel matrix + sdist publish and GitHub release; then verify PyPI and tidy dev-docs.
---

# Release

sonara publishes to **PyPI only** (no crates.io), as four native abi3 wheels
(Linux x86_64, macOS x86_64/arm64, Windows x64) plus an **sdist**, from **three
version fields that must stay identical**:

- `sonara/Cargo.toml` line 3 (`version = "x.y.z"`) — the pure-Rust core crate.
- `sonara-python/Cargo.toml` line 3 (`version = "x.y.z"`) — the PyO3 binding crate.
- `pyproject.toml` line 7 (`version = "x.y.z"`) — **the field the publish job
  reads and version-gates on.**

Bump **all three in the same commit** — a mismatch means the wheel/sdist and the
crates disagree. Publishing is **automatic and version-gated**: pushing to
`main` runs `ci.yml`, whose `publish` job (only on `push` to `main`) reads
`pyproject.toml`'s version, checks PyPI's JSON API (HTTP 200 → already published
→ skip; anything else → publish), consumes the exact five artifacts already
built and smoke-installed by the CI artifact matrix, uploads via **PyPI Trusted
Publishing** (OIDC — no secret, `environment: pypi`),
then **tags `vx.y.z` and creates a GitHub release** with generated notes. You
never run `maturin publish` by hand — the three-file bump + the `main` push are
the whole trigger.

`Cargo.lock` **is tracked** — a version bump changes the `sonara` /
`sonara-python` entries in it, so stage it with the bump. sonara keeps a
committed **`CHANGELOG.md`**: the release commit adds its entry (this is part of
the record, alongside git history — not skipped).

## Preconditions
- Run `python scripts/sync_workflow_skills.py --check-installed`. Missing or
  drifted installed workflow mirrors block release until they are synchronized
  from the committed canonical sources.
- Check no release is already staged: `git log origin/main..HEAD --oneline | grep -iE "release"`.
  If it returns a release commit not yet pushed, **keep that version** — fold
  work into the same `[x.y.z]` bump (one version bump per push).
- On `main` (or a fold-into-main branch). Working tree ideally clean — but
  if there's **unrelated uncommitted work**, don't block on it and don't sweep
  it in: **stage the release files explicitly by path**
  (`git add sonara/Cargo.toml sonara-python/Cargo.toml pyproject.toml CHANGELOG.md Cargo.lock`,
  never `git add -A`/`.`) and leave the unrelated changes untouched. Verify with
  `git status --porcelain` that only those files are staged.

## Steps
1. **Goal check — did we achieve what we set out to do?** If this release ships
   a `phased-plan` project, read its plan (`dev-docs/plans/<slug>.md`) and the
   PR checklist, and confirm every planned phase actually shipped. List any
   phase that was **dropped, deferred, or only partially done**, and surface
   the gaps to the user before bumping. Each gap is a conscious choice: finish
   it now, or carry it to `dev-docs/todos.md` — don't let it vanish silently.
2. **Gate — run the full local suite (THE release gate).** Treat a green run
   here, **not** a green CI badge, as the gate. Run, in order:
   - `cargo test -p sonara` — pure-Rust core + accuracy suite. On macOS also
     `cargo test -p sonara --features accelerate` (matches the macOS CI leg).
   - `cargo clippy -p sonara` — lint (surfaced).
   - `maturin develop --release -m sonara-python/Cargo.toml` — build the bindings
     into the run venv (confirm a successful build; don't read status through a
     `tail`/`head` pipe).
   - The Python API scripts, exactly as CI runs them:
     `python scripts/run_python_tests.py`.
   - **If this release changes detection/accuracy**, route the changed paths via
     `python scripts/run_fidelity_gate.py --changed <paths>`. A blocked domain
     or unavailable required local dataset blocks the release. Run any broader
     labeled local gate required by the plan as well; report-only scripts do
     not count as accuracy evidence.
   - **If this release could touch the `default = []` fast path**, run
     `cargo bench -p sonara` and confirm no default-path regression; record the
     numbers to `dev-docs/bench/results/results.csv`.

   Fix any failure before bumping — never bump over a red gate.
3. **Bump version — patch by default** (`x.y.Z` → `x.y.Z+1`), editing **all
   three** fields to the *same* value: `sonara/Cargo.toml` line 3,
   `sonara-python/Cargo.toml` line 3, `pyproject.toml` line 7. If the changes
   warrant a **minor/major** bump (new feature, breaking change), STOP and ask
   one quick clarification question first; otherwise proceed with the patch bump.
   Run `cargo build` (or `cargo update -p sonara -p sonara-python --precise
   <ver>`) so `Cargo.lock` picks up the new versions.
4. **Update `CHANGELOG.md`.** Add a new `## [x.y.z] - <YYYY-MM-DD>` section at
   the top (below the intro), following the existing style:
   - A **### Validated on real music** paragraph when the release touches
     detection/accuracy — the concrete before/after evidence from the step-2
     fidelity gate (dataset size, BPM/key/similarity numbers). sonara's
     signature; don't ship a detection change without it.
   - `### Added` / `### Changed` / `### Fixed` bullets, user-visible and terse.
   Use the session date; convert any relative date to absolute.
5. **Commit** as the final step: `release(x.y.z): ...` — one commit carrying the
   three version files, `CHANGELOG.md`, and `Cargo.lock`. Stage by path
   (`git add sonara/Cargo.toml sonara-python/Cargo.toml pyproject.toml
   CHANGELOG.md Cargo.lock`), never `-A`.
6. **Push — invoking `/release` is the authorization.** Running this skill
   authorizes the `main` push it produces (the publish-triggering one) — no
   separate in-the-moment "push" prompt (this is the one carve-out to the
   standing "no push without approval" rule, scoped to this release run: the
   `release(x.y.z)` push + its CI fix-and-push loop, lapsing once published or
   the user pivots). All pre-push safeguards still apply: gate green, surgical
   staging, ff-merge clean.
   - **ff mechanic — push the branch HEAD straight to `main`, don't
     `checkout main`.** When unrelated WIP sits in the working tree, a local
     `git checkout main` drags it across and risks conflicts. Instead: confirm
     fast-forward (`git merge-base --is-ancestor origin/main HEAD`), then
     `git push origin HEAD:<branch>` (update the PR, if any) and
     `git push origin HEAD:main` (ff `main` → triggers publish). The working
     tree never moves.
7. **Poll CI + publish until green.** Poll the GitHub **Actions API** directly
   (`gh run list`, `gh run view`). On the `main` push, `ci.yml` runs the test
   test + artifact matrix, then — only on push to `main` and only if the
   pyproject version is new on PyPI — the `publish` job rechecks and publishes
   the four verified abi3 wheels plus sdist to PyPI, tags
   `vx.y.z`, and creates the GitHub release. The publish job self-skips if the
   version already exists on PyPI, which is normal on a no-op re-push.
   - CI fix-and-push loop: if a push fails on a shipped-code/infra bug (not a
     scope change), push `fix(...)`/`ci(...)` without re-asking until green.
     Stop after ~3 iterations or any release-shape change.
8. **Verify published**: PyPI shows `sonara` at `x.y.z` and exactly the promised
   four wheels plus sdist:
   `curl -s https://pypi.org/pypi/sonara/json | jq -r .info.version`. Confirm the
   `vx.y.z` tag + GitHub release exist (`gh release view vx.y.z`). (There is no
   crates.io publish to verify.)
9. **Delete the released branch** if this shipped from a feature branch (the
   `/release` invocation authorizes it; no prompt). Once publish is verified the
   branch is fully ff-merged into `main`. Delete local + remote without
   disturbing WIP: `git branch -f main origin/main`, `git switch main`
   (zero-diff when `main == HEAD`, so WIP is preserved), `git branch -d
   <branch>` (refuses if unmerged — don't `-D` past that), `git push origin
   --delete <branch>`. Confirm the PR shows `MERGED`. **Never** delete
   `gh-pages` or open `dependabot/*` branches.
10. **Tidy dev-docs — perform directly, no prompt** (the `/release` invocation
    is the authorization). Follow the **`dev-docs-cleanup`** logic, which is
    todos.md-driven: auto-purge the time-boxed dirs, then read **only
    `todos.md`** — archive the now-shipped plan to `dev-docs/bin/` and prune its
    `todos.md` entry, move any other completed/stale entries' docs to `bin/`,
    and trim the entries (read a backlinked doc only to confirm it shipped).
    Carry the step-1 gaps into `todos.md`. Don't read `designs/` or sweep
    through `plans/`.

## Notes
- Keep responses under 400 tokens; write long diffs/logs to a file and report the path.
- Version source of truth for **publish gating** is `pyproject.toml` line 7 —
  but the two `Cargo.toml` fields must match it exactly. Three fields, one
  number; bump them together.
- Publish is push-triggered and idempotent (version-gated): a re-push at the
  same version is a safe no-op, not a double-publish.
- CI publishes only the artifact set that passed native smoke installation and
  `scripts/check_release_artifacts.py`; the publish job must never rebuild it.
