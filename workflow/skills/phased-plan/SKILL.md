---
name: phased-plan
description: Run a large feature or refactor as a gated, phased project. Starts with an investigation phase (read-only Explore agents map scale and impacted paths) — NOT standard plan mode — then builds a custom gated phased plan, creates a branch + draft PR against main for CI tracking, and executes each phase autonomously (code → test → lint → commit → push) until done. Ships only via the release skill.
---

# Phased plan

For any large feature or non-trivial refactor. **Demand this skill** when the
user kicks off such work. Do **not** use standard plan mode
(`EnterPlanMode` / `ExitPlanMode`) — this skill builds its own gated phased
plan instead of the harness's generic plan.

## Working dir: `dev-docs/` (gitignored)
All plans, scratch, and intermediates live under **`dev-docs/`** — gitignored
local working state. **The canonical layout + lifecycle is `dev-docs/README.md`
— read it; it's the source of truth, this is just the phased-plan-relevant
subset:**
- This project's plan → **`dev-docs/plans/<slug>.md`** (durable).
- Design choices/trade-offs you weigh → **`dev-docs/designs/`** (durable).
- Open threads → a lean one-line backlink in **`dev-docs/todos.md`** (detail in
  the linked durable doc, never inline).
- **Offload large output to `dev-docs/temp/` and report the path** (>1-day
  purge) instead of printing it — stays under the response token gate.
- Benchmarks: harnesses → **`bench/scripts/`**, regression rows →
  **`bench/results/results.csv`**, heavy generated dumps → **`bench/out/`**
  (>14-day purge; never write artifacts next to the script).

## Phase −1 — Start fresh (recommend cleanup first)
Before investigating, **recommend the user run the `dev-docs-cleanup` skill**
so we start from a tidy `dev-docs/` and a current `todos.md`. Relevant
carried-over todos can then be folded into this plan — **only with the user's
go-ahead.** If they decline, proceed without it.

## Phase 0 — Investigation (get a feel for scale before committing to a plan)
- **Do not enter plan mode.** Investigate first, plan second.
- **Read-only until approval.** The main loop makes **zero edits** during
  Phase 0 and Phase 1 — no branch, no PR, no code, no file writes. All
  investigation goes through **read-only `Explore` agents**; nothing touches
  the working tree until the user approves the plan in Phase 1.
- Kick off **investigator agents** (`Explore`). Fan them out in parallel — one
  per subsystem / suspected blast-radius area. sonara's blast-radius map:
  the pure-Rust core DSP (`sonara/src/core/` — `fft.rs`, `constantq.rs`,
  `spectrum.rs`, `pitch.rs`, `harmonic.rs`, `notation.rs`, …), the DSP
  primitives (`sonara/src/dsp/`), the feature extractors
  (`sonara/src/feature/`), the top-level analysis modules
  (`sonara/src/analyze.rs`, `beat.rs`, `beatgrid.rs`, `tonal.rs`, `onset.rs`,
  `segment.rs`, `sequence.rs`, `structure.rs`, `similarity.rs`,
  `fingerprint.rs`, `perceptual.rs`, `vocal.rs`, `loudness_ext.rs`,
  `decompose.rs`, `effects.rs`, `filters.rs`), the **PyO3 layer**
  (`#[pymethods]`/`#[pyfunction]` in `sonara-python/src/`), and the Python
  wrapper (`python/sonara/` — `__init__.py`, `_result.py`, `display.py`).
  **Scale the count to blast radius:** 1–2 for a medium change, more only for a
  genuinely large one; don't over-spend on investigation. Have them report:
  structure of the affected area, impacted paths / callers, hidden couplings,
  existing test coverage (`tests/python/test_*.py` + Rust `#[cfg(test)]` units),
  and a rough size estimate.
- If this is a bug-driven refactor: reproduce and confirm the **root cause with
  evidence** before planning the fix (repository working rules, "Working
  style"). For a
  detection/accuracy bug (BPM, key, chord, beatgrid, similarity), capture the
  failing case as a **deterministic test fixture** — a synthesized signal or a
  committable reference vector, never a copyrighted audio file.
- **For a behaviour-preserving refactor, probe current behaviour first.** Write
  a throwaway scratch script (or a temporary test) that exercises the code paths
  you're about to move and capture their *actual* outputs (analysis results /
  feature vectors / detected values) — don't trust your mental model. Catching
  latent bugs before planning beats discovering them mid-execution.
- **Confirm your intended safety net actually catches *this class* of change.**
  Decide the net in Phase 0: which `cargo test -p sonara` cases / Python API
  scripts / fixtures prove the behaviour, and whether they actually exercise the
  path you're changing (the PyO3 layer and the pure-Rust core need *separate*
  coverage — a Python-only run misses Rust-core unit regressions and vice-versa).
- **If the change could touch the `default = []` fast path**, capture a criterion
  baseline now (`cargo bench -p sonara`, the relevant `bench_*`) so you can prove
  no regression at the gate. New capability that would slow the default path
  belongs behind a cargo feature — decide that boundary in Phase 0.
- Synthesize their findings into a scale read: small/medium/large, risk hot
  spots, what could invalidate a naive plan.

## Phase 1 — Build the gated phased plan
- Write the plan to **`dev-docs/plans/<slug>.md`** (the durable copy; the PR
  description in Phase 2 mirrors it as a checklist).
- Break the work into numbered phases. Each phase must be independently
  **buildable, testable, committable** (bisectable).
- For each phase spell out: the change, the tests that prove it, the green gate.
- No phase touches the version fields (`sonara/Cargo.toml`,
  `sonara-python/Cargo.toml`, `pyproject.toml`), `CHANGELOG.md`, or publish
  config — shipping is the `release` skill's job.
- Present the plan, then **invite revision: ask the user to revise or approve,
  and loop on their feedback until they approve.**
- **This is the stage that invites design critique — raise it now or hold it.**
  "I would have designed this differently" belongs here, before the code exists,
  where an alternative can be argued and settled cheaply. Say so explicitly when
  presenting the plan, and put your own reservations about the chosen shape in
  the plan doc. After approval, review measures the implementation against *this*
  plan and against correctness — a design opinion formed mid-diff is input to the
  **next** plan, not a review finding (the review half of this rule is in the
  repository working rules, "Code review — report what is broken").
- **Hard stop — wait for an explicit go-ahead.** Do not create the branch, open
  the PR, or write any code until the user says proceed (e.g. "proceed", "go
  ahead", "approved", "ship it"). A simple proceed is enough — no formal
  sign-off. Until then, stay read-only.
- Once approved, **do not pause between phases.**

## Phase 2 — Branch + draft PR (the CI tracking handle)
- Create a feature branch: `feat/<slug>` or `refactor/<slug>` (never work the
  project directly on `main`).
- Push the branch and **open a draft PR against `main`**. This is what makes
  CI run on the branch: `ci.yml` triggers on `pull_request`, so every push to
  the branch now runs the full CI matrix (Rust tests on Linux/macOS/Windows +
  the maturin build + the Python API scripts) — while **nothing publishes** (the
  `publish` job is `if: github.event_name == 'push' && github.ref ==
  'refs/heads/main'` and version-gated, so it never fires for a branch PR).
- Put the phased plan into the **PR description as a checklist** (one box per
  phase). The PR tab then shows plan + progress + CI status in one place.
- **Run the CI-only sweep once, before the first push** — not per phase.
  Everything the fast local loop skips comes back as a red first CI run, one
  full cycle per blocker. sonara's sweep is three commands:
  - `python scripts/run_fidelity_gate.py --base <merge-base> --dry-run
    --check-contract` — routing runs on the **ubuntu CI job only**, so it is
    invisible locally until it blocks you.
  - `cargo test -p sonara --features accelerate` — the macOS CI leg.
  - `python scripts/sync_workflow_skills.py --check-installed` — read **its own**
    exit code, never a pipe's.

## Phase 3 — Execute each phase (the autonomous loop)
For every phase, in order:
1. Implement the phase's code + its tests.
2. **Local green gate before committing** (mirror CI):
   - `cargo test -p sonara` (the pure-Rust core + accuracy suite — the headline
     gate). On macOS also run `cargo test -p sonara --features accelerate` to
     match the macOS CI leg.
   - `cargo clippy -p sonara` (lint hygiene — keep it clean).
   - Build the bindings so the PyO3 layer + Python suite run against the real
     extension: `maturin develop --release -m sonara-python/Cargo.toml` inside
     the project venv (confirm maturin prints a successful build — don't read
     status through a `tail`/`head` pipe).
   - The Python API tests run as **scripts**, matching CI:
     `python scripts/run_python_tests.py`.
     (`test_tonal_batch.py` is the large-local-dataset real-music gate — run it
     only at the Phase 4 fidelity gate, not every phase.)
   Run **both** the Rust and Python suites after any Rust behaviour change — a
   Python-only subset skips the pure-Rust-core assertions and vice-versa.
   **Gate on what *this* change could break; run the full battery once, at the
   end.** A phase's gate is chosen for its blast radius — the touched surface
   plus its direct consumers (a `#[pymethods]`/`#[pyfunction]` edit → the Python
   API scripts **and** the Rust-side unit). The both-suites rule above is that
   "direct consumers" case, not a contradiction. The whole suite runs **once**,
   at the project's completion, over the union of what every phase touched; a
   repo-wide re-run after each phase buys wall clock, not information.
   **A NEW GATE IS NOT TRUSTED UNTIL YOU HAVE SEEN IT FAIL.** If the phase adds
   or changes a check — a test, a CI step, an assertion in a script — break the
   thing it guards, confirm it goes red, then restore. Reading a gate cannot tell
   you whether it works: every vacuous gate found on 2026-07-28 looked correct,
   and the only thing that separated the live ones from the dead ones was
   mutation. Three ways a gate is born dead:
   - **Substring subsumption.** `assert "cmd" in block` also matches
     `cmd --self-test`, so deleting the real invocation stays green. Compare
     whole stripped lines, not `in`.
   - **Comment subsumption.** The words you assert on usually also appear in the
     comment explaining them. Strip comment lines before matching.
   - **`exit` inside `$( )`.** A shell guard that exits inside command
     substitution kills only the subshell; the caller reads the empty output as 0
     and passes. Return a sentinel the caller checks.
   **Verify the probe, not just the result.** A mutation that silently edited the
   wrong text makes a working gate look broken, and an unchanged file makes a dead
   gate look alive. Confirm the subject actually changed before believing either
   verdict.
   **A pipeline reports its LAST stage's status**, so `cmd … | tail` says 0 for a
   `cmd` that exited non-zero. Never declare a phase green off piped output.
3. **Commit** the phase (`feat(...)` / `refactor(...)` / `fix(...)`), one
   commit per phase. Do **not** edit `CHANGELOG.md` here — the changelog entry
   is written once, at `release` time.
4. **Push to the branch** → CI runs on the PR for that phase. Tick the phase's
   checkbox in the PR description. (Pushing the *branch* is fine without a
   prompt; only the `main` push at release time is approval-gated.)
5. **Retire any `todos.md` action this phase completed.** If the phase shipped
   work that fully closes a backlog thread, do the same soft-delete tidy
   `dev-docs-cleanup` performs — at phase-commit time, not as a separate pass:
   - **Fully done** → remove the backlink line from `todos.md` and move its
     supporting `plans/<doc>.md` to `dev-docs/bin/` (7-day grace, `mv`).
   - **Partially done** → leave the doc; trim the entry to only what's left.
   - **Shared doc** (a `plans/` file backing several todos, e.g.
     `consider-for-future.md`) → remove only the closed entry; move the doc to
     `bin/` *only* once no live backlink points at it.
   `dev-docs/` is gitignored, so this is local bookkeeping alongside the commit,
   not part of the git change. Note each retirement in the report-out.
6. Continue into the next phase. If a phase's CI comes back red, fold the fix
   into the loop before the project merges — don't leave the PR red.

**Final branch gate — the one full-battery run.** When the last phase is
committed, run the whole suite once over the union of everything the phases
touched: `cargo test -p sonara` (plus `--features accelerate`),
`cargo clippy -p sonara`, a fresh `maturin develop --release -m
sonara-python/Cargo.toml`, `python scripts/run_python_tests.py`, and the routed
`python scripts/run_fidelity_gate.py --base <merge-base> --check-contract`.
This is what the per-phase targeting defers to; without it, "full battery at the
end" has no owner and never runs.

Stop mid-plan only for a genuine blocker (unfixable test, architectural
surprise invalidating a later phase). Surface it; don't push through.

**Bugs that surface mid-plan — fix them as they surface, don't step over them.**
When executing a phase reveals a defect:
- **In scope** (same file/subsystem you're touching): reproduce + confirm the
  root cause, then fix it as its **own bisectable phase** — insert a `Phase Nb`
  with its own test/fixture + commit. Don't fold a behaviour change into a
  mechanical-refactor commit — keep bisection clean.
- **Out of scope** (different subsystem): don't silently leave it. Reproduce,
  confirm, file it to `dev-docs/plans/consider-for-future.md` with a `todos.md`
  backlink, and add a cheap deterministic regression fixture if one fits.
Either way, record it in the **report-out** below — a discovered bug never
vanishes.

## Phase 4 — Fidelity / perf gate (only for detection- or perf-sensitive work)
Before declaring done on detection-accuracy or throughput work:
- **Accuracy** — run the real-music gate: the accuracy suite in
  `cargo test -p sonara` plus `tests/python/test_tonal_batch.py` over the large
  local music library. Per `CONTRIBUTING.md`, a detection change (BPM, key,
  chord, beatgrid, similarity) must show **before/after** on a labeled set —
  octave-error rate, median BPM error, key accuracy — and must not regress. If
  the local dataset is absent, say so; do not claim the gate passed.
- **Perf** — if the change could touch the `default = []` fast path, run
  `cargo bench -p sonara` (the relevant `bench_*`, `--features accelerate` on
  macOS to match CI) and record pre/post numbers to `bench/results/results.csv`.
  A default-path regression is a blocker, not a follow-up. Confirm new capability
  stayed behind its cargo feature.
Fix regressions now, not in a follow-up.

**Three ways a number reads clean and wrong.** Every one of these produces a
plausible, low-concern reading, so a clean result deserves the same suspicion as
a bad one.
1. **`min` is only for a steady-state repeated op.** A *once-per-event* cost —
   cold decode, model load, first-call init — has no steady state, and its min is
   a warm-cache run no user ever sees; report the **mean of first events** across
   fresh instances. Judge a heavy-tailed bench whose min sits far below its own
   median by median/mean instead: the min is reporting the lucky round. Always
   state which statistic a number is.
2. **Carry an unchanged-path control bench in every comparison.** If the control
   moved too, you measured the machine (thermals, background load, cold caches
   right after a build), not the code. **No control cell, no verdict.** This is
   what replaces waiting for an idle machine — run the capture under whatever
   load exists and let the instrument carry the validity.
3. **Measure the headline quantity two independent ways** — criterion's own
   timer versus wall clock around the call. The two routes diverge exactly when
   the instrument is broken, which a single route reports as a clean result.

## Report out (when the plan completes, before Ship)
Surface a concise summary so nothing actioned silently — keep it under the
400-token rule and link the plan doc for detail:
- **Phases** done (one line each) + the PR link / final commit shas.
- **Bugs surfaced** during execution and each one's disposition: *fixed in
  Phase Nb* (name the fix) or *filed to backlog* (`consider-for-future.md` +
  `todos.md`). This list is mandatory even if empty ("no bugs surfaced").
- **Fidelity/perf gate** result if run (accuracy before/after + real-music
  verdict; pre/post throughput if measured; default-path perf held or not).
- **`todos.md` changes**: actions *retired* (entry removed + doc → `bin/`,
  per phase) and carried-over / out-of-scope items *added*.
- **Plan deviations** (inserted phases, re-scopes) and why.

## Phase 5 — Ship (only on request)
When the user asks to ship, run the **`release`** skill. It ff-merges the
branch into `main`, bumps the version in **all three** fields
(`sonara/Cargo.toml`, `sonara-python/Cargo.toml`, `pyproject.toml` — kept in
lockstep), adds the `CHANGELOG.md` entry, commits, and — with explicit approval
— pushes `main` (triggering the auto-version-gated PyPI sdist publish + GitHub
release) and verifies PyPI. This skill never bumps, never edits the changelog,
never pushes `main`.

## Notes
- Keep responses under 400 tokens; write long diffs/logs to a file, report the path.
- Branch pushes during the loop are routine (no publish). Only the **`main`**
  push at `release` time is the approval-gated one (per your standing "no push
  without approval" rule).
- Changed a `#[pymethods]`/`#[pyfunction]` in `sonara-python/src/`? Add/adjust
  the matching Python API script coverage **and** the Rust-side unit test — the
  binding and the core are tested separately.
