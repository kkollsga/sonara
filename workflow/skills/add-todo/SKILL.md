---
name: add-todo
description: Capture work into the dev-docs backlog the right way — scope it, put the detail in a plans/ doc (reuse an existing one or create a new one), and add a lean backlink line to todos.md under the correct section. Handles both a single one-off item (`/add-todo <free-text>`) and a deeper body of analysis (research output, an audit, a review) that decomposes into several actionable items. The canonical authority on todo-entry shape — other skills (e.g. read-inbox) defer here for how a todo is written.
---

# add-todo

Capture work into the dev-docs backlog the right way, respecting the convention:

> **`todos.md` holds one lean backlink line per thread; the detail lives in a
> `plans/*.md` file.** Never put detail in `todos.md`; never leave a `plans/`
> doc unlinked.

The whole point is to capture fast *and* scope well, so the item is actionable
later without rediscovery. Do the work directly; only ask the user when a
classification or scope decision is genuinely ambiguous.

**This skill is the single authority on *how a todo entry is shaped*** —
classification, the lean-backlink format, detail-in-`plans/`, fix-site
grounding. Other skills that file todos (e.g. `read-inbox`) follow the entry
rules below rather than restating them.

`dev-docs/todos.md` is the *local* working index. It is **not** `CHANGELOG.md`
(the committed, shipped-work record at the repo root) — don't file shipped-work
notes here, and don't touch `CHANGELOG.md` from this skill.

## Two modes

- **One-off** — a single free-text item (`/add-todo <description>`). Run
  steps 1–6 below once.
- **Batch / deeper analysis** — a body of findings (a research report, an
  audit, a code review, an accuracy review, inbox content) that contains
  *several* actionable items. Decompose it first, then run the per-item logic
  for each. See **§0 Decompose** before the single-item steps.

## 0. Decompose (batch mode only)
When the input is analysis rather than one ask, first split it into discrete,
*independently actionable* items — each a single change a future session could
pick up alone. For the whole set, before filing:
- **Drop non-actionable material** — background, confirmations, "no action"
  conclusions. A todo is something to *do*, not a record of what was read.
- **Group by theme.** Items that share a subsystem go in *one* `plans/` doc as
  sections (one backlink), not N scattered docs. Distinct threads get their own.
- **Dedup against the existing backlog** — read `todos.md` first; fold an item
  that extends an existing thread into that thread instead of a new line.
- **Order by priority/effort** so the backlink hooks read sensibly.
Then run steps 2–5 for each resulting item (step 1's index read is done once).
Keep the report (step 6) to the set: one line per filed entry.

## 1. Read the index + understand the ask
Read `dev-docs/todos.md` (the section layout + existing backlinks) and
`ls dev-docs/plans/`. Parse the user's text into: **type**, **the concrete
change**, and any **evidence** they gave. If the text is too terse to classify,
infer from context; ask only if you truly can't place it.

## 2. Classify → target `todos.md` section
- **Surfaced defect / wrong behaviour** → `## Bugs (surfaced, not yet fixed)`
- **Enhancement / optimization / code-health / refactor** → `## Engineering backlog (live)`
- **Migration / pipeline follow-up** → `## Migration follow-ups`
- **Deliberately deferred scope-creep** → goes in `plans/consider-for-future.md`
  (the parking lot), backlinked from the relevant section.

## 3. Ground it (cheap, high-value)
For anything touching code, spend one or two `grep`/`Read` calls to **pin the
fix site** (file:line) and confirm it's real — a scoped entry with a concrete
location is worth far more than a vague one. For a claimed detection/accuracy
bug (BPM, key, chord, similarity), confirm it's a real defect and not intended
behaviour before filing it as a bug (read the surrounding code/tests). Convert
any relative date to an absolute one.

## 4. Choose the detail home (reuse first)
- **Fits an existing `plans/` doc's theme** → append a section to that doc.
  Prefer this.
- **Substantial new standalone thread** → create `plans/<kebab-title>.md`.
- **Small deferred item** → append to `plans/consider-for-future.md`.

Scope the detail with these bullets (adapt to the item):
- **What it is** — the concrete change.
- **Why it matters (long-run)** — the leverage, not just the symptom.
- **Evidence** — reproduction / failing track / bench, if any.
- **Fix site + approach** — `file:line` + the shape of the change.
- **Test** — for correctness bugs, the test the fix must land: a Rust unit
  (`cargo test -p sonara`) and/or a Python API script
  (`tests/python/test_*.py`). For a detection/accuracy change, the
  before/after evidence it needs on a real-music set per `CONTRIBUTING.md`
  (octave-error rate, median BPM error, key accuracy).
- **Perf note** — if it could touch the `default = []` fast path, the criterion
  before/after it must carry (`cargo bench -p sonara`) before it's called done.
- **Effort** — rough size.

## 5. Add the lean backlink
Append **one line** to the chosen `todos.md` section:
`- <short title> → [plans/<doc>.md](plans/<doc>.md) — <≤200-char hook with fix-site + effort>. Surfaced <date>.`
Match the terse style of the existing lines. Do not duplicate the detail.

## 6. Report
State: the section it went under, the plans/ doc (new vs appended), and the
one-line backlink — nothing more. Keep under ~200 tokens.

## Notes
- This skill **adds**; it never prunes. Cleanup/triage of stale entries is
  `dev-docs-cleanup`'s job.
- Don't bump the version, edit `CHANGELOG.md`, or touch code — this is backlog
  capture only.
- **Batch input is the norm for decomposed analysis** — research reports,
  audits, reviews, and inbox triage all surface multiple items. Use §0 to
  split + group, then file each. `read-inbox` lifts its actionable items as
  todos following these entry rules (it owns the inbox-specific routing,
  status footer, and archival; it does not restate todo shape).
