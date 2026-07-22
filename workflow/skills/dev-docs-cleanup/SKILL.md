---
name: dev-docs-cleanup
description: Tidy the gitignored dev-docs/ working folder — auto-purge time-boxed dirs, then run a todos.md-driven tidy (read only todos.md; reconcile misplaced plans/ files and stale/completed actions, reading a specific backlinked doc only when a check points at it), soft-deleting finished docs to dev-docs/bin/ and pruning their todos entries. Run before a new phased-plan to start fresh, or at the end of a release. Never reads design docs.
---

# dev-docs cleanup

`dev-docs/` accumulates project plans, intermediate files, and scratch (it's
gitignored working state). Over time it goes stale and cluttered. This skill
tidies it: nothing is hard-deleted — stale files are soft-deleted to
`dev-docs/bin/` with a 7-day grace, and any open actions are preserved in an
evergreen `dev-docs/todos.md`.

**Layout is `dev-docs/README.md` (the canonical map).** Durable dirs —
`plans/`, `designs/`, `bench/scripts/`, `bench/results/`, and `todos.md` —
are never auto-purged; only `temp/`, `bench/out/`, and `bin/` are time-boxed.

## 1. Auto-purge generated dirs (always first)
At skill start, hard-delete the three time-boxed locations. **Never touch
`bench/scripts/` or `bench/results/`** — harnesses and the benchmark result
history are durable; only `bench/out/` (heavy built artifacts) is disposable.

```bash
mkdir -p dev-docs/temp dev-docs/bin dev-docs/bench/out
find dev-docs/temp      -type f -mmin  +1440  -print -delete   # ephemeral handoff, >1 day
find dev-docs/bench/out -type f -mtime +14    -print -delete   # heavy built artifacts/dumps, >14 days
find dev-docs/bin       -type f -mtime +7     -print -delete   # soft-deleted docs, >7 days
```

Report what was purged (path list, or "nothing aged out"). Only `bench/out/`
is purged from `bench/` — the small csv/json regression record in
`bench/results/` is kept indefinitely.

## 2. Read `todos.md` — the only file read by default
`todos.md` is the index of open threads; its backlinks point to the durable
docs under `plans/`. **Read only `todos.md` to start.** Do NOT read through
`plans/`, and **never read `designs/`** — design-reference docs aren't
todos-driven (they're durable reference, out of scope for cleanup). You open a
specific doc *only* when a check below points you at it (a misplaced file, or
the backlink behind a stale action) — never the whole folder. This keeps the
cleanup cheap: one file read, plus at most the few docs the checks flag.

## 3. Two checks, both driven by `todos.md`

**a) Misplaced files in `plans/`.** `ls plans/`. Every file there should be
backlinked from `todos.md`. For any `plans/` file with **no backlink**, read
*that file only* and decide:
- a live thread missing from the index → add a one-line `todos.md` backlink; or
- finished / abandoned → soft-delete to `bin/`.
When genuinely unsure, surface it to the user. (`designs/`, `bench/`, and the
`README`/`todos.md` themselves are exempt — this check is `plans/`-only.)

**b) Stale / completed actions in `todos.md`.** Scan the entries — especially
at end-of-release, where shipped work leaves completed items behind. For any
entry that reads as done/outdated, **read its backlinked doc to confirm**, then:
- shipped / doc fully complete → move the doc to `bin/` and **remove the
  entry** (code + git history + `CHANGELOG.md` are the record);
- partially done → trim the entry to only what's left;
- superseded / abandoned → drop it (keep a one-line "Closed / dead" note only
  if it's worth not rediscovering).
Don't read a backlinked doc unless its entry looks stale — a healthy entry
needs no read.

## 4. Surface the plan
Report a short summary: what purged, misplaced `plans/` files found (+ the
decision per each), stale `todos.md` entries to prune, and docs to soft-delete
to `bin/`.
- **Run standalone** (e.g. before a phased-plan): wait for the user's go-ahead
  before moving files or editing `todos.md`. A simple proceed is enough.
- **Run inside an authorized flow** (e.g. `/release`'s tidy step): perform the
  tidy directly — no prompt — then report what was done. The flow's invocation
  is the authorization.

## 5. Soft-delete processed files
On go-ahead, move processed stale files into `dev-docs/bin/` (preserves them
for 7 days in case something was lifted wrongly). Never delete the active
plans, `todos.md`, or anything the user chose to keep.

## Output discipline
Keep the response under 400 tokens. If the stale-doc review is long, write the
full report to `dev-docs/temp/cleanup-report.md` and report that path; surface
only the new-todos list and the keep/drop confirmations inline.

## Relationship to phased-plan
`phased-plan` recommends running this skill first, so a new project starts from
a tidy `dev-docs/` and a current `todos.md`. Relevant carried-over todos can
then be folded into the new plan — only with the user's go-ahead.
