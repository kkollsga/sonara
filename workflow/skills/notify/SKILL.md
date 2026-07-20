---
name: notify
description: Send a coordination/feedback note to another local project's inbox. Resolves a target repo by name anywhere under the Koding/ parent tree, composes a message file per the inbox schema, and drops it in that repo's inbox/unread/ (creating the folder if missing).
---

# notify

Deliver a message to a sibling project's inbox so its maintainer/agent picks it
up on their next `read-inbox`. Input: a **target repo** (name or path) and the
**message** (topic + body; compose from the conversation if not given).

## 1. Resolve the target repo path
The target lives somewhere under the `Koding/` parent (category folders like
`Rust/`, `Go/`, `JS/`, `Python/`, `mcp-servers/`; repos sit at depth 1–2,
sometimes deeper). Search by name (case-insensitive):

```bash
KODING="${PWD%%/Koding/*}/Koding"
find "$KODING" -maxdepth 3 -type d -iname '<name>' \
  -not -path '*/node_modules/*' -not -path '*/.git/*' \
  -not -path '*/__pycache__/*' -not -path '*/target/*' \
  -not -path '*/.venv/*' -not -path '*/mcp-servers/*'
```

- **`mcp-servers/` is one externally-managed project, not a tree of repos.**
  Its subdirs (`code_review/`, `open_source/`, …) are **not** notify
  targets — that's why `*/mcp-servers/*` is excluded above. To reach anything in
  that ecosystem, target **`mcp-servers`** itself (the top dir is still
  matchable) → its single `inbox/unread/`. Never resolve a name to
  `mcp-servers/<subdir>/`.
- **Exactly one match** → use it.
- **Several matches** → prefer a git repo (has `.git/`); if still ambiguous,
  **ask the user which path** (show the candidates).
- **No match** → widen with `-maxdepth 4`, then ask the user for the path.
- If the caller gave an absolute path directly, skip the search and use it.

Confirm the resolved path before writing if there was any ambiguity.

## 2. Ensure the inbox exists
```bash
mkdir -p "<target>/inbox/unread"
```
(Create it if the project has no inbox yet — that's expected for a first note.)

## 3. Compose the message (the schema)
Filename: **`<YYYY-MM-DD>-from-sonara-<topic-slug>.md`** (date = session
date, kebab-case topic). Body:

```markdown
# <Short title>

- **From:** sonara
- **To:** <target repo>
- **Date:** <YYYY-MM-DD>
- **Type:** feedback | bug | coordination | heads-up | request
- **Re:** <optional — version, file, PR, or prior message it responds to>

<1–3 paragraphs of context: what happened / what's needed and why.>

## Ask / action requested
- <concrete, actionable item(s) — or "FYI, no action needed">

## References
- <links, file paths, commit SHAs, versions — optional>
```

Keep it actionable: only file a note if there's something for them to do or
genuinely useful to know (repository working rules, "Route to the party who
can act").

## 4. Write + report
Write the file to `<target>/inbox/unread/<filename>` and report the full path.
Don't move or touch anything in our own inbox — this skill only *sends*.

## Notes
- Keep the response under 400 tokens.
- This is the send side; `read-inbox` is the receive side. Same filename schema
  (`YYYY-MM-DD-from-<sender>-<topic>.md`) so the recipient's triage just works.
- Sending writes into another project's working tree — if the resolved target
  was ambiguous, confirm with the user before writing.
