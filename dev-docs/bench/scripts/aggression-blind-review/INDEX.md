# Aggression blind-review — evidence package

**Canonical local path (not in any shared workspace — this tree is gitignored):**
`/Volumes/EksternalHome/Koding/Rust/sonara/dev-docs/bench/scripts/aggression-blind-review/`

## Evaluator disclosure
The labeling model cannot itself audition audio. Labels were produced by an
**independent audio-perception rater** — LAION CLAP (`clap-htsat-unfused`),
whose audio encoder ingests the waveform directly — used as the evaluator.
`audio_perception_confirmed: true` in the label files refers to that rater's
**gate-verified** hearing (see `out/perception_gate_clap.json`), not a human or
LLM listening.

## Files
| Path | What it is | Shareable |
|---|---|---|
| `development_labels.json` | 24 dev pairs, protocol JSON (open) | yes |
| `locked_labels.json` | 20 locked pairs — **SEALED**, contents withheld | hash only |
| `packets/dev_pairs.jsonl` | dev manifest: pair structure + clip sha256 + source | yes |
| `packets/locked_pairs.jsonl` | locked manifest: pairing + hashes (no judgments) | yes |
| `packets/clips/<sha>.wav` | 20 s anonymized excerpts, content-addressed | local only¹ |
| `rater/*.py` | deterministic rater + packet builder + validator | yes |
| `controls/*.wav` | synthesized perception-gate controls | yes |
| `out/perception_gate_clap.json` | perception-gate result | yes |
| `SHA256SUMS.txt` | checksums over all of the above | yes |

¹ The excerpts are from commercial recordings; they are **not** bundled for
external transfer. They remain at the local path for on-machine validation.

## Seal policy
`locked_labels.json` stays sealed and is **not** used for any tuning until
Sonara's candidate, thresholds, transforms, and model choice are frozen. Its
SHA-256 and pair/tie/abstain counts are published for integrity + disjointness
validation; its judgments are revealed only on an explicit "final locked
evaluation" request. Hash-disjointness (dev vs locked) can be validated now
from the two manifests without opening the sealed file.

## Validate
```
cd rater
python validate.py            # counts, hash-disjointness, sources, schema
python sanity_gate.py clap     # re-verify the rater perceives audio
shasum -c ../SHA256SUMS.txt    # (run from repo root-relative dir)
```
