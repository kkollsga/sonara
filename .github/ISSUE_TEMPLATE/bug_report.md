---
name: Bug report
about: Report incorrect behavior or a crash
title: ''
labels: bug
assignees: ''
---

## Description

A clear description of what went wrong and what you expected instead.

## Environment

- sonara version:
- OS:
- Python version:
- Audio format / codec (e.g. MP3 CBR 320, FLAC, WAV PCM):

## Reproduction

Minimal code or steps that trigger the bug. If it depends on a specific file, describe it (format, sample rate, duration) or link one if you can.

```python
import sonara
# ...
```

## Accuracy bugs (BPM / key / chords)

Fill this in if the issue is a wrong detected value rather than a crash:

- Detected value:
- Expected / true value:
- Reference tool used for the expected value (e.g. Mixed In Key, Rekordbox, Serato):
- Does the error look like a 2x / 0.5x octave error (BPM), or a relative/parallel key confusion? Yes / No / Unsure
