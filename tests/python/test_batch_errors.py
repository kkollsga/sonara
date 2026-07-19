#!/usr/bin/env python3
"""Test per-file error isolation in `sonara.analyze_batch`.

A single unreadable/undecodable file must never abort the batch. Every input
path must produce exactly one result entry, in input order, and failures must
carry a structured (`path`, `error`, `error_kind`) payload while valid files
still succeed.
"""

import math
import os
import struct
import sys
import tempfile
import wave

import sonara

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


def write_sine_wav(path, freq=440.0, sr=22050, seconds=2.0):
    """Synthesize a mono 16-bit PCM WAV using the stdlib `wave` module."""
    n = int(sr * seconds)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            v = int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / sr))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))


def main():
    tmp = tempfile.mkdtemp(prefix="sonara_batch_")

    valid_wav = os.path.join(tmp, "valid.wav")
    write_sine_wav(valid_wav)

    missing = os.path.join(tmp, "does_not_exist.mp3")

    # A text file masquerading as an mp3 — probe must reject it.
    fake_mp3 = os.path.join(tmp, "not_audio.mp3")
    with open(fake_mp3, "w") as f:
        f.write("this is plain text, definitely not an audio bitstream\n" * 50)

    fixtures = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    # Mid-stream frame damage → decode must skip the bad packets and succeed.
    corrupt_mp3 = os.path.join(fixtures, "corrupt.mp3")
    # Every packet damaged → decode recovery must NOT mask total failure.
    allbad_mp3 = os.path.join(fixtures, "allbad.mp3")

    paths = [valid_wav, missing, fake_mp3, corrupt_mp3, allbad_mp3]
    results = sonara.analyze_batch(paths, mode="compact")

    print("=" * 60)
    print("  analyze_batch per-file error isolation")
    print("=" * 60)

    # 1. One entry per input, in input order.
    check("one result per input path", len(results) == len(paths))

    valid_res, missing_res, fake_res, corrupt_res, allbad_res = results

    # 2. Valid file succeeded and carries real features (no error).
    check("valid file succeeded (no error key)", not valid_res.failed)
    check("valid file has bpm", "bpm" in valid_res)
    check("valid file has duration", valid_res.get("duration_sec", 0) > 1.5)
    check("valid file carries its input path", valid_res.get("path") == valid_wav)

    # 3. Missing file → structured 'io' failure, batch not aborted.
    check("missing file failed", missing_res.failed)
    check("missing file error_kind == io",
          missing_res.get("error_kind") == "io")
    check("missing file path preserved",
          missing_res.get("path") == missing)
    check("missing file error message non-empty",
          bool(missing_res.get("error")))

    # 4. Non-audio file → decode/unsupported failure (not io, not success).
    check("fake mp3 failed", fake_res.failed)
    check("fake mp3 error_kind is decode/unsupported",
          fake_res.get("error_kind") in ("decode", "unsupported_format"))
    check("fake mp3 error mentions the path",
          os.path.basename(fake_mp3) in fake_res.get("error", ""))

    # 5. Damaged-mid-stream mp3 → packet-level recovery, analysis succeeds.
    check("corrupt mp3 recovered (no error)", not corrupt_res.failed)
    check("corrupt mp3 duration ≈ 3s",
          2.5 <= corrupt_res.get("duration_sec", 0) <= 3.2)

    # 6. Every-packet-damaged mp3 → still a structured decode failure.
    check("all-bad mp3 failed", allbad_res.failed)
    check("all-bad mp3 error_kind == decode",
          allbad_res.get("error_kind") == "decode")
    check("all-bad mp3 error mentions the path",
          os.path.basename(allbad_mp3) in allbad_res.get("error", ""))

    print(f"\n  error_kinds: "
          f"{[r.get('error_kind') for r in results]}")

    # 7. Progress callback: fires once per file (completion order), reporting
    #    a monotonic `done` count and a constant `total == len(paths)`.
    calls = []
    prog_results = sonara.analyze_batch(
        paths, mode="compact",
        progress=lambda d, t: calls.append((d, t)),
    )
    check("progress called once per file", len(calls) == len(paths))
    check("progress total always == len(paths)",
          all(t == len(paths) for _, t in calls))
    check("progress done reports each completion exactly once",
          sorted(d for d, _ in calls) == list(range(1, len(paths) + 1)))
    check("progress run returns one entry per path", len(prog_results) == len(paths))
    check("progress run entries all carry 'path'",
          all(r.get("path") for r in prog_results))

    # 8. A raising callback must never abort the batch nor propagate.
    def boom(d, t):
        raise ZeroDivisionError("callback intentionally broken")

    try:
        raising_results = sonara.analyze_batch(paths, mode="compact", progress=boom)
        check("raising callback does not propagate", True)
        check("raising callback still returns all entries",
              len(raising_results) == len(paths))
    except Exception as e:  # noqa: BLE001
        check("raising callback does not propagate", False)
        print(f"    unexpected exception: {e!r}")

    # 9. A non-callable progress fails fast with TypeError.
    try:
        sonara.analyze_batch(paths, mode="compact", progress=42)
        check("non-callable progress raises TypeError", False)
    except TypeError:
        check("non-callable progress raises TypeError", True)
    except Exception as e:  # noqa: BLE001
        check("non-callable progress raises TypeError", False)
        print(f"    wrong exception type: {e!r}")

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed} PASSED, {failed} FAILED")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
