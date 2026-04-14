"""Test tonal features (chords, dissonance) on 400 random tracks from the Greatest Hits dataset.

Reports:
  - Success rate (files that analyze without error)
  - Performance (total time, per-track time, throughput)
  - Feature distributions (chord change rate, dissonance, predominant chords)
  - Sanity checks (dissonance in [0,1], chord labels valid, etc.)
"""

import os
import random
import time
import sys

import sonara

DATASET = (
    "/Volumes/EksternalHome/Downloads/Apps/"
    "Greatest Hits - Collection 1958-2021 _1074 ALBUMS_ MP3 Part 2 Of 2-TL"
)
N_TRACKS = 400
SR = 22050
SEED = 42


def collect_files(root, exts=(".mp3", ".flac", ".wav", ".ogg")):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(exts):
                files.append(os.path.join(dirpath, f))
    return files


def main():
    print(f"Scanning {DATASET} ...")
    all_files = collect_files(DATASET)
    print(f"Found {len(all_files)} audio files")

    random.seed(SEED)
    sample = random.sample(all_files, min(N_TRACKS, len(all_files)))
    print(f"Selected {len(sample)} random tracks\n")

    # -- Run playlist analysis with chords + dissonance --
    results = []
    errors = []
    t0 = time.perf_counter()

    for i, path in enumerate(sample):
        try:
            r = sonara.analyze_file(path, sr=SR, mode="playlist")
            results.append(r)
        except Exception as e:
            errors.append((path, str(e)))
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{len(sample)}] {elapsed:.1f}s elapsed "
                  f"({(i+1)/elapsed:.1f} tracks/s)")

    total_time = time.perf_counter() - t0
    n_ok = len(results)
    n_err = len(errors)

    # ===== PERFORMANCE =====
    print("\n" + "=" * 60)
    print("PERFORMANCE")
    print("=" * 60)
    print(f"  Tracks analyzed:   {n_ok}/{len(sample)}")
    print(f"  Errors:            {n_err}")
    print(f"  Total time:        {total_time:.2f}s")
    print(f"  Per-track (avg):   {total_time/len(sample)*1000:.1f}ms")
    print(f"  Throughput:        {n_ok/total_time:.1f} tracks/s")

    if errors:
        print(f"\n  First 5 errors:")
        for p, e in errors[:5]:
            print(f"    {os.path.basename(p)}: {e}")

    # ===== FEATURE DISTRIBUTIONS =====
    print("\n" + "=" * 60)
    print("TONAL FEATURES")
    print("=" * 60)

    # Chord data
    has_chords = [r for r in results if "chord_sequence" in r]
    has_diss = [r for r in results if "dissonance" in r]

    print(f"\n  Tracks with chord_sequence:   {len(has_chords)}/{n_ok}")
    print(f"  Tracks with dissonance:       {len(has_diss)}/{n_ok}")

    if has_chords:
        change_rates = [r["chord_change_rate"] for r in has_chords]
        seq_lens = [len(r["chord_sequence"]) for r in has_chords]
        predominant = [r["predominant_chord"] for r in has_chords]

        print(f"\n  Chord change rate:")
        print(f"    min:    {min(change_rates):.3f} changes/s")
        print(f"    max:    {max(change_rates):.3f} changes/s")
        print(f"    mean:   {sum(change_rates)/len(change_rates):.3f} changes/s")
        print(f"    median: {sorted(change_rates)[len(change_rates)//2]:.3f} changes/s")

        print(f"\n  Chord sequence length:")
        print(f"    min:    {min(seq_lens)}")
        print(f"    max:    {max(seq_lens)}")
        print(f"    mean:   {sum(seq_lens)/len(seq_lens):.1f}")

        # Top predominant chords
        from collections import Counter
        chord_counts = Counter(predominant)
        print(f"\n  Top 10 predominant chords:")
        for chord, count in chord_counts.most_common(10):
            pct = count / len(has_chords) * 100
            print(f"    {chord:5s}: {count:4d} ({pct:.1f}%)")

        # Chord diversity across all sequences
        all_chords_flat = []
        for r in has_chords:
            all_chords_flat.extend(r["chord_sequence"])
        all_chord_counts = Counter(all_chords_flat)
        print(f"\n  All chord labels seen: {len(all_chord_counts)}")
        print(f"  Top 10 chord labels overall:")
        for chord, count in all_chord_counts.most_common(10):
            pct = count / len(all_chords_flat) * 100
            print(f"    {chord:5s}: {count:5d} ({pct:.1f}%)")

    if has_diss:
        diss_vals = [r["dissonance"] for r in has_diss]
        print(f"\n  Dissonance:")
        print(f"    min:    {min(diss_vals):.4f}")
        print(f"    max:    {max(diss_vals):.4f}")
        print(f"    mean:   {sum(diss_vals)/len(diss_vals):.4f}")
        print(f"    median: {sorted(diss_vals)[len(diss_vals)//2]:.4f}")

        # Distribution buckets
        buckets = [0] * 10
        for d in diss_vals:
            idx = min(int(d * 10), 9)
            buckets[idx] += 1
        print(f"\n  Dissonance distribution:")
        for i, count in enumerate(buckets):
            lo = i * 0.1
            hi = lo + 0.1
            bar = "#" * (count * 40 // max(max(buckets), 1))
            print(f"    [{lo:.1f}-{hi:.1f}): {count:4d} {bar}")

    # ===== SANITY CHECKS =====
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    valid_notes = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
    valid_chords = set()
    for note in valid_notes:
        valid_chords.add(note)       # major
        valid_chords.add(note + "m") # minor
    valid_chords.add("N")            # no chord

    n_invalid_chords = 0
    n_diss_out_of_range = 0
    n_chords_empty = 0

    for r in results:
        if "chord_sequence" in r:
            if len(r["chord_sequence"]) == 0:
                n_chords_empty += 1
            for c in r["chord_sequence"]:
                if c not in valid_chords:
                    n_invalid_chords += 1
        if "dissonance" in r:
            d = r["dissonance"]
            if d < 0.0 or d > 1.0:
                n_diss_out_of_range += 1

    print(f"  Invalid chord labels:        {n_invalid_chords}")
    print(f"  Empty chord sequences:       {n_chords_empty}")
    print(f"  Dissonance out of [0,1]:     {n_diss_out_of_range}")

    all_ok = n_invalid_chords == 0 and n_diss_out_of_range == 0
    print(f"\n  {'PASS' if all_ok else 'FAIL'}: all sanity checks {'passed' if all_ok else 'FAILED'}")

    # ===== EXISTING FEATURES SPOT CHECK =====
    print("\n" + "=" * 60)
    print("EXISTING FEATURES SPOT CHECK")
    print("=" * 60)

    if results:
        bpms = [r["bpm"] for r in results]
        energies = [r.get("energy", 0) for r in results if "energy" in r]
        danceabilities = [r.get("danceability", 0) for r in results if "danceability" in r]
        keys = [r.get("key", "") for r in results if "key" in r]

        print(f"  BPM:          min={min(bpms):.1f}, max={max(bpms):.1f}, "
              f"mean={sum(bpms)/len(bpms):.1f}")
        if energies:
            print(f"  Energy:       min={min(energies):.3f}, max={max(energies):.3f}, "
                  f"mean={sum(energies)/len(energies):.3f}")
        if danceabilities:
            print(f"  Danceability: min={min(danceabilities):.3f}, max={max(danceabilities):.3f}, "
                  f"mean={sum(danceabilities)/len(danceabilities):.3f}")
        if keys:
            from collections import Counter
            key_counts = Counter(keys)
            print(f"  Key distribution (top 5):")
            for k, c in key_counts.most_common(5):
                print(f"    {k:12s}: {c:4d} ({c/len(keys)*100:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
