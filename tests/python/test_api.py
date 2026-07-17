#!/usr/bin/env python3
"""Test all major API usage patterns.

Verifies that sonara's core functions work correctly across
common audio analysis workflows.
"""

import sys
import traceback
import numpy as np

passed = 0
failed = 0
errors = []

def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")

import sonara

# ============================================================
# Generate test signals (no audio files needed)
# ============================================================
np.random.seed(42)
sr = 22050
y_1s = np.random.randn(sr).astype(np.float32)
y_5s = np.random.randn(5 * sr).astype(np.float32)
# Pure sine for pitch tests
y_sine = np.sin(2 * np.pi * 440.0 * np.arange(2 * sr) / sr).astype(np.float32)
# Click train for beat tests
y_clicks = np.zeros(4 * sr, dtype=np.float32)
for i in range(0, 4 * sr, sr // 2):  # 120 BPM
    y_clicks[i:i+100] = np.sin(2 * np.pi * 1000 * np.arange(100) / sr)

print("=" * 70)
print("  Testing sonara API patterns")
print("=" * 70)

# ============================================================
# Pattern 1: Basic STFT workflow (from plot_display.py)
# ============================================================
print("\n--- Pattern 1: STFT Workflow ---")

test("stft(y)", lambda: sonara.stft(y_1s))
test("stft with custom params", lambda: sonara.stft(y_1s, n_fft=4096, hop_length=256))

D = sonara.stft(y_1s)
test("amplitude_to_db(|D|)", lambda: sonara.amplitude_to_db(np.abs(D).astype(np.float32)))
test("power_to_db(|D|^2)", lambda: sonara.power_to_db(np.abs(D).astype(np.float32)**2))
test("istft(D)", lambda: sonara.istft(D))
test("istft roundtrip length", lambda: (
    np.testing.assert_equal(len(sonara.istft(D, length=len(y_1s))), len(y_1s))
))

# ============================================================
# Pattern 2: Mel spectrogram workflow (from plot_display.py)
# ============================================================
print("\n--- Pattern 2: Mel Spectrogram ---")

test("melspectrogram(y=y)", lambda: sonara.melspectrogram(y=y_1s, sr=22050.0))
test("melspectrogram(y=y, n_mels=40)", lambda: sonara.melspectrogram(y=y_1s, sr=22050.0, n_mels=40))

M = sonara.melspectrogram(y=y_1s, sr=22050.0)
test("power_to_db(mel)", lambda: sonara.power_to_db(M))

# ============================================================
# Pattern 3: MFCC workflow
# ============================================================
print("\n--- Pattern 3: MFCC ---")

test("mfcc(y=y, n_mfcc=13)", lambda: sonara.mfcc(y=y_1s, sr=22050.0, n_mfcc=13))
test("mfcc(y=y, n_mfcc=20)", lambda: sonara.mfcc(y=y_1s, sr=22050.0, n_mfcc=20))
test("mfcc(y=y, n_mfcc=40)", lambda: sonara.mfcc(y=y_1s, sr=22050.0, n_mfcc=40))
test("mfcc shape", lambda: np.testing.assert_equal(sonara.mfcc(y=y_1s, sr=22050.0, n_mfcc=13).shape[0], 13))

# ============================================================
# Pattern 4: Chroma features (from plot_chroma.py)
# ============================================================
print("\n--- Pattern 4: Chroma ---")

test("chroma_stft(y=y)", lambda: sonara.chroma_stft(y=y_1s, sr=22050.0))
test("chroma_stft shape is (12, N)", lambda: np.testing.assert_equal(sonara.chroma_stft(y=y_1s, sr=22050.0).shape[0], 12))

# ============================================================
# Pattern 5: CQT workflow (from plot_display.py)
# ============================================================
print("\n--- Pattern 5: CQT ---")

test("cqt(y)", lambda: sonara.cqt(y_1s, sr=22050))
test("cqt(n_bins=36)", lambda: sonara.cqt(y_1s, sr=22050, n_bins=36))
C = sonara.cqt(y_1s, sr=22050, n_bins=84)
test("cqt shape", lambda: np.testing.assert_equal(C.shape[0], 84))
test("amplitude_to_db(|cqt|)", lambda: sonara.amplitude_to_db(np.abs(C).astype(np.float32)))
test("vqt(y)", lambda: sonara.vqt(y_1s, sr=22050, n_bins=36))
test("pseudo_cqt(y)", lambda: sonara.pseudo_cqt(y_1s, sr=22050, n_bins=36))
test("hybrid_cqt(y)", lambda: sonara.hybrid_cqt(y_1s, sr=22050, n_bins=36))

# ============================================================
# Pattern 6: Beat tracking (from plot_dynamic_beat.py)
# ============================================================
print("\n--- Pattern 6: Beat Tracking ---")

test("beat_track(y=y)", lambda: sonara.beat_track(y=y_5s, sr=22050))
test("beat_track accepts bpm range", lambda: sonara.beat_track(y=y_clicks, sr=22050, bpm_min=79.0, bpm_max=192.0))
tempo, beats = sonara.beat_track(y=y_clicks, sr=22050)
test("beat_track returns tempo", lambda: (None if tempo > 0 else (_ for _ in ()).throw(ValueError("bad tempo"))))
test("beat_track returns beats", lambda: (None if len(beats) > 0 else (_ for _ in ()).throw(ValueError("no beats"))))
test("onset_strength(y)", lambda: sonara.onset_strength(y_5s, sr=22050))
test("onset_detect(y=y)", lambda: sonara.onset_detect(y=y_clicks, sr=22050))

# ============================================================
# Pattern 7: Spectral features
# ============================================================
print("\n--- Pattern 7: Spectral Features ---")

test("spectral_centroid(y=y)", lambda: sonara.spectral_centroid(y=y_1s, sr=22050.0))
test("rms(y=y)", lambda: sonara.rms(y=y_1s))

# ============================================================
# Pattern 8: Pitch estimation
# ============================================================
print("\n--- Pattern 8: Pitch ---")

test("yin(y, fmin, fmax)", lambda: sonara.yin(y_sine, fmin=65.0, fmax=2093.0, sr=22050, frame_length=2048))
test("pyin(y, fmin, fmax)", lambda: sonara.pyin(y_sine, fmin=65.0, fmax=2093.0, sr=22050, frame_length=2048))

f0, voiced_flag, voiced_prob = sonara.pyin(y_sine, fmin=65.0, fmax=2093.0, sr=22050, frame_length=2048)
test("pyin returns 3 arrays", lambda: np.testing.assert_equal(len(f0), len(voiced_flag)))
test("piptrack(y)", lambda: sonara.piptrack(y_1s, sr=22050))
test("estimate_tuning(y=y)", lambda: sonara.estimate_tuning(y=y_sine, sr=22050))

# ============================================================
# Pattern 9: Unit conversions
# ============================================================
print("\n--- Pattern 9: Conversions ---")

test("hz_to_mel(440)", lambda: np.testing.assert_almost_equal(sonara.hz_to_mel(440.0), 6.6, decimal=1))
test("mel_to_hz(6.6)", lambda: (None if sonara.mel_to_hz(6.6) > 400 else (_ for _ in ()).throw(ValueError())))
test("note_to_hz('A4') == 440", lambda: np.testing.assert_almost_equal(sonara.note_to_hz("A4"), 440.0, decimal=1))
test("midi_to_hz(69) == 440", lambda: np.testing.assert_almost_equal(sonara.midi_to_hz(69.0), 440.0))
test("hz_to_midi(440) == 69", lambda: np.testing.assert_almost_equal(sonara.hz_to_midi(440.0), 69.0))
test("midi_to_note(69) == 'A4'", lambda: np.testing.assert_equal(sonara.midi_to_note(69.0), "A4"))
test("hz_to_note(440)", lambda: sonara.hz_to_note(440.0))
test("hz_to_octs(440)", lambda: sonara.hz_to_octs(440.0))
test("octs_to_hz(4)", lambda: sonara.octs_to_hz(4.0))
test("A4_to_tuning(440)", lambda: np.testing.assert_almost_equal(sonara.A4_to_tuning(440.0), 0.0))
test("tuning_to_A4(0)", lambda: np.testing.assert_almost_equal(sonara.tuning_to_A4(0.0), 440.0))

# ============================================================
# Pattern 10: Frame/time/sample conversions
# ============================================================
print("\n--- Pattern 10: Frame/Time/Sample ---")

test("frames_to_samples([0,1,2])", lambda: np.testing.assert_equal(sonara.frames_to_samples([0,1,2]), [0, 512, 1024]))
test("frames_to_time([0,1])", lambda: sonara.frames_to_time([0, 1]))
test("samples_to_frames([0,512])", lambda: np.testing.assert_equal(sonara.samples_to_frames([0, 512]), [0, 1]))
test("time_to_frames([0, 1.0])", lambda: sonara.time_to_frames([0.0, 1.0]))
test("time_to_samples([0, 1.0])", lambda: sonara.time_to_samples([0.0, 1.0]))
test("samples_to_time([0, 22050])", lambda: sonara.samples_to_time([0, 22050]))
test("blocks_to_frames([0,1], block_length=2048)", lambda: sonara.blocks_to_frames([0, 1], block_length=2048))

# ============================================================
# Pattern 11: Frequency generators
# ============================================================
print("\n--- Pattern 11: Frequency Generators ---")

test("fft_frequencies(sr=22050, n_fft=2048)", lambda: np.testing.assert_equal(len(sonara.fft_frequencies(sr=22050.0, n_fft=2048)), 1025))
test("mel_frequencies(n_mels=128)", lambda: np.testing.assert_equal(len(sonara.mel_frequencies(n_mels=128)), 128))
test("cqt_frequencies(n_bins=84)", lambda: np.testing.assert_equal(len(sonara.cqt_frequencies(84)), 84))
test("tempo_frequencies(n_bins=100)", lambda: sonara.tempo_frequencies(100))

# ============================================================
# Pattern 12: Weighting functions
# ============================================================
print("\n--- Pattern 12: Weighting ---")

freqs = sonara.fft_frequencies(sr=22050.0, n_fft=2048)
test("A_weighting(freqs)", lambda: sonara.A_weighting(freqs))
test("B_weighting(freqs)", lambda: sonara.B_weighting(freqs))
test("C_weighting(freqs)", lambda: sonara.C_weighting(freqs))
test("D_weighting(freqs)", lambda: sonara.D_weighting(freqs))
test("Z_weighting(freqs)", lambda: sonara.Z_weighting(freqs))

# ============================================================
# Pattern 13: Notation (Indian music, FJS)
# ============================================================
print("\n--- Pattern 13: Notation ---")

test("key_to_notes('C:maj')", lambda: np.testing.assert_equal(sonara.key_to_notes("C:maj"), ["C","D","E","F","G","A","B"]))
test("key_to_degrees('C:maj')", lambda: sonara.key_to_degrees("C:maj"))
test("list_mela() has 72", lambda: np.testing.assert_equal(len(sonara.list_mela()), 72))
test("list_thaat() has 10", lambda: np.testing.assert_equal(len(sonara.list_thaat()), 10))
test("mela_to_svara(1)", lambda: sonara.mela_to_svara(1))
test("thaat_to_degrees('Bilaval')", lambda: sonara.thaat_to_degrees("Bilaval"))
test("fifths_to_note(0) == 'C'", lambda: np.testing.assert_equal(sonara.fifths_to_note(0), "C"))
test("interval_to_fjs(1.5) == 'P5'", lambda: np.testing.assert_equal(sonara.interval_to_fjs(1.5), "P5"))
test("pythagorean_intervals(12)", lambda: sonara.pythagorean_intervals())

# ============================================================
# Pattern 14: Svara conversions
# ============================================================
print("\n--- Pattern 14: Svara ---")

test("hz_to_svara_h(440)", lambda: sonara.hz_to_svara_h(440.0))
test("hz_to_svara_c(440)", lambda: sonara.hz_to_svara_c(440.0))
test("midi_to_svara_h(69)", lambda: sonara.midi_to_svara_h(69.0))
test("midi_to_svara_c(69)", lambda: sonara.midi_to_svara_c(69.0))
test("note_to_svara_h('A4')", lambda: sonara.note_to_svara_h("A4"))
test("note_to_svara_c('A4')", lambda: sonara.note_to_svara_c("A4"))
test("hz_to_fjs(660, ref_freq=440)", lambda: sonara.hz_to_fjs(660.0, ref_freq=440.0))

# ============================================================
# Pattern 15: Signal generation
# ============================================================
print("\n--- Pattern 15: Signal Generation ---")

test("tone(440, sr=22050, length=22050)", lambda: np.testing.assert_equal(len(sonara.tone(440.0, sr=22050, length=22050)), 22050))
test("chirp(fmin=200, fmax=2000)", lambda: sonara.chirp(fmin=200.0, fmax=2000.0, sr=22050, length=22050))
test("clicks(times=[0.5, 1.0])", lambda: sonara.clicks(times=[0.5, 1.0], sr=22050, length=22050))

# ============================================================
# Pattern 16: Audio utility
# ============================================================
print("\n--- Pattern 16: Audio Utility ---")

test("autocorrelate(y)", lambda: sonara.autocorrelate(y_1s))
test("lpc(y, order=4)", lambda: sonara.lpc(y_1s, order=4))
test("zero_crossings(y)", lambda: sonara.zero_crossings(y_1s))
test("mu_compress(y)", lambda: sonara.mu_compress(y_1s))
test("mu_expand(mu_compress(y))", lambda: sonara.mu_expand(sonara.mu_compress(y_1s)))

# ============================================================
# Pattern 17: Magphase and phase vocoder
# ============================================================
print("\n--- Pattern 17: Magphase + Phase Vocoder ---")

D = sonara.stft(y_1s)
test("magphase(D)", lambda: sonara.magphase(D))
mag, phase = sonara.magphase(D)
test("magphase: mag >= 0", lambda: np.testing.assert_array_less(-1e-10, mag))
test("phase_vocoder(D, rate=1.0)", lambda: sonara.phase_vocoder(D, rate=1.0))
test("phase_vocoder(D, rate=2.0)", lambda: sonara.phase_vocoder(D, rate=2.0))

# ============================================================
# Pattern 18: Filters
# ============================================================
print("\n--- Pattern 18: Filters ---")

test("mel filterbank", lambda: np.testing.assert_equal(sonara.mel(sr=22050.0, n_fft=2048, n_mels=128).shape, (128, 1025)))

# ============================================================
# Pattern 19: Utility functions
# ============================================================
print("\n--- Pattern 19: Utility ---")

test("samples_like(n_frames=10)", lambda: sonara.samples_like(10))
test("times_like(n_frames=10)", lambda: sonara.times_like(10))

# ============================================================
# Pattern 20: Display (requires matplotlib)
# ============================================================
print("\n--- Pattern 20: Display ---")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import sonara.display as disp

    fig, ax = plt.subplots()
    M = sonara.melspectrogram(y=y_1s, sr=22050.0)
    M_db = sonara.power_to_db(M)
    test("specshow(mel)", lambda: disp.specshow(M_db, x_axis='time', y_axis='mel', ax=ax))
    test("waveshow(y)", lambda: disp.waveshow(y_1s, sr=22050, ax=ax))
    test("cmap(data)", lambda: disp.cmap(M_db))
    plt.close('all')
except ImportError:
    test("display (matplotlib not available)", lambda: None)

# ============================================================
# Pattern 21: Fused Analysis
# ============================================================
print("\n--- Pattern 21: Fused Analysis ---")

test("analyze_signal compact", lambda: sonara.analyze_signal(y_clicks, sr=22050))
test("analyze_signal accepts bpm range", lambda: sonara.analyze_signal(y_clicks, sr=22050, bpm_min=79.0, bpm_max=192.0))


def _check_bpm_candidate_fields():
    r = sonara.analyze_signal(y_clicks, sr=22050, mode="playlist")
    assert "bpm_candidates" in r, "missing bpm_candidates"
    assert "bpm_raw" in r, "missing bpm_raw"
    cands = r["bpm_candidates"]
    assert len(cands) > 0, "expected at least one tempo candidate"
    assert len(cands) <= 5, "expected at most 5 tempo candidates"
    # Each candidate is a (bpm, score) pair.
    for c in cands:
        assert len(c) == 2, f"candidate {c} is not a (bpm, score) pair"
    # Sorted by score descending.
    scores = [c[1] for c in cands]
    assert scores == sorted(scores, reverse=True), "candidates not sorted by score desc"
    # bpm is derivable from the candidate list (equal or octave-related).
    bpm = r["bpm"]

    def close(a, b):
        return abs(a - b) <= max(2.0, 0.03 * b)

    ok = any(close(bpm, c[0]) or close(bpm, 2 * c[0]) or close(bpm, c[0] / 2) for c in cands)
    assert ok, f"bpm {bpm} not derivable from candidates {cands}"


test("analyze_signal exposes bpm_candidates/bpm_raw", _check_bpm_candidate_fields)


def _check_provenance_fields():
    r = sonara.analyze_signal(y_clicks, sr=22050)
    assert "provenance" in r, "missing provenance"
    p = r["provenance"]
    for key in ("schema_version", "sample_rate", "hop_length", "mode"):
        assert key in p, f"provenance missing {key}"
    assert p["schema_version"] >= 1
    assert p["sample_rate"] == 22050
    assert p["hop_length"] == 512
    assert p["mode"] == "compact"
    assert "requested_features" not in p, "requested_features present without features=[...]"
    # Frame -> seconds using the carried sr/hop must land inside the track.
    if r["beats"]:
        last_sec = r["beats"][-1] * p["hop_length"] / p["sample_rate"]
        assert 0.0 <= last_sec <= r["duration_sec"] + 0.1, f"beat at {last_sec}s outside track"


def _check_provenance_feature_override():
    r = sonara.analyze_signal(y_clicks, sr=22050, mode="playlist", features=["key", "energy"])
    p = r["provenance"]
    assert p["mode"] == "playlist"
    assert p["requested_features"] == ["energy", "key"], p["requested_features"]


test("analyze_signal carries provenance", _check_provenance_fields)
test("provenance records feature override (sorted)", _check_provenance_feature_override)


def _check_chord_events():
    r = sonara.analyze_signal(y_clicks, sr=22050, mode="playlist")
    assert "chord_sequence" in r, "playlist mode should compute chords"
    assert "chord_events" in r, "chord_events should mirror chord_sequence"
    events = r["chord_events"]
    seq = r["chord_sequence"]
    if not seq:
        assert events == []
        return
    for e in events:
        assert set(e) == {"label", "start_sec", "end_sec"}, f"bad event shape {e}"
        assert e["end_sec"] > e["start_sec"], f"empty span {e}"
    assert events[0]["start_sec"] == 0.0
    assert abs(events[-1]["end_sec"] - r["duration_sec"]) < 1e-4
    # Contiguous + merged (no two adjacent events share a label)
    for a, b in zip(events, events[1:]):
        assert abs(a["end_sec"] - b["start_sec"]) < 1e-9, "gap between events"
        assert a["label"] != b["label"], "adjacent events not merged"


def _check_chord_events_absent_when_not_requested():
    r = sonara.analyze_signal(y_clicks, sr=22050)  # compact: no chords
    assert "chord_events" not in r
    assert "chord_sequence" not in r


test("chord_events typed spans (playlist)", _check_chord_events)
test("chord_events absent in compact", _check_chord_events_absent_when_not_requested)


# --- tags --- opt-in file metadata passthrough (analyze_file only)
import os

_FLAC_FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "tagged.flac")
_MP3_FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "tagged.mp3")


def _check_tags_present():
    r = sonara.analyze_file(_FLAC_FIXTURE, features=["tags"])
    assert "tags" in r, "features=['tags'] should add a 'tags' sub-dict"
    t = r["tags"]
    assert t["title"] == "Test Title", t
    assert t["artist"] == "Test Artist", t
    assert t["album"] == "Test Album", t
    assert t["genre"] == "Electronic", t
    assert t["year"] == 2024, t
    # ORIGINALDATE=1969 → original_year, distinct from the file year (2024).
    assert t["original_year"] == 1969, t
    assert t["track_no"] == 3, t
    # tags must not trigger the extended DSP pass.
    assert "mfcc_mean" not in r, "tags must not enable extended features"
    assert "energy" not in r


def _check_original_year_mp3():
    # ID3v2.3 TORY=1969 → original_year; file year (TYER) stays 2024.
    t = sonara.analyze_file(_MP3_FIXTURE, features=["tags"])["tags"]
    assert t["year"] == 2024, t
    assert t["original_year"] == 1969, t


def _check_original_year_type():
    # original_year crosses the binding as an int, not a string. (The
    # file-lacks-the-tag → key absent case is covered by the Rust mapper
    # unit tests, since both audio fixtures here carry an original-date tag.)
    t = sonara.analyze_file(_FLAC_FIXTURE, features=["tags"])["tags"]
    assert isinstance(t["original_year"], int), t


def _check_tags_absent_when_not_requested():
    r = sonara.analyze_file(_FLAC_FIXTURE)  # compact, no tags feature
    assert "tags" not in r, "tags key should be absent unless features=['tags']"


test("tags sub-dict populated (analyze_file)", _check_tags_present)
test("original_year passthrough (mp3 TORY)", _check_original_year_mp3)
test("original_year is int", _check_original_year_type)
test("tags absent without feature", _check_tags_absent_when_not_requested)

# ============================================================
# Pattern 22: Chroma/key are sample-rate invariant
# ============================================================
# Regression for the chroma sr-bias bug: before the librosa octave-domain
# weighting landed in the chroma filterbank, the >11 kHz broadband band flooded
# chroma at 44.1k/48k and tonal tracks mis-detected (e.g. as F major). A C-major
# I-IV-V-I cadence of pure sines must detect as "C major" at every sample rate.
print("\n--- Pattern 22: Chroma/key sr-invariance ---")


def _c_major_progression(sr):
    # I: C5 E5 G5 / IV: F4 A4 C5 / V: G4 B4 D5 / I: C5 E5 G5, 1s each, x3.
    chords = [
        [523.25, 659.26, 783.99],
        [349.23, 440.00, 523.25],
        [392.00, 493.88, 587.33],
        [523.25, 659.26, 783.99],
    ]
    blocks = []
    for _ in range(3):
        for freqs in chords:
            t = np.arange(sr) / sr
            block = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def _check_key_sr_invariance():
    for sr in (22050, 44100, 48000):
        y = _c_major_progression(sr)
        r = sonara.analyze_signal(y, sr=sr, features=["key", "chroma"])
        assert r.get("key") == "C major", f"sr={sr}: expected C major, got {r.get('key')!r}"


test("analyze_signal key is C major at 22050/44100/48000", _check_key_sr_invariance)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"  RESULTS: {passed} PASSED, {failed} FAILED out of {passed + failed} tests")
print(f"{'='*70}")

if errors:
    print(f"\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")

sys.exit(1 if failed > 0 else 0)
