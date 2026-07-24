"""Public API checks for Sonara's bundled aggression model."""

from pathlib import Path
import sys
import tempfile
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import sonara


passed = 0


def test(name, fn):
    global passed
    fn()
    passed += 1
    print(f"  PASS  {name}")


def metadata():
    assert sonara.AGGRESSION_MODEL_VERSION == 3
    assert sonara.AGGRESSION_SAMPLE_RATE == 22_050
    assert sonara.AGGRESSION_EMBEDDING_VERSION == sonara.SIMILARITY_VERSION
    assert sonara.AGGRESSION_MODEL_ID == "aggression-rank-v3-sr22050"
    assert sonara.LEGACY_AGGRESSION_MODEL_ID == "aggression-logistic-v1"


def frozen_vectors():
    zeros = sonara.aggression_score([0.0] * sonara.EMBEDDING_DIM)
    ones = sonara.aggression_score([1.0] * sonara.EMBEDDING_DIM)
    ramp = sonara.aggression_score(np.linspace(0.0, 1.0, sonara.EMBEDDING_DIM, dtype=np.float32))
    assert np.float32(zeros).view(np.uint32) == 0x3DC24C01
    assert np.float32(ones).view(np.uint32) == 0x3F49EB2B
    assert np.float32(ramp).view(np.uint32) == 0x3E9EF194


def errors():
    for values in ([0.0] * 47, [float("nan")] * 48, [float("inf")] * 48):
        try:
            sonara.aggression_score(values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid embedding accepted")
    try:
        sonara.aggression_score([0.0] * 48, embedding_version=999)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong embedding version accepted")


def signal_parity():
    sr = 22_050
    time = np.arange(sr, dtype=np.float32) / sr
    signal = (0.3 * np.sin(2 * np.pi * 220 * time) + 0.2 * np.sin(2 * np.pi * 660 * time)).astype(np.float32)
    direct = sonara.analyze_aggression_signal(signal, sr=sr)
    fused = sonara.analyze_signal(signal, sr=sr, features=["aggression"])
    analysis = sonara.analyze_signal(signal, sr=sr, features=["aggression", "embedding"])
    scored = sonara.aggression_score(
        analysis["embedding"], embedding_version=analysis["embedding_version"]
    )
    assert 0.0 <= direct["aggression_score"] <= 1.0
    assert direct["aggression_score"] == fused["aggression_score"] == analysis["aggression_score"]
    assert 0.0 <= scored <= 1.0  # Explicitly retained legacy embedding scorer.
    assert direct["aggression_model_id"] == sonara.AGGRESSION_MODEL_ID
    for key in (
        "aggression_confidence", "aggression_forcefulness", "aggression_harshness",
        "aggression_tension", "aggression_rhythm",
    ):
        assert 0.0 <= direct[key] <= 1.0, key
    assert fused["provenance"]["aggression_model_id"] == sonara.AGGRESSION_MODEL_ID
    assert "embedding" not in fused and "embedding_version" not in fused
    for dependency in (
        "mfcc_mean", "chroma_mean", "spectral_contrast_mean", "energy",
        "danceability", "key", "valence", "dissonance", "chord_sequence",
    ):
        assert dependency not in fused, dependency


def _canonical_fixture():
    sr = sonara.AGGRESSION_SAMPLE_RATE
    time = np.arange(sr * 10, dtype=np.float32) / sr
    carrier = np.sin(2 * np.pi * 180 * time)
    edge = np.sign(np.sin(2 * np.pi * 2_900 * time))
    pulse = np.where(
        (time * 5.0) % 1.0 < 0.12,
        np.sin(2 * np.pi * 900 * time),
        0.0,
    )
    return (0.28 * carrier + 0.08 * edge + 0.22 * pulse).astype(np.float32)


def _resample_linear(signal, source_rate, target_rate):
    length = round(len(signal) * target_rate / source_rate)
    positions = np.arange(length, dtype=np.float64) * source_rate / target_rate
    return np.interp(positions, np.arange(len(signal)), signal).astype(np.float32)


def cross_rate_routes():
    canonical = _canonical_fixture()
    reference = sonara.analyze_aggression_signal(
        canonical, sr=sonara.AGGRESSION_SAMPLE_RATE
    )
    component_keys = (
        "aggression_score", "aggression_confidence", "aggression_forcefulness",
        "aggression_harshness", "aggression_tension", "aggression_rhythm",
    )
    for sr in (32_000, 44_100, 48_000):
        signal = _resample_linear(canonical, sonara.AGGRESSION_SAMPLE_RATE, sr)
        direct = sonara.analyze_aggression_signal(signal, sr=sr)
        fused = sonara.analyze_signal(signal, sr=sr, features=["aggression"])
        for key in component_keys:
            assert abs(direct[key] - reference[key]) <= 0.03, (sr, key)
            assert direct[key] == fused[key], (sr, key)
        assert fused["provenance"]["sample_rate"] == sr
        assert fused["provenance"]["aggression_model_id"] == sonara.AGGRESSION_MODEL_ID

    sr = 48_000
    signal = _resample_linear(canonical, sonara.AGGRESSION_SAMPLE_RATE, sr)
    generic = sonara.analyze_signal(signal, sr=sr, features=["rms"])
    fused = sonara.analyze_signal(signal, sr=sr, features=["rms", "aggression"])
    for key in ("bpm", "beats", "rms_mean", "spectral_centroid_mean"):
        assert fused[key] == generic[key], key
    assert fused["provenance"]["requested_features"] == ["aggression", "rms"]

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture-48k.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sr)
            output.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
        file_reference = sonara.analyze_aggression_file(
            str(path), sr=sonara.AGGRESSION_SAMPLE_RATE
        )
        for requested_rate in (0, 32_000, 44_100, 48_000):
            file_result = sonara.analyze_aggression_file(str(path), sr=requested_rate)
            fused_file = sonara.analyze_file(
                str(path), sr=requested_rate, features=["aggression"]
            )
            for key in component_keys:
                assert file_result[key] == file_reference[key], (requested_rate, key)
                assert file_result[key] == fused_file[key], (requested_rate, key)
        batch = sonara.analyze_aggression_batch([str(path), str(path)], sr=sr)
        for key in component_keys:
            assert batch[0][key] == file_reference[key] == batch[1][key]


def batch_contract():
    sr = 22_050
    time = np.arange(sr, dtype=np.float32) / sr
    signal = (0.4 * np.sin(2 * np.pi * 330 * time)).astype(np.float32)
    with tempfile.TemporaryDirectory() as directory:
        valid = Path(directory) / "valid.wav"
        missing = Path(directory) / "missing.wav"
        with wave.open(str(valid), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sr)
            output.writeframes((signal * 32767).astype("<i2").tobytes())
        results = sonara.analyze_aggression_batch([str(valid), str(missing)], sr=sr)
        assert len(results) == 2
        assert results[0]["path"] == str(valid)
        assert results[0]["aggression_score"] == sonara.analyze_aggression_file(str(valid), sr=sr)["aggression_score"]
        assert results[1]["path"] == str(missing)
        assert results[1]["error_kind"] == "io"


def silence_abstains():
    result = sonara.analyze_aggression_signal(np.zeros(22_050, dtype=np.float32))
    assert result["aggression_score"] is None
    assert result["aggression_confidence"] == 0.0


test("model metadata", metadata)
test("frozen vectors", frozen_vectors)
test("invalid inputs", errors)
test("signal and embedding parity", signal_parity)
test("cross-rate signal/file/batch/fused routes", cross_rate_routes)
test("batch order and error isolation", batch_contract)
test("silence abstention", silence_abstains)

print(f"\n{passed} aggression API checks passed")
