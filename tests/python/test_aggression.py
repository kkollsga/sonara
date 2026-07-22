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
    assert sonara.AGGRESSION_MODEL_VERSION == 1
    assert sonara.AGGRESSION_EMBEDDING_VERSION == sonara.SIMILARITY_VERSION
    assert sonara.AGGRESSION_MODEL_ID == "aggression-logistic-v1"


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
    assert 0.0 <= direct <= 1.0
    assert direct == fused["aggression_score"] == analysis["aggression_score"] == scored
    assert fused["provenance"]["aggression_model_id"] == sonara.AGGRESSION_MODEL_ID
    assert "embedding" not in fused and "embedding_version" not in fused
    for dependency in (
        "mfcc_mean", "chroma_mean", "spectral_contrast_mean", "energy",
        "danceability", "key", "valence", "dissonance", "chord_sequence",
    ):
        assert dependency not in fused, dependency


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
        assert results[0]["aggression_score"] == sonara.analyze_aggression_file(str(valid), sr=sr)
        assert results[1]["path"] == str(missing)
        assert results[1]["error_kind"] == "io"


test("model metadata", metadata)
test("frozen vectors", frozen_vectors)
test("invalid inputs", errors)
test("signal and embedding parity", signal_parity)
test("batch order and error isolation", batch_contract)

print(f"\n{passed} aggression API checks passed")
