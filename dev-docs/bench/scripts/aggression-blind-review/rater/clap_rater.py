"""CLAP zero-shot aggression rater.

Genuinely ingests the waveform (not features/spectrograms as a proxy for MY
judgement -- CLAP's own audio encoder consumes the samples) and scores it
against calibrated aggression vs non-aggression text anchors. Coarse on the
*why* (reason tags are anchor-derived, not reasoned), but real, reproducible,
independent audio perception that fits 16 GB. ~600 MB, CPU/MPS.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from base import Rater, ClipVerdict, REASON_TAGS
from audio_io import decode_normalized, rms_dbfs, TARGET_SR

MODEL_ID = "laion/clap-htsat-unfused"

# Anchor groups: aggression vs its absence. Averaged, not cherry-picked.
AGGR_ANCHORS = [
    "harsh abrasive aggressive music",
    "distorted overdriven hostile sound",
    "brutal forceful confrontational intensity",
    "screaming harsh vocals over a wall of noise",
    "rough gritty punishing physical impact",
]
CALM_ANCHORS = [
    "calm gentle soft soothing music",
    "smooth mellow relaxed peaceful sound",
    "quiet delicate tender ambient music",
    "clean warm consonant harmony",
    "light easy pleasant background music",
]

# Coarse reason-tag probes. Sim above the group's z-threshold -> tag emitted.
TAG_PROBES = {
    "roughness": "rough gritty grainy sound texture",
    "distortion": "distorted overdriven clipping fuzzy tone",
    "abrasive_high_band": "harsh bright piercing abrasive high frequencies",
    "forceful_attacks": "hard punchy sharp percussive forceful attacks",
    "vocal_intensity": "screaming shouting intense aggressive vocals",
    "tonal_tension": "dissonant tense clashing unstable harmony",
    "sustained_impact": "relentless heavy sustained physical impact wall of sound",
    "regular_dance_energy": "steady four on the floor dance beat club groove",
}

SILENCE_DBFS = -45.0  # after loudnorm, real music sits well above this


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClapRater(Rater):
    name = "clap-htsat-unfused"

    def __init__(self, device: str | None = None):
        self.device = device or _pick_device()
        self.model = ClapModel.from_pretrained(MODEL_ID).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(MODEL_ID)
        self._text_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _text_feats(self, texts: list[str]) -> torch.Tensor:
        key = "".join(texts)
        if key not in self._text_cache:
            inp = self.processor(text=texts, return_tensors="pt", padding=True)
            inp = {k: v.to(self.device) for k, v in inp.items()}
            feats = self.model.get_text_features(**inp)
            self._text_cache[key] = torch.nn.functional.normalize(feats, dim=-1)
        return self._text_cache[key]

    @torch.no_grad()
    def _audio_feats(self, wave: np.ndarray) -> torch.Tensor:
        inp = self.processor(audios=wave, sampling_rate=TARGET_SR, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        feats = self.model.get_audio_features(**inp)
        return torch.nn.functional.normalize(feats, dim=-1)

    def score_clip(self, path: str) -> ClipVerdict:
        wave = decode_normalized(path)
        level = rms_dbfs(wave)
        if level < SILENCE_DBFS or wave.size < TARGET_SR // 2:
            return ClipVerdict(
                aggression=0.0, confidence=0.9,
                reason_tags=["insufficient_music_content"],
                insufficient=True,
                raw={"rms_dbfs": level, "reason": "near-silent / too short"},
            ).clean()

        a = self._audio_feats(wave)                       # (1, D)
        aggr = self._text_feats(AGGR_ANCHORS)             # (Na, D)
        calm = self._text_feats(CALM_ANCHORS)             # (Nc, D)

        sim_aggr = (a @ aggr.T).squeeze(0)                # cosine sims
        sim_calm = (a @ calm.T).squeeze(0)
        scale = self.model.logit_scale_a.exp().item()

        # Softmax margin between the two anchor groups -> P(aggressive).
        logit_aggr = scale * sim_aggr.mean().item()
        logit_calm = scale * sim_calm.mean().item()
        p_aggr = float(np.exp(logit_aggr) / (np.exp(logit_aggr) + np.exp(logit_calm)))
        aggression = 100.0 * p_aggr

        # Confidence: margin magnitude + within-group agreement.
        margin = abs(p_aggr - 0.5) * 2.0
        agree = 1.0 - float(sim_aggr.std().item() + sim_calm.std().item())
        confidence = max(0.0, min(1.0, 0.5 * margin + 0.5 * max(0.0, agree)))

        # Coarse reason tags: probe sims, z-scored; keep those above +0.5 sigma.
        tags: list[str] = []
        probe_names = list(TAG_PROBES.keys())
        probe_feats = self._text_feats([TAG_PROBES[n] for n in probe_names])
        probe_sims = (a @ probe_feats.T).squeeze(0).cpu().numpy()
        mu, sd = float(probe_sims.mean()), float(probe_sims.std()) or 1e-6
        for name, s in zip(probe_names, probe_sims):
            z = (float(s) - mu) / sd
            if z > 0.5:
                tags.append(name)
        if not tags:
            tags = ["other"]
        # Only surface aggression-supporting tags when the clip reads aggressive.
        if aggression < 40 and "regular_dance_energy" not in tags:
            tags = [t for t in tags if t in ("regular_dance_energy",)] or ["other"]

        return ClipVerdict(
            aggression=aggression,
            confidence=confidence,
            reason_tags=[t for t in tags if t in REASON_TAGS],
            insufficient=False,
            raw={
                "p_aggressive": p_aggr,
                "mean_sim_aggr": float(sim_aggr.mean().item()),
                "mean_sim_calm": float(sim_calm.mean().item()),
                "rms_dbfs": level,
                "device": self.device,
            },
        ).clean()
