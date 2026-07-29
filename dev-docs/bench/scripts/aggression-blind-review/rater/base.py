"""Common rater interface + shared vocabulary for the aggression blind review.

A Rater turns a single loudness-matched clip into a per-clip verdict. The
pairwise reducer (built separately) consumes two verdicts to decide
left/right/tie/abstain. Keeping this interface tiny lets CLAP (coarse, fits
16 GB today) and Qwen2.5-Omni-3B (reasoning, downloading) be swapped freely.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

# Verbatim from the evaluation protocol -- the ONLY tags a rater may emit.
REASON_TAGS = [
    "roughness",
    "distortion",
    "abrasive_high_band",
    "forceful_attacks",
    "vocal_intensity",
    "tonal_tension",
    "sustained_impact",
    "regular_dance_energy",
    "insufficient_music_content",
    "other",
]

# The aggression definition + guardrails, handed to any reasoning model verbatim.
RUBRIC = (
    "Musical aggression means a harsh, forceful, confrontational, or physically "
    "intense musical presentation. Do NOT treat any of these ALONE as aggression: "
    "high energy or loudness; fast tempo; regular dance percussion; minor key; "
    "genre or artist identity; listener preference. Audible roughness, distortion, "
    "abrasive timbre, forceful attacks, vocal intensity, tonal tension, and "
    "sustained physical impact may contribute. A loud, energetic dance track may "
    "still be LESS aggressive than a quieter but harsh or forceful recording."
)


@dataclass
class ClipVerdict:
    aggression: float          # 0-100
    confidence: float          # 0.0-1.0
    reason_tags: List[str]
    insufficient: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    def clean(self) -> "ClipVerdict":
        self.aggression = float(max(0.0, min(100.0, self.aggression)))
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.reason_tags = [t for t in self.reason_tags if t in REASON_TAGS]
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Rater(ABC):
    """Scores one loudness-matched clip. Sees audio + rubric only -- no metadata."""

    name: str = "abstract"

    @abstractmethod
    def score_clip(self, path: str) -> ClipVerdict:
        ...
