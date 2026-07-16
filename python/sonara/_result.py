"""TrackAnalysis: dict subclass returned by analyze_file / analyze_signal / analyze_batch."""

from __future__ import annotations


def _fmt_duration(sec: float) -> str:
    total = int(round(sec))
    m, s = divmod(total, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class TrackAnalysis(dict):
    """Result of `sonara.analyze_*`. Behaves as a dict; adds `.print()` for a human-readable summary."""

    @property
    def failed(self) -> bool:
        """True if this entry is a per-file failure from `analyze_batch`."""
        return "error" in self

    def __repr__(self) -> str:
        if self.failed:
            return f"<TrackAnalysis FAILED [{self.get('error_kind')}] {self.get('path')}>"
        parts = []
        if "bpm" in self:
            parts.append(f"{self['bpm']:.0f} BPM")
        if "key" in self:
            parts.append(str(self["key"]))
        if "energy" in self:
            parts.append(f"energy {self['energy']:.2f}")
        if "duration_sec" in self:
            parts.append(_fmt_duration(self["duration_sec"]))
        return f"<TrackAnalysis {' | '.join(parts)}>" if parts else "<TrackAnalysis>"

    def print(self) -> None:
        """Print a mode-aware summary, including only fields that were computed."""
        if self.failed:
            print(
                f"TrackAnalysis  FAILED ({self.get('error_kind')})\n"
                f"  path   {self.get('path')}\n"
                f"  error  {self.get('error')}"
            )
            return
        lines: list[str] = []
        if "duration_sec" in self:
            lines.append(f"TrackAnalysis  ({_fmt_duration(self['duration_sec'])})")
        else:
            lines.append("TrackAnalysis")

        rhythm: list[tuple[str, str]] = []
        if "bpm" in self:
            rhythm.append(("BPM", f"{self['bpm']:.1f}"))
        if "bpm_raw" in self and abs(self["bpm_raw"] - self.get("bpm", self["bpm_raw"])) > 0.05:
            rhythm.append(("BPM (raw)", f"{self['bpm_raw']:.1f}"))
        if self.get("bpm_candidates"):
            top = ", ".join(f"{bpm:.1f}" for bpm, _score in self["bpm_candidates"][:3])
            rhythm.append(("BPM candidates", top))
        if "n_beats" in self:
            rhythm.append(("Beats", str(self["n_beats"])))
        if "onset_density" in self:
            rhythm.append(("Onset density", f"{self['onset_density']:.2f}/sec"))
        if "tempo_variability" in self:
            rhythm.append(("Tempo variability", f"{self['tempo_variability']:.3f}"))
        if "time_signature" in self:
            ts = self["time_signature"]
            conf = self.get("time_signature_confidence")
            rhythm.append(("Time signature", f"{ts}  (conf {conf:.2f})" if conf is not None else str(ts)))
        # --- silence ---
        if "leading_silence_sec" in self or "trailing_silence_sec" in self:
            lead = self.get("leading_silence_sec", 0.0)
            trail = self.get("trailing_silence_sec", 0.0)
            rhythm.append(("Silence", f"{lead:.2f}s lead / {trail:.2f}s trail"))

        # --- beat grid ---
        beatgrid: list[tuple[str, str]] = []
        if "grid_offset_sec" in self:
            beatgrid.append(("Grid offset", f"{self['grid_offset_sec']:.3f} sec"))
        if "downbeats" in self:
            beatgrid.append(("Downbeats", str(len(self["downbeats"]))))
        if "grid_stability" in self:
            beatgrid.append(("Grid stability", f"{self['grid_stability']:.3f}"))

        tonal: list[tuple[str, str]] = []
        if "key" in self:
            key_str = str(self["key"])
            camelot = self.get("key_camelot")
            if camelot is not None:
                key_str = f"{key_str} ({camelot})"
            conf = self.get("key_confidence")
            tonal.append(("Key", f"{key_str}  (conf {conf:.2f})" if conf is not None else key_str))
        if "predominant_chord" in self:
            tonal.append(("Predominant chord", str(self["predominant_chord"])))
        if "chord_change_rate" in self:
            tonal.append(("Chord changes", f"{self['chord_change_rate']:.2f}/sec"))
        if "dissonance" in self:
            tonal.append(("Dissonance", f"{self['dissonance']:.3f}"))
        # --- key candidates ---
        if "key_candidates" in self and self["key_candidates"]:
            def _short_key(name: str) -> str:
                parts = str(name).split()
                if len(parts) == 2 and parts[1] == "minor":
                    return f"{parts[0]}m"
                return parts[0] if parts else str(name)
            cands = " · ".join(
                f"{_short_key(k)} {float(s):.2f}" for k, _cam, s in self["key_candidates"][:3]
            )
            tonal.append(("Key candidates", cands))

        perceptual: list[tuple[str, str]] = []
        for key, label in (
            ("energy", "Energy"),
            ("danceability", "Danceability"),
            ("valence", "Valence"),
            ("acousticness", "Acousticness"),
            ("instrumentalness", "Instrumentalness"),
        ):
            if key in self:
                perceptual.append((label, f"{self[key]:.2f}"))
        if "loudness_lufs" in self:
            perceptual.append(("Loudness", f"{self['loudness_lufs']:.1f} LUFS"))
        if "dynamic_range_db" in self:
            perceptual.append(("Dynamic range", f"{self['dynamic_range_db']:.1f} dB"))
        # Extended loudness / gain metrics (opt-in via features=["loudness"]).
        if "true_peak_db" in self:
            perceptual.append(("True peak", f"{self['true_peak_db']:.1f} dBTP"))
        if "replaygain_db" in self:
            perceptual.append(("ReplayGain", f"{self['replaygain_db']:+.1f} dB"))
        if "loudness_range_lu" in self:
            perceptual.append(("Loudness range", f"{self['loudness_range_lu']:.1f} LU"))
        # --- vocalness ---
        if "vocalness" in self:
            perceptual.append(("Vocalness", f"{self['vocalness']:.2f}"))
        # --- mood (heuristic v1) ---
        if "mood_happy" in self:
            perceptual.append((
                "Mood",
                f"happy {self['mood_happy']:.2f} · relaxed {self['mood_relaxed']:.2f} "
                f"· sad {self['mood_sad']:.2f} · aggressive {self['mood_aggressive']:.2f}",
            ))
        # --- similarity ---
        if "embedding" in self:
            ver = self.get("embedding_version", "?")
            perceptual.append(("Embedding", f"{len(self['embedding'])}-dim v{ver}"))

        spectral: list[tuple[str, str]] = []
        if "spectral_centroid_mean" in self:
            spectral.append(("Centroid", f"{self['spectral_centroid_mean']:.0f} Hz"))
        if "spectral_bandwidth_mean" in self:
            spectral.append(("Bandwidth", f"{self['spectral_bandwidth_mean']:.0f} Hz"))
        if "spectral_rolloff_mean" in self:
            spectral.append(("Rolloff", f"{self['spectral_rolloff_mean']:.0f} Hz"))
        if "spectral_flatness_mean" in self:
            spectral.append(("Flatness", f"{self['spectral_flatness_mean']:.3f}"))
        if "zero_crossing_rate" in self:
            spectral.append(("ZCR", f"{self['zero_crossing_rate']:.3f}"))

        # --- tags ---
        tags: list[tuple[str, str]] = []
        if "tags" in self and self["tags"]:
            t = self["tags"]
            for key, label in (
                ("title", "Title"),
                ("artist", "Artist"),
                ("album", "Album"),
                ("genre", "Genre"),
            ):
                if key in t:
                    tags.append((label, str(t[key])))
            if "year" in t:
                tags.append(("Year", str(t["year"])))
            if "track_no" in t:
                tags.append(("Track", str(t["track_no"])))

        # --- structure ---
        structure: list[tuple[str, str]] = []
        if "energy_level" in self:
            structure.append(("Energy level", f"{self['energy_level']}/10"))
        if "segments" in self:
            structure.append(("Segments", str(len(self["segments"]))))
        if "intro_end_sec" in self:
            structure.append(("Intro end", _fmt_duration(self["intro_end_sec"])))
        if "outro_start_sec" in self:
            structure.append(("Outro start", _fmt_duration(self["outro_start_sec"])))

        for name, rows in (
            ("Rhythm", rhythm),
            ("Tonal", tonal),
            ("Perceptual", perceptual),
            ("Spectral", spectral),
            # --- beat grid ---
            ("Beat grid", beatgrid),
            # --- structure ---
            ("Structure", structure),
            # --- tags ---
            ("Tags", tags),
        ):
            if not rows:
                continue
            lines.append("")
            lines.append(f"  {name}")
            width = max(len(label) for label, _ in rows)
            for label, value in rows:
                lines.append(f"    {label:<{width}}  {value}")

        # --- provenance ---
        if "provenance" in self:
            p = self["provenance"]
            parts = [
                f"schema v{p['schema_version']}",
                f"{p['sample_rate']} Hz",
                f"hop {p['hop_length']}",
                str(p.get("mode", "?")),
            ]
            if "requested_features" in p:
                parts.append(f"features [{', '.join(p['requested_features'])}]")
            lines.append("")
            lines.append(f"  Provenance  {' · '.join(parts)}")

        # --- fingerprint ---
        if "fingerprint" in self:
            fp = self["fingerprint"]
            ver = self.get("fingerprint_version", 1)
            # Each sub-fingerprint is a little-endian u32 (4 bytes); recover the
            # count from the base64 string length without decoding it.
            n_bytes = (len(fp) // 4) * 3 - fp.count("=")
            n_sub = n_bytes // 4
            lines.append("")
            lines.append(f"  Fingerprint  v{ver} ({n_sub} subprints)")

        print("\n".join(lines))
