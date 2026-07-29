#!/usr/bin/env python3
import json
import re
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "out" / "aggression-file-perf"
pattern = re.compile(r"dense-(\d+)-(\d+)s(?:(-128|-320))?\.(wav|flac|mp3)$")

for workers, rayon in [(1, 1), (1, 10), (4, 10), (10, 1), (10, 10)]:
    broad = json.loads((root / f"matrix-broad-w{workers}-r{rayon}.json").read_text())
    aggression = json.loads(
        (root / f"matrix-aggression-w{workers}-r{rayon}.json").read_text()
    )
    reverse_broad_path = root / f"matrix-broad-w{workers}-r{rayon}-reverse.json"
    reverse_aggression_path = root / f"matrix-aggression-w{workers}-r{rayon}-reverse.json"
    if reverse_broad_path.exists():
        reverse_broad = json.loads(reverse_broad_path.read_text())
        reverse_aggression = json.loads(reverse_aggression_path.read_text())
    else:
        reverse_broad = broad
        reverse_aggression = aggression
    assert broad["ordered_paths"] == aggression["ordered_paths"]
    broad_run = broad["runs"][0]
    aggression_run = aggression["runs"][0]
    reverse_broad_run = reverse_broad["runs"][0]
    reverse_aggression_run = reverse_aggression["runs"][0]
    groups = defaultdict(lambda: [0, 0])
    durations = defaultdict(lambda: [0, 0])
    files = []
    for index, (path, base_ns, candidate_ns) in enumerate(zip(
        broad["ordered_paths"],
        broad_run["track_ns_ordered"],
        aggression_run["track_ns_ordered"],
    )):
        base_ns = (base_ns + reverse_broad_run["track_ns_ordered"][index]) / 2
        candidate_ns = (
            candidate_ns + reverse_aggression_run["track_ns_ordered"][index]
        ) / 2
        match = pattern.search(path)
        assert match, path
        rate, duration, bitrate, extension = match.groups()
        codec = extension if extension != "mp3" else f"mp3{bitrate}"
        groups[(rate, codec)][0] += base_ns
        groups[(rate, codec)][1] += candidate_ns
        durations[duration][0] += base_ns
        durations[duration][1] += candidate_ns
        files.append((100.0 * (candidate_ns / base_ns - 1.0), path))
    group_overheads = {
        f"{rate}/{codec}": 100.0 * (values[1] / values[0] - 1.0)
        for (rate, codec), values in sorted(groups.items())
    }
    duration_overheads = {
        duration: 100.0 * (values[1] / values[0] - 1.0)
        for duration, values in sorted(durations.items(), key=lambda item: int(item[0]))
    }
    print(
        json.dumps(
            {
                "workers": workers,
                "rayon": rayon,
                "wall_overhead_percent": 100.0
                * (
                    (aggression_run["wall_ns"] + reverse_aggression_run["wall_ns"])
                    / (broad_run["wall_ns"] + reverse_broad_run["wall_ns"])
                    - 1.0
                ),
                "broad_peak_rss": max(
                    broad_run["peak_rss_raw"], reverse_broad_run["peak_rss_raw"]
                ),
                "aggression_peak_rss": max(
                    aggression_run["peak_rss_raw"],
                    reverse_aggression_run["peak_rss_raw"],
                ),
                "group_min_percent": min(group_overheads.values()),
                "group_max_percent": max(group_overheads.values()),
                "group_overheads_percent": group_overheads,
                "duration_overheads_percent": duration_overheads,
                "worst_file": max(files),
            },
            sort_keys=True,
        )
    )

adaptive = [
    json.loads((root / f"adaptive-{index}-{mode}-w10-r10.json").read_text())
    for index, mode in [(1, "broad"), (2, "aggression"), (3, "aggression"), (4, "broad")]
]
adaptive_runs = [value["runs"][0] for value in adaptive]
adaptive_groups = defaultdict(lambda: [0, 0])
for index, path in enumerate(adaptive[0]["ordered_paths"]):
    match = pattern.search(path)
    assert match, path
    rate, _, bitrate, extension = match.groups()
    codec = extension if extension != "mp3" else f"mp3{bitrate}"
    adaptive_groups[(rate, codec)][0] += (
        adaptive_runs[0]["track_ns_ordered"][index]
        + adaptive_runs[3]["track_ns_ordered"][index]
    ) / 2
    adaptive_groups[(rate, codec)][1] += (
        adaptive_runs[1]["track_ns_ordered"][index]
        + adaptive_runs[2]["track_ns_ordered"][index]
    ) / 2
adaptive_group_overheads = {
    f"{rate}/{codec}": 100.0 * (values[1] / values[0] - 1.0)
    for (rate, codec), values in sorted(adaptive_groups.items())
}
print(
    json.dumps(
        {
            "adaptive_abba_10x10": True,
            "wall_overhead_percent": 100.0
            * (
                (adaptive_runs[1]["wall_ns"] + adaptive_runs[2]["wall_ns"])
                / (adaptive_runs[0]["wall_ns"] + adaptive_runs[3]["wall_ns"])
                - 1.0
            ),
            "group_min_percent": min(adaptive_group_overheads.values()),
            "group_max_percent": max(adaptive_group_overheads.values()),
            "group_overheads_percent": adaptive_group_overheads,
        },
        sort_keys=True,
    )
)
