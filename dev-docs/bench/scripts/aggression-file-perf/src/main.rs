use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::json;
use sha2::{Digest, Sha256};
use sonara::analyze::{self, AnalysisConfig, AnalysisMode, TrackAnalysis};

#[derive(Debug)]
struct Options {
    workers: usize,
    requested_rate: u32,
    aggression: bool,
    repeats: usize,
    paths: Vec<PathBuf>,
}

fn parse_options() -> Result<Options, String> {
    let mut workers = 1;
    let mut requested_rate = 0;
    let mut aggression = false;
    let mut repeats = 1;
    let mut paths = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--workers" => {
                workers = args
                    .next()
                    .ok_or("--workers requires a value")?
                    .parse()
                    .map_err(|_| "invalid --workers value")?;
            }
            "--requested-rate" => {
                requested_rate = args
                    .next()
                    .ok_or("--requested-rate requires a value")?
                    .parse()
                    .map_err(|_| "invalid --requested-rate value")?;
            }
            "--aggression" => aggression = true,
            "--repeats" => {
                repeats = args
                    .next()
                    .ok_or("--repeats requires a value")?
                    .parse()
                    .map_err(|_| "invalid --repeats value")?;
            }
            "--help" | "-h" => {
                return Err(
                    "usage: sonara-aggression-file-perf [--workers N] [--requested-rate SR] [--aggression] [--repeats N] FILE..."
                        .to_owned(),
                );
            }
            _ if argument.starts_with('-') => {
                return Err(format!("unknown option: {argument}"));
            }
            _ => paths.push(PathBuf::from(argument)),
        }
    }
    if workers == 0 || repeats == 0 || paths.is_empty() {
        return Err(
            "workers/repeats must be positive and at least one file is required".to_owned(),
        );
    }
    Ok(Options {
        workers,
        requested_rate,
        aggression,
        repeats,
        paths,
    })
}

fn broad_config(with_aggression: bool) -> Result<AnalysisConfig, String> {
    let mut features = HashSet::from([
        "bandwidth".to_owned(),
        "rolloff".to_owned(),
        "flatness".to_owned(),
        "contrast".to_owned(),
        "mfcc".to_owned(),
        "chroma".to_owned(),
        "chords".to_owned(),
        "dissonance".to_owned(),
        "energy".to_owned(),
        "danceability".to_owned(),
        "key".to_owned(),
        "valence".to_owned(),
        "acousticness".to_owned(),
        "tempo_curve".to_owned(),
        "time_signature".to_owned(),
        "tags".to_owned(),
        "mood".to_owned(),
        "instrumentalness".to_owned(),
        "loudness".to_owned(),
        "structure".to_owned(),
        "beatgrid".to_owned(),
        "silence".to_owned(),
        "embedding".to_owned(),
        "vocalness".to_owned(),
        "key_candidates".to_owned(),
    ]);
    if with_aggression {
        features.insert("aggression".to_owned());
    }
    Ok(AnalysisConfig {
        mode: AnalysisMode::Playlist,
        features: Some(features),
        vocalness_model: Some(Arc::new(
            sonara::vocal_model::bundled().map_err(|error| error.to_string())?,
        )),
        ..AnalysisConfig::default()
    })
}

fn percentile(sorted: &[u128], quantile: f64) -> u128 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

fn cohort_hash(paths: &[PathBuf]) -> Result<String, String> {
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    for path in paths {
        digest.update(path.to_string_lossy().as_bytes());
        digest.update([0]);
        let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        digest.update([0xff]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn config_hash(config: &AnalysisConfig, requested_rate: u32) -> String {
    let mut features = config
        .features
        .as_ref()
        .map(|values| values.iter().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    features.sort_unstable();
    let mut digest = Sha256::new();
    digest.update(format!("mode={:?}\n", config.mode).as_bytes());
    digest.update(format!("requested_rate={requested_rate}\n").as_bytes());
    for feature in features {
        digest.update(feature.as_bytes());
        digest.update([0]);
    }
    format!("{:x}", digest.finalize())
}

fn result_hash(results: &[TrackAnalysis]) -> String {
    let mut digest = Sha256::new();
    for result in results {
        digest.update(
            format!(
                "{:?}",
                (
                    result.duration_sec,
                    result.bpm,
                    result.bpm_raw,
                    result.bpm_confidence,
                    &result.bpm_candidates,
                    &result.beats,
                    &result.onset_frames,
                    result.rms_mean,
                    result.rms_max,
                    result.spectral_centroid_mean,
                    result.zero_crossing_rate,
                    result.onset_density,
                )
            )
            .as_bytes(),
        );
        digest.update(format!("{:?}", &result.embedding).as_bytes());
        digest.update(
            format!(
                "{:?}",
                (
                    result.aggression_score,
                    result.aggression_confidence,
                    result.aggression_forcefulness,
                    result.aggression_harshness,
                    result.aggression_tension,
                    result.aggression_rhythm,
                )
            )
            .as_bytes(),
        );
        digest.update([0]);
    }
    format!("{:x}", digest.finalize())
}

#[cfg(unix)]
fn peak_rss_raw() -> i64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status == 0 {
        unsafe { usage.assume_init() }.ru_maxrss
    } else {
        -1
    }
}

#[cfg(not(unix))]
fn peak_rss_raw() -> i64 {
    -1
}

fn run_once(options: &Options, config: &AnalysisConfig) -> Result<serde_json::Value, String> {
    let next = AtomicUsize::new(0);
    let slots = (0..options.paths.len())
        .map(|_| Mutex::new(None::<(Result<TrackAnalysis, String>, u128)>))
        .collect::<Vec<_>>();
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..options.workers.min(options.paths.len()) {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= options.paths.len() {
                    break;
                }
                let track_started = Instant::now();
                let result =
                    analyze::analyze_file(&options.paths[index], options.requested_rate, config)
                        .map_err(|error| format!("{}: {error}", options.paths[index].display()));
                let elapsed = track_started.elapsed().as_nanos();
                *slots[index].lock().expect("result slot poisoned") = Some((result, elapsed));
            });
        }
    });
    let wall = started.elapsed();
    let mut results = Vec::with_capacity(slots.len());
    let mut timings = Vec::with_capacity(slots.len());
    for slot in slots {
        let (result, elapsed) = slot
            .into_inner()
            .map_err(|_| "result slot poisoned")?
            .ok_or("worker left an empty result slot")?;
        results.push(result?);
        timings.push(elapsed);
    }
    let track_ns_ordered = timings.clone();
    timings.sort_unstable();
    Ok(json!({
        "wall_ns": wall.as_nanos(),
        "tracks_per_second": options.paths.len() as f64 / wall.as_secs_f64(),
        "track_ns_p50": percentile(&timings, 0.50),
        "track_ns_p90": percentile(&timings, 0.90),
        "track_ns_max": timings[timings.len() - 1],
        "track_ns_ordered": track_ns_ordered,
        "result_sha256": result_hash(&results),
        "peak_rss_raw": peak_rss_raw(),
    }))
}

fn main() {
    let options = parse_options().unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(2);
    });
    let config = broad_config(options.aggression).unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(2);
    });
    let cohort_sha256 = cohort_hash(&options.paths).unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(1);
    });
    let config_sha256 = config_hash(&config, options.requested_rate);
    let mut runs = Vec::with_capacity(options.repeats);
    for _ in 0..options.repeats {
        runs.push(run_once(&options, &config).unwrap_or_else(|message| {
            eprintln!("{message}");
            std::process::exit(1);
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "format": "sonara.aggression-file-perf.v1",
            "sonara_version": env!("CARGO_PKG_VERSION"),
            "aggression": options.aggression,
            "workers": options.workers,
            "requested_rate": options.requested_rate,
            "rayon_num_threads": std::env::var("RAYON_NUM_THREADS").ok(),
            "track_count": options.paths.len(),
            "cohort_sha256": cohort_sha256,
            "config_sha256": config_sha256,
            "ordered_paths": options.paths,
            "runs": runs,
        }))
        .expect("serialize report")
    );
}
