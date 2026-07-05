//! External labeled-dataset accuracy evaluator (Layer 2).
//!
//! Reads a CSV of ground-truth labels, analyzes each audio file (in parallel),
//! and reports the same accuracy metrics as the synthetic Layer 1 suite
//! (`tests/bpm_accuracy.rs`) plus a worst-offenders table — so you can validate
//! the detector against a real labeled corpus (e.g. tracks tagged with Mixed In
//! Key ground truth).
//!
//! ## CSV format
//!
//! One row per track. A header line is optional (auto-detected). Columns:
//!
//! ```text
//! path,bpm_ref[,key_ref]
//! /music/track01.mp3,128,A minor
//! /music/track02.wav,174,F# major
//! /music/track03.flac,90
//! ```
//!
//!   - `path`     — path to an audio file (no commas in the path).
//!   - `bpm_ref`  — reference tempo in BPM (float). Use empty to skip a row's BPM.
//!   - `key_ref`  — optional reference key, e.g. "A minor", "Db major", "F#m".
//!                  Enharmonic spellings (Db == C#) are normalized before compare.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --example accuracy_eval -- <labels.csv> [options]
//!
//! Options:
//!   --sr <HZ>        target sample rate for analysis (default 22050)
//!   --mode <M>       analysis mode: compact | playlist | full (default playlist)
//!                    compact is fastest but computes no key; use playlist/full
//!                    if your CSV has key_ref.
//!   --top <N>        rows in the worst-offenders table (default 20)
//!   -h, --help       print this help
//! ```
//!
//! Metrics printed: accuracy @ +/-0.5 BPM, accuracy @ +/-2%, octave-error rate,
//! median / p95 absolute BPM error, and (if `key_ref` present) key accuracy.

use std::path::{Path, PathBuf};
use std::process::exit;

use sonara::analyze;

type Float = f32;

const PCT_TOL: Float = 0.02; // +/-2%
const OCTAVE_TOL: Float = 0.04;

struct Args {
    csv: PathBuf,
    sr: u32,
    mode: String,
    top: usize,
}

const HELP: &str = "\
accuracy_eval — external labeled-dataset BPM/key accuracy evaluator

USAGE:
    cargo run --release --example accuracy_eval -- <labels.csv> [options]

CSV FORMAT (header optional, auto-detected):
    path,bpm_ref[,key_ref]
    /music/track01.mp3,128,A minor
    /music/track02.wav,174,F# major
    /music/track03.flac,90

OPTIONS:
    --sr <HZ>      target sample rate for analysis (default 22050)
    --mode <M>     analysis mode: compact | playlist | full (default playlist)
                   compact computes no key; use playlist/full for key_ref.
    --top <N>      rows in the worst-offenders table (default 20)
    -h, --help     print this help

Prints: accuracy @ +/-0.5 BPM, accuracy @ +/-2%, octave-error rate,
median / p95 absolute BPM error, key accuracy (if key_ref present),
and a worst-offenders table (top N by BPM error, with 0.5x/2x flags).";

fn parse_args() -> Args {
    let mut csv: Option<PathBuf> = None;
    let mut sr = 22050u32;
    let mut mode = "playlist".to_string();
    let mut top = 20usize;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                exit(0);
            }
            "--sr" => {
                sr = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| fail("--sr needs an integer"));
            }
            "--mode" => {
                mode = it.next().unwrap_or_else(|| fail("--mode needs a value"));
            }
            "--top" => {
                top = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| fail("--top needs an integer"));
            }
            other if other.starts_with('-') => fail(&format!("unknown option: {other}")),
            other => {
                if csv.is_some() {
                    fail("multiple CSV paths given");
                }
                csv = Some(PathBuf::from(other));
            }
        }
    }

    let csv = csv.unwrap_or_else(|| {
        eprintln!("{HELP}\n");
        fail("missing <labels.csv> argument");
    });
    Args { csv, sr, mode, top }
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    exit(2);
}

/// One CSV ground-truth row.
struct Label {
    path: String,
    bpm_ref: Option<Float>,
    key_ref: Option<String>,
}

fn parse_csv(path: &Path) -> Vec<Label> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| fail(&format!("reading {}: {e}", path.display())));
    let mut labels = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.splitn(3, ',').map(|f| f.trim()).collect();
        // Skip a header row (first line whose bpm field is non-numeric).
        if i == 0 && fields.get(1).map(|f| f.parse::<Float>().is_err()).unwrap_or(false) {
            continue;
        }
        let path = fields.first().unwrap_or(&"").to_string();
        if path.is_empty() {
            continue;
        }
        let bpm_ref = fields.get(1).and_then(|f| f.parse::<Float>().ok());
        let key_ref = fields
            .get(2)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        labels.push(Label { path, bpm_ref, key_ref });
    }
    labels
}

fn config_for(mode: &str) -> analyze::AnalysisConfig {
    match mode {
        "compact" => analyze::compact(),
        "playlist" => analyze::playlist(),
        "full" => analyze::full(),
        other => fail(&format!("unknown --mode '{other}' (compact|playlist|full)")),
    }
}

// ---- metric helpers ----

fn is_octave(detected: Float, reference: Float) -> bool {
    for &mult in &[0.5f32, 2.0, 1.0 / 3.0, 3.0] {
        let target = reference * mult;
        if (detected - target).abs() <= OCTAVE_TOL * target {
            return true;
        }
    }
    false
}

/// Which octave multiple the detection is closest to (for the flag column).
fn octave_flag(detected: Float, reference: Float) -> &'static str {
    if reference <= 0.0 {
        return "";
    }
    let ratio = detected / reference;
    if (ratio - 0.5).abs() <= OCTAVE_TOL {
        "0.5x"
    } else if (ratio - 2.0).abs() <= OCTAVE_TOL {
        "2x"
    } else if (ratio - 1.0 / 3.0).abs() <= OCTAVE_TOL {
        "1/3x"
    } else if (ratio - 3.0).abs() <= OCTAVE_TOL {
        "3x"
    } else {
        ""
    }
}

fn median(mut v: Vec<Float>) -> Float {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[m - 1] + v[m]) / 2.0
    } else {
        v[m]
    }
}

fn percentile(mut v: Vec<Float>, p: Float) -> Float {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (v.len() - 1) as Float).round() as usize;
    v[idx.min(v.len() - 1)]
}

/// Normalize a key string to `(pitch_class 0..11, is_minor)` for comparison.
/// Accepts "A minor", "A min", "Am", "F# major", "Db", "Gbm", etc.
fn normalize_key(s: &str) -> Option<(u8, bool)> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let lower = s.to_lowercase();
    let is_minor = lower.contains("min") || lower.trim_end().ends_with('m');

    // Parse the leading note token.
    let bytes: Vec<char> = s.chars().collect();
    let mut idx = 0;
    let letter = bytes.get(idx)?.to_ascii_uppercase();
    idx += 1;
    let mut pc: i32 = match letter {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        _ => return None,
    };
    match bytes.get(idx) {
        Some('#') | Some('♯') => {
            pc += 1;
            idx += 1;
        }
        Some('b') | Some('♭') => {
            // 'b' is a flat only when it is the accidental, not the mode 'b'... none.
            pc -= 1;
            idx += 1;
        }
        _ => {}
    }
    let _ = idx;
    Some((((pc % 12) + 12) as u8 % 12, is_minor))
}

fn keys_match(detected: &str, reference: &str) -> bool {
    match (normalize_key(detected), normalize_key(reference)) {
        (Some(d), Some(r)) => d == r,
        _ => false,
    }
}

struct Row {
    path: String,
    bpm_ref: Option<Float>,
    detected_bpm: Option<Float>,
    abs_err: Option<Float>,
    key_ref: Option<String>,
    detected_key: Option<String>,
    key_ok: Option<bool>,
    error: Option<String>,
}

fn main() {
    let args = parse_args();
    let labels = parse_csv(&args.csv);
    if labels.is_empty() {
        fail("no usable rows in CSV");
    }
    let config = config_for(&args.mode);

    println!(
        "Evaluating {} tracks (mode={}, sr={})...\n",
        labels.len(),
        args.mode,
        args.sr
    );

    let paths: Vec<PathBuf> = labels.iter().map(|l| PathBuf::from(&l.path)).collect();
    let path_refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();

    let start = std::time::Instant::now();
    let results = analyze::analyze_batch(&path_refs, args.sr, &config);
    let elapsed = start.elapsed();

    let mut rows = Vec::with_capacity(labels.len());
    for (label, res) in labels.iter().zip(results.iter()) {
        match res {
            Ok(r) => {
                let abs_err = label.bpm_ref.map(|ref_bpm| (r.bpm - ref_bpm).abs());
                let key_ok = match (&label.key_ref, &r.key) {
                    (Some(kr), Some(kd)) => Some(keys_match(kd, kr)),
                    _ => None,
                };
                rows.push(Row {
                    path: label.path.clone(),
                    bpm_ref: label.bpm_ref,
                    detected_bpm: Some(r.bpm),
                    abs_err,
                    key_ref: label.key_ref.clone(),
                    detected_key: r.key.clone(),
                    key_ok,
                    error: None,
                });
            }
            Err(e) => rows.push(Row {
                path: label.path.clone(),
                bpm_ref: label.bpm_ref,
                detected_bpm: None,
                abs_err: None,
                key_ref: label.key_ref.clone(),
                detected_key: None,
                key_ok: None,
                error: Some(e.to_string()),
            }),
        }
    }

    // ---- BPM metrics (over rows that analyzed OK and have a bpm_ref) ----
    let scored: Vec<&Row> = rows
        .iter()
        .filter(|r| r.abs_err.is_some() && r.detected_bpm.is_some())
        .collect();

    let n_err = rows.iter().filter(|r| r.error.is_some()).count();

    println!("=== BPM accuracy ===");
    if scored.is_empty() {
        println!("  (no rows with both a detected BPM and bpm_ref)");
    } else {
        let n = scored.len() as Float;
        let acc_half = scored
            .iter()
            .filter(|r| r.abs_err.unwrap() <= 0.5)
            .count() as Float
            / n;
        let acc_pct = scored
            .iter()
            .filter(|r| r.abs_err.unwrap() <= PCT_TOL * r.bpm_ref.unwrap())
            .count() as Float
            / n;
        let octaves = scored
            .iter()
            .filter(|r| {
                let d = r.detected_bpm.unwrap();
                let rf = r.bpm_ref.unwrap();
                r.abs_err.unwrap() > PCT_TOL * rf && is_octave(d, rf)
            })
            .count();
        let errs: Vec<Float> = scored.iter().map(|r| r.abs_err.unwrap()).collect();
        println!("  tracks scored          : {}", scored.len());
        println!("  accuracy @ +/-0.5 BPM  : {:.1}%", acc_half * 100.0);
        println!("  accuracy @ +/-2%       : {:.1}%", acc_pct * 100.0);
        println!(
            "  octave-error rate      : {:.1}%  ({} tracks)",
            octaves as Float / n * 100.0,
            octaves
        );
        println!("  median abs error       : {:.2} BPM", median(errs.clone()));
        println!("  p95 abs error          : {:.2} BPM", percentile(errs, 95.0));
    }

    // ---- Key metrics ----
    let key_scored: Vec<&Row> = rows.iter().filter(|r| r.key_ok.is_some()).collect();
    if !key_scored.is_empty() {
        let correct = key_scored.iter().filter(|r| r.key_ok == Some(true)).count();
        println!("\n=== Key accuracy ===");
        println!(
            "  key accuracy (exact/enharmonic): {:.1}%  ({}/{})",
            correct as Float / key_scored.len() as Float * 100.0,
            correct,
            key_scored.len()
        );
    }

    // ---- Worst offenders ----
    let mut offenders: Vec<&Row> = scored.clone();
    offenders.sort_by(|a, b| b.abs_err.unwrap().partial_cmp(&a.abs_err.unwrap()).unwrap());
    println!("\n=== Worst offenders (top {} by BPM error) ===", args.top);
    println!(
        "{:>8} {:>8} {:>8} {:>6}  {:<20} {:<20} {}",
        "ref", "detected", "abs_err", "flag", "key_ref", "key_det", "file"
    );
    for r in offenders.iter().take(args.top) {
        let rf = r.bpm_ref.unwrap();
        let d = r.detected_bpm.unwrap();
        let file = Path::new(&r.path)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(&r.path);
        println!(
            "{:>8.1} {:>8.1} {:>8.2} {:>6}  {:<20} {:<20} {}",
            rf,
            d,
            r.abs_err.unwrap(),
            octave_flag(d, rf),
            r.key_ref.as_deref().unwrap_or("-"),
            r.detected_key.as_deref().unwrap_or("-"),
            file,
        );
    }

    // ---- Errors ----
    if n_err > 0 {
        println!("\n=== {} tracks failed to analyze ===", n_err);
        for r in rows.iter().filter(|r| r.error.is_some()) {
            println!("  {}: {}", r.path, r.error.as_deref().unwrap_or("?"));
        }
    }

    println!(
        "\nAnalyzed {} tracks in {:.2}s ({:.0} tracks/sec).",
        labels.len(),
        elapsed.as_secs_f64(),
        labels.len() as f64 / elapsed.as_secs_f64().max(1e-9),
    );
}
