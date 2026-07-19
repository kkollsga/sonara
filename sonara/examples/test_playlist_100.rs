use std::io::BufRead;
use std::path::Path;
use std::time::Instant;

use sonara::analyze;

fn main() {
    let file_list = std::fs::File::open("/tmp/sonara_test_200.txt").expect("open file list");
    let paths: Vec<String> = std::io::BufReader::new(file_list)
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.is_empty())
        .collect();

    println!("Analyzing {} files in playlist mode...\n", paths.len());

    let config = analyze::playlist();
    let path_refs: Vec<&Path> = paths.iter().map(|p| Path::new(p.as_str())).collect();

    let start = Instant::now();
    let results = analyze::analyze_batch(&path_refs, 22050, &config);
    let elapsed = start.elapsed();

    let mut ok_count = 0usize;
    let mut err_count = 0usize;

    println!(
        "{:<5} {:>6} {:>7} {:>7} {:>5} {:>5} {:>5} {:>5} {:>12} {:>6}  {}",
        "#", "BPM", "LUFS", "Energy", "Dance", "Val", "Acou", "DynRg", "Key", "Dur", "File"
    );
    println!("{}", "-".repeat(110));

    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(r) => {
                ok_count += 1;
                let filename = Path::new(&paths[i])
                    .file_name()
                    .and_then(|f| f.to_str())
                    .unwrap_or("?");
                let filename_short = if filename.len() > 35 {
                    &filename[..35]
                } else {
                    filename
                };
                println!("{:<5} {:>6.1} {:>7.1} {:>7.3} {:>5.2} {:>5.2} {:>5.2} {:>5.1} {:>12} {:>5.0}s  {}",
                    i + 1,
                    r.bpm,
                    r.loudness_lufs,
                    r.energy.unwrap_or(0.0),
                    r.danceability.unwrap_or(0.0),
                    r.valence.unwrap_or(0.0),
                    r.acousticness.unwrap_or(0.0),
                    r.dynamic_range_db,
                    r.key.as_deref().unwrap_or("?"),
                    r.duration_sec,
                    filename_short,
                );
            }
            Err(e) => {
                err_count += 1;
                eprintln!("  ERROR [{}]: {}", i + 1, e);
            }
        }
    }

    println!("\n{}", "=".repeat(110));
    println!(
        "Results: {} OK, {} errors out of {} files",
        ok_count,
        err_count,
        paths.len()
    );
    println!(
        "Total time: {:.2}s ({:.1}ms per file, {:.0} files/sec)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / paths.len() as f64,
        paths.len() as f64 / elapsed.as_secs_f64(),
    );

    // Print feature statistics
    if ok_count > 0 {
        let ok_results: Vec<&analyze::TrackAnalysis> =
            results.iter().filter_map(|r| r.as_ref().ok()).collect();

        println!("\nFeature Statistics (n={}):", ok_count);
        println!(
            "  BPM:          min={:.0}  avg={:.0}  max={:.0}",
            ok_results
                .iter()
                .map(|r| r.bpm)
                .fold(f32::INFINITY, f32::min),
            ok_results.iter().map(|r| r.bpm).sum::<f32>() / ok_count as f32,
            ok_results
                .iter()
                .map(|r| r.bpm)
                .fold(f32::NEG_INFINITY, f32::max)
        );
        println!(
            "  LUFS:         min={:.1}  avg={:.1}  max={:.1}",
            ok_results
                .iter()
                .map(|r| r.loudness_lufs)
                .fold(f32::INFINITY, f32::min),
            ok_results.iter().map(|r| r.loudness_lufs).sum::<f32>() / ok_count as f32,
            ok_results
                .iter()
                .map(|r| r.loudness_lufs)
                .fold(f32::NEG_INFINITY, f32::max)
        );
        println!(
            "  Energy:       min={:.2}  avg={:.2}  max={:.2}",
            ok_results
                .iter()
                .filter_map(|r| r.energy)
                .fold(f32::INFINITY, f32::min),
            ok_results.iter().filter_map(|r| r.energy).sum::<f32>() / ok_count as f32,
            ok_results
                .iter()
                .filter_map(|r| r.energy)
                .fold(f32::NEG_INFINITY, f32::max)
        );
        println!(
            "  Danceability: min={:.2}  avg={:.2}  max={:.2}",
            ok_results
                .iter()
                .filter_map(|r| r.danceability)
                .fold(f32::INFINITY, f32::min),
            ok_results
                .iter()
                .filter_map(|r| r.danceability)
                .sum::<f32>()
                / ok_count as f32,
            ok_results
                .iter()
                .filter_map(|r| r.danceability)
                .fold(f32::NEG_INFINITY, f32::max)
        );
        println!(
            "  Valence:      min={:.2}  avg={:.2}  max={:.2}",
            ok_results
                .iter()
                .filter_map(|r| r.valence)
                .fold(f32::INFINITY, f32::min),
            ok_results.iter().filter_map(|r| r.valence).sum::<f32>() / ok_count as f32,
            ok_results
                .iter()
                .filter_map(|r| r.valence)
                .fold(f32::NEG_INFINITY, f32::max)
        );
        println!(
            "  Acousticness: min={:.2}  avg={:.2}  max={:.2}",
            ok_results
                .iter()
                .filter_map(|r| r.acousticness)
                .fold(f32::INFINITY, f32::min),
            ok_results
                .iter()
                .filter_map(|r| r.acousticness)
                .sum::<f32>()
                / ok_count as f32,
            ok_results
                .iter()
                .filter_map(|r| r.acousticness)
                .fold(f32::NEG_INFINITY, f32::max)
        );

        // Key distribution
        let mut key_counts = std::collections::HashMap::new();
        for r in &ok_results {
            if let Some(ref k) = r.key {
                *key_counts.entry(k.clone()).or_insert(0usize) += 1;
            }
        }
        let mut key_vec: Vec<_> = key_counts.into_iter().collect();
        key_vec.sort_by(|a, b| b.1.cmp(&a.1));
        print!("  Keys:         ");
        for (k, c) in key_vec.iter().take(8) {
            print!("{}: {}  ", k, c);
        }
        println!();
    }
}
