use std::collections::HashSet;
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sonara::analyze::{self, AnalysisConfig, AnalysisMode};
use sonara::types::Float;

const NATIVE_SAMPLE_RATE: u32 = 44_100;

struct TempWave(PathBuf);

impl TempWave {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempWave {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn dense_sample(index: usize, sample_rate: u32) -> Float {
    let time = index as Float / sample_rate as Float;
    let pulse_phase = (time * 3.7).fract();
    let pulse = if pulse_phase < 0.018 {
        (1.0 - pulse_phase / 0.018) * 0.28
    } else {
        0.0
    };
    let partials = [
        (73.0, 0.12),
        (109.0, 0.10),
        (167.0, 0.09),
        (223.0, 0.08),
        (331.0, 0.07),
        (509.0, 0.06),
        (761.0, 0.05),
        (1_153.0, 0.04),
        (1_729.0, 0.035),
        (2_593.0, 0.03),
        (3_887.0, 0.025),
        (5_831.0, 0.02),
    ];
    let harmonic = partials
        .iter()
        .map(|&(frequency, amplitude)| {
            amplitude * (2.0 * PI * frequency * time + 0.3 * (time * 0.71).sin()).sin()
        })
        .sum::<Float>();
    let mixed = index as u32 ^ (index as u32).rotate_left(13);
    let noise =
        (mixed.wrapping_mul(0x9e37_79b9) >> 9) as Float / ((1_u32 << 23) - 1) as Float - 0.5;
    (harmonic + pulse + noise * 0.035).clamp(-0.95, 0.95)
}

fn write_dense_wave(seconds: usize) -> TempWave {
    let path = std::env::temp_dir().join(format!(
        "sonara-aggression-file-bench-{}-{seconds}s.wav",
        std::process::id()
    ));
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: NATIVE_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).expect("create benchmark wave");
    for index in 0..NATIVE_SAMPLE_RATE as usize * seconds {
        let left = dense_sample(index, NATIVE_SAMPLE_RATE);
        let right = dense_sample(index + 97, NATIVE_SAMPLE_RATE);
        writer
            .write_sample((left * i16::MAX as Float) as i16)
            .expect("write left sample");
        writer
            .write_sample((right * i16::MAX as Float) as i16)
            .expect("write right sample");
    }
    writer.finalize().expect("finalize benchmark wave");
    TempWave(path)
}

fn broad_config(with_aggression: bool) -> AnalysisConfig {
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
    AnalysisConfig {
        mode: AnalysisMode::Playlist,
        features: Some(features),
        vocalness_model: Some(Arc::new(
            sonara::vocal_model::bundled().expect("load bundled vocalness model"),
        )),
        ..AnalysisConfig::default()
    }
}

fn bench_broad_file_routes(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggression_file_broad");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    let broad = broad_config(false);
    let aggression = broad_config(true);

    for seconds in [5, 30] {
        let wave = write_dense_wave(seconds);
        for (route, requested_rate) in [("native", 0), ("requested-22050", 22_050)] {
            let input = format!("{route}-{seconds}s");
            group.bench_with_input(BenchmarkId::new("broad", &input), wave.path(), |b, path| {
                b.iter(|| analyze::analyze_file(path, requested_rate, &broad).unwrap())
            });
            group.bench_with_input(
                BenchmarkId::new("broad+aggression", &input),
                wave.path(),
                |b, path| {
                    b.iter(|| analyze::analyze_file(path, requested_rate, &aggression).unwrap())
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_broad_file_routes);
criterion_main!(benches);
