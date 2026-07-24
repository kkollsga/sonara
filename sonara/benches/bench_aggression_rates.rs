use std::collections::HashSet;
use std::f32::consts::PI;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;

use sonara::aggression;
use sonara::analyze::{self, AnalysisConfig};
use sonara::types::Float;

fn generate_signal(sample_rate: u32, duration_secs: usize) -> Array1<Float> {
    Array1::from_shape_fn(sample_rate as usize * duration_secs, |index| {
        let time = index as Float / sample_rate as Float;
        0.3 * (2.0 * PI * 220.0 * time).sin()
            + 0.2 * (2.0 * PI * 660.0 * time).sin()
            + 0.1 * (2.0 * PI * 1_100.0 * time).sin()
    })
}

fn embedding_config() -> AnalysisConfig {
    AnalysisConfig {
        features: Some(HashSet::from(["embedding".to_owned()])),
        ..AnalysisConfig::default()
    }
}

fn fused_config() -> AnalysisConfig {
    AnalysisConfig {
        features: Some(HashSet::from([
            "aggression".to_owned(),
            "embedding".to_owned(),
        ])),
        ..AnalysisConfig::default()
    }
}

fn bench_audio_rates(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggression_audio_rates");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    let embedding = embedding_config();
    let fused = fused_config();
    for sample_rate in [22_050, 32_000, 44_100, 48_000] {
        for seconds in [1, 5, 30] {
            let signal = generate_signal(sample_rate, seconds);
            let input = format!("{sample_rate}hz-{seconds}s");
            group.bench_with_input(
                BenchmarkId::new("standalone", &input),
                &signal,
                |b, signal| {
                    b.iter(|| aggression::analyze_signal(signal.view(), sample_rate).unwrap())
                },
            );
            group.bench_with_input(
                BenchmarkId::new("embedding", &input),
                &signal,
                |b, signal| {
                    b.iter(|| {
                        analyze::analyze_signal(signal.view(), sample_rate, &embedding).unwrap()
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("embedding+aggression", &input),
                &signal,
                |b, signal| {
                    b.iter(|| analyze::analyze_signal(signal.view(), sample_rate, &fused).unwrap())
                },
            );
            if sample_rate != aggression::AGGRESSION_SAMPLE_RATE {
                group.bench_with_input(
                    BenchmarkId::new("resample", &input),
                    &signal,
                    |b, signal| {
                        b.iter(|| {
                            sonara::core::audio::resample(
                                signal.view(),
                                sample_rate,
                                aggression::AGGRESSION_SAMPLE_RATE,
                            )
                            .unwrap()
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_audio_rates);
criterion_main!(benches);
