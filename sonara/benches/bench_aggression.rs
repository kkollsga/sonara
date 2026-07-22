use std::collections::HashSet;
use std::f32::consts::PI;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;

use sonara::aggression;
use sonara::analyze::{self, AnalysisConfig};
use sonara::similarity::EMBEDDING_DIM;
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

fn bench_score(c: &mut Criterion) {
    let embedding = std::array::from_fn::<_, EMBEDDING_DIM, _>(|index| {
        index as Float / (EMBEDDING_DIM - 1) as Float
    });
    c.bench_function("aggression_score_48d", |b| {
        b.iter(|| aggression::score(black_box(&embedding)).unwrap())
    });
}

fn bench_audio(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggression_audio");
    let embedding = embedding_config();
    let fused = fused_config();
    for seconds in [1, 5, 30] {
        let signal = generate_signal(22_050, seconds);
        group.bench_with_input(
            BenchmarkId::new("embedding", format!("{seconds}s")),
            &signal,
            |b, signal| {
                b.iter(|| analyze::analyze_signal(signal.view(), 22_050, &embedding).unwrap())
            },
        );
        group.bench_with_input(
            BenchmarkId::new("embedding+aggression", format!("{seconds}s")),
            &signal,
            |b, signal| b.iter(|| analyze::analyze_signal(signal.view(), 22_050, &fused).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("score", format!("{seconds}s")),
            &signal,
            |b, signal| b.iter(|| aggression::analyze_signal(signal.view(), 22_050).unwrap()),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_score, bench_audio);
criterion_main!(benches);
