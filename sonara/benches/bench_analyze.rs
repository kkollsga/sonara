use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use std::f32::consts::PI;

use sonara::analyze;
use sonara::types::Float;
use std::collections::HashSet;

fn generate_signal(sr: u32, duration_secs: usize) -> Array1<Float> {
    let n = sr as usize * duration_secs;
    Array1::from_shape_fn(n, |i| {
        let t = i as Float / sr as Float;
        0.3 * (2.0 * PI * 440.0 * t).sin()
            + 0.2 * (2.0 * PI * 554.37 * t).sin()
            + 0.15 * (2.0 * PI * 659.25 * t).sin()
            + 0.1 * (2.0 * PI * 220.0 * t).sin()
            + 0.05 * (2.0 * PI * 110.0 * t).sin()
    })
}

fn bench_compact(c: &mut Criterion) {
    let mut group = c.benchmark_group("analyze_compact");
    let cfg = analyze::compact();

    for secs in [1, 5, 30] {
        let signal = generate_signal(22050, secs);
        group.bench_with_input(
            BenchmarkId::new("compact", format!("{}s", secs)),
            &signal,
            |b, sig| {
                b.iter(|| analyze::analyze_signal(sig.view(), 22050, &cfg).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_playlist(c: &mut Criterion) {
    let mut group = c.benchmark_group("analyze_playlist");
    let cfg = analyze::playlist();

    for secs in [1, 5, 30] {
        let signal = generate_signal(22050, secs);
        group.bench_with_input(
            BenchmarkId::new("playlist", format!("{}s", secs)),
            &signal,
            |b, sig| {
                b.iter(|| analyze::analyze_signal(sig.view(), 22050, &cfg).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("analyze_full");
    let cfg = analyze::full();

    for secs in [1, 5, 30] {
        let signal = generate_signal(22050, secs);
        group.bench_with_input(
            BenchmarkId::new("full", format!("{}s", secs)),
            &signal,
            |b, sig| {
                b.iter(|| analyze::analyze_signal(sig.view(), 22050, &cfg).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_all_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("analyze_all_modes");
    let signal = generate_signal(22050, 10);

    group.bench_with_input(BenchmarkId::new("mode", "compact"), &signal, |b, sig| {
        b.iter(|| analyze::analyze_signal(sig.view(), 22050, &analyze::compact()).unwrap());
    });

    group.bench_with_input(BenchmarkId::new("mode", "playlist"), &signal, |b, sig| {
        b.iter(|| analyze::analyze_signal(sig.view(), 22050, &analyze::playlist()).unwrap());
    });

    group.bench_with_input(BenchmarkId::new("mode", "full"), &signal, |b, sig| {
        b.iter(|| analyze::analyze_signal(sig.view(), 22050, &analyze::full()).unwrap());
    });

    group.finish();
}

fn bench_mood_opt_in(c: &mut Criterion) {
    let mut group = c.benchmark_group("analyze_mood_opt_in");
    let cfg = analyze::AnalysisConfig {
        features: Some(HashSet::from(["mood".to_string()])),
        ..analyze::compact()
    };

    for secs in [1, 5, 30] {
        let signal = generate_signal(22050, secs);
        group.bench_with_input(
            BenchmarkId::new("mood", format!("{}s", secs)),
            &signal,
            |b, sig| {
                b.iter(|| analyze::analyze_signal(sig.view(), 22050, &cfg).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compact,
    bench_playlist,
    bench_full,
    bench_all_modes,
    bench_mood_opt_in
);
criterion_main!(benches);
