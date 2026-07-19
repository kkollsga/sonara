use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use std::f32::consts::PI;

use sonara::core::spectrum;
use sonara::types::{Float, PadMode, WindowSpec};

fn generate_signal(n: usize) -> Array1<Float> {
    Array1::from_shape_fn(n, |i| (2.0 * PI * 440.0 * i as Float / 22050.0).sin())
}

fn default_window() -> WindowSpec {
    WindowSpec::Named("hann".into())
}

fn bench_stft_by_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_by_length");

    for duration_secs in [1, 5, 30] {
        let n_samples = 22050 * duration_secs;
        let signal = generate_signal(n_samples);

        group.bench_with_input(
            BenchmarkId::new("n_fft=2048", format!("{}s", duration_secs)),
            &signal,
            |b, sig| {
                b.iter(|| {
                    spectrum::stft(
                        sig.view(),
                        2048,
                        None,
                        None,
                        &default_window(),
                        true,
                        PadMode::Constant,
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_stft_by_nfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_by_nfft");
    let signal = generate_signal(22050 * 5); // 5 seconds

    for n_fft in [512, 1024, 2048, 4096] {
        group.bench_with_input(
            BenchmarkId::new("5s", format!("n_fft={}", n_fft)),
            &signal,
            |b, sig| {
                b.iter(|| {
                    spectrum::stft(
                        sig.view(),
                        n_fft,
                        None,
                        None,
                        &default_window(),
                        true,
                        PadMode::Constant,
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_istft(c: &mut Criterion) {
    let mut group = c.benchmark_group("istft");

    for duration_secs in [1, 5] {
        let n_samples = 22050 * duration_secs;
        let signal = generate_signal(n_samples);
        let stft_result = spectrum::stft(
            signal.view(),
            2048,
            None,
            None,
            &default_window(),
            true,
            PadMode::Constant,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("n_fft=2048", format!("{}s", duration_secs)),
            &stft_result,
            |b, s| {
                b.iter(|| {
                    spectrum::istft(
                        s.view(),
                        None,
                        None,
                        &default_window(),
                        true,
                        Some(n_samples),
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_power_to_db(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_to_db");

    for n_frames in [100, 500, 1000] {
        let spec = ndarray::Array2::from_shape_fn((1025, n_frames), |(i, j)| {
            ((i + j) as f32 * 0.001).max(1e-10)
        });

        group.bench_with_input(BenchmarkId::new("1025_bins", n_frames), &spec, |b, s| {
            b.iter(|| spectrum::power_to_db(s.view(), 1.0, 1e-10, Some(80.0)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_stft_by_length,
    bench_stft_by_nfft,
    bench_istft,
    bench_power_to_db,
);
criterion_main!(benches);
