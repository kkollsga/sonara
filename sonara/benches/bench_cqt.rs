use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use std::f32::consts::PI;

use sonara::core::constantq;
use sonara::types::Float;

fn generate_signal(n: usize) -> Array1<Float> {
    Array1::from_shape_fn(n, |i| (2.0 * PI * 440.0 * i as Float / 22050.0).sin())
}

fn bench_cqt(c: &mut Criterion) {
    let mut group = c.benchmark_group("cqt");
    group.sample_size(10); // CQT is slow, use fewer samples

    for (label, n_samples) in [("1s", 22050), ("5s", 110250)] {
        let signal = generate_signal(n_samples);
        group.bench_with_input(BenchmarkId::new("84bins", label), &signal, |b, sig| {
            b.iter(|| constantq::cqt(sig.view(), 22050, 512, None, 84, 12, 1.0).unwrap());
        });
    }

    // Different n_bins
    let signal = generate_signal(22050);
    for n_bins in [36, 60, 84] {
        group.bench_with_input(
            BenchmarkId::new("1s", format!("{}bins", n_bins)),
            &signal,
            |b, sig| {
                b.iter(|| constantq::cqt(sig.view(), 22050, 512, None, n_bins, 12, 1.0).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cqt);
criterion_main!(benches);
