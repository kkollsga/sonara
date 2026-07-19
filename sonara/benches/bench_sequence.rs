use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;

use sonara::sequence;
use sonara::types::Float;

fn bench_dtw_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtw");

    for size in [100, 500, 1000] {
        let cost =
            Array2::from_shape_fn((size, size), |(i, j)| ((i as Float) - (j as Float)).abs());

        group.bench_with_input(BenchmarkId::new("square", size), &cost, |b, c| {
            b.iter(|| sequence::dtw(c.view(), None).unwrap());
        });
    }

    group.finish();
}

fn bench_viterbi(c: &mut Criterion) {
    let mut group = c.benchmark_group("viterbi");

    for (n_states, n_frames) in [(10, 1000), (50, 1000), (100, 1000), (50, 5000)] {
        let log_prob = Array2::from_shape_fn((n_states, n_frames), |(i, j)| {
            -((i as Float - (j % n_states) as Float).abs() + 1.0).ln()
        });
        let log_trans = Array2::from_shape_fn((n_states, n_states), |(i, j)| {
            if i == j {
                -0.1_f32.ln()
            } else {
                -(n_states as Float).ln()
            }
        });

        group.bench_with_input(
            BenchmarkId::new(format!("{}states", n_states), format!("{}frames", n_frames)),
            &(&log_prob, &log_trans),
            |b, (lp, lt)| {
                b.iter(|| sequence::viterbi(lp.view(), lt.view(), None).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dtw_by_size, bench_viterbi);
criterion_main!(benches);
