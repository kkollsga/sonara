use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use sonara::aggression::{self, AGGRESSION_FEATURE_COUNT};
use sonara::types::Float;

fn bench_rank_model(c: &mut Criterion) {
    let features = std::array::from_fn::<_, AGGRESSION_FEATURE_COUNT, _>(|index| {
        index as Float / (AGGRESSION_FEATURE_COUNT - 1) as Float
    });
    c.bench_function("aggression_rank_39d", |b| {
        b.iter(|| aggression::benchmark_rank_score(black_box(&features)).unwrap())
    });
}

criterion_group!(benches, bench_rank_model);
criterion_main!(benches);
