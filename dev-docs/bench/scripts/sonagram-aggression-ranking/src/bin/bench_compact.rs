use std::env;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;

#[derive(Deserialize)]
struct Candidate {
    linear_weight: f64,
    tree_weight: f64,
    harshness_correction: f64,
    linear: Linear,
    tree_ensemble: TreeEnsemble,
}

#[derive(Deserialize)]
struct Linear {
    coefficients: Vec<f64>,
    feature_center: Vec<f64>,
    calibration: Calibration,
}

#[derive(Deserialize)]
struct Calibration {
    slope: f64,
    intercept: f64,
}

#[derive(Deserialize)]
struct TreeEnsemble {
    baseline: f64,
    trees: Vec<Vec<Node>>,
}

#[derive(Deserialize)]
struct Node {
    feature: usize,
    threshold: f64,
    left: usize,
    right: usize,
    value: f64,
    leaf: bool,
}

impl Candidate {
    #[inline]
    fn predict(&self, features: &[f64]) -> f64 {
        let raw = self
            .linear
            .coefficients
            .iter()
            .zip(&self.linear.feature_center)
            .zip(features)
            .map(|((weight, center), value)| weight * (value - center))
            .sum::<f64>();
        let linear = 1.0
            / (1.0
                + (-(self.linear.calibration.slope * raw
                    + self.linear.calibration.intercept))
                    .exp());
        let mut tree = self.tree_ensemble.baseline;
        for nodes in &self.tree_ensemble.trees {
            let mut index = 0;
            loop {
                let node = &nodes[index];
                if node.leaf {
                    tree += node.value;
                    break;
                }
                index = if features[node.feature] <= node.threshold {
                    node.left
                } else {
                    node.right
                };
            }
        }
        (self.linear_weight * linear
            + self.tree_weight * tree.clamp(0.0, 1.0)
            + self.harshness_correction * (features[26] - 0.5))
            .clamp(0.0, 1.0)
    }
}

fn first_features(path: &Path) -> Vec<f64> {
    let text = fs::read_to_string(path).expect("features");
    text.lines()
        .nth(1)
        .expect("feature row")
        .split('\t')
        .skip(2)
        .map(|value| value.parse().expect("number"))
        .collect()
}

fn main() {
    let root = env::args().nth(1).expect("usage: bench_compact <pool>");
    let root = Path::new(&root);
    let candidate: Candidate = serde_json::from_slice(
        &fs::read(root.join("compact_candidate.json")).expect("candidate"),
    )
    .expect("valid candidate");
    let features = first_features(&root.join("evaluation_development_features.tsv"));
    assert_eq!(features.len(), candidate.linear.coefficients.len());
    for _ in 0..100_000 {
        black_box(candidate.predict(black_box(&features)));
    }
    let iterations = 2_000_000_u32;
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..iterations {
        checksum += black_box(candidate.predict(black_box(&features)));
    }
    let elapsed = start.elapsed();
    println!(
        "iterations={iterations} ns_per_inference={:.2} checksum={checksum:.6}",
        elapsed.as_nanos() as f64 / f64::from(iterations),
    );
}
