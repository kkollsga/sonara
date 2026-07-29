use std::cmp::Ordering;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

use ferricml::data::{BinaryTargets, DenseMatrix};
use ferricml::linear_model::{LogisticRegression, LogisticRegressionParams};
use serde::Deserialize;
use serde_json::json;

const FEATURE_SCHEMA_SHA256: [u8; 32] = [
    0x6a, 0xc0, 0x3f, 0x40, 0x12, 0x6b, 0x70, 0x25, 0xfd, 0x28, 0x90, 0xf8, 0xa8, 0xf4,
    0x0c, 0x95, 0x06, 0x53, 0x11, 0x7d, 0xb4, 0x7a, 0xdf, 0x13, 0xb6, 0xe9, 0xac, 0x37,
    0x6d, 0xb4, 0xb7, 0x85,
];

#[derive(Clone)]
struct Track {
    id: String,
    target: f32,
    features: Vec<f32>,
}

struct Tracks {
    names: Vec<String>,
    values: HashMap<String, Track>,
}

#[derive(Clone, Deserialize)]
struct LabelRow {
    sample_id: String,
    target: Option<f32>,
    insufficient: bool,
}

#[derive(Clone, Deserialize)]
struct PairRow {
    left_id: String,
    right_id: String,
    decision: String,
    category: String,
}

#[derive(Clone, Copy, Debug)]
struct Calibration {
    slope: f32,
    intercept: f32,
}

#[derive(Clone, Copy, Debug)]
struct PairMetrics {
    decisive_correct: usize,
    decisive_total: usize,
    tie_correct: usize,
    tie_total: usize,
    hard_correct: usize,
    hard_total: usize,
    all_correct: usize,
    all_total: usize,
}

fn read_labels(path: &Path) -> Result<HashMap<String, f32>, Box<dyn Error>> {
    let mut result = HashMap::new();
    for line in fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.is_empty())
    {
        let row: LabelRow = serde_json::from_str(line)?;
        if !row.insufficient {
            result.insert(
                row.sample_id,
                row.target.ok_or("usable label without target")?,
            );
        }
    }
    Ok(result)
}

fn read_features(path: &Path) -> Result<(Vec<String>, HashMap<String, Vec<f32>>), Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or("missing feature header")?
        .split('\t')
        .collect::<Vec<_>>();
    if header.len() < 3 || header[0] != "sample_id" || header[1] != "support" {
        return Err("invalid feature header".into());
    }
    let names = header[2..]
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let mut values = HashMap::new();
    for line in lines {
        let columns = line.split('\t').collect::<Vec<_>>();
        if columns.len() != header.len() {
            return Err("invalid feature row width".into());
        }
        values.insert(
            columns[0].to_owned(),
            columns[2..]
                .iter()
                .map(|value| value.parse())
                .collect::<Result<Vec<f32>, _>>()?,
        );
    }
    Ok((names, values))
}

fn load_tracks(feature_path: &Path, label_path: &Path) -> Result<Tracks, Box<dyn Error>> {
    let labels = read_labels(label_path)?;
    let (names, features) = read_features(feature_path)?;
    let mut values = HashMap::new();
    for (id, target) in labels {
        if let Some(vector) = features.get(&id) {
            values.insert(
                id.clone(),
                Track {
                    id,
                    target,
                    features: vector.clone(),
                },
            );
        }
    }
    Ok(Tracks { names, values })
}

fn read_pairs(path: &Path) -> Result<Vec<PairRow>, Box<dyn Error>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).map_err(Into::into))
        .collect()
}

fn fit_pairwise(
    tracks: &Tracks,
    pairs: &[PairRow],
    c: f32,
    tie_repeats: usize,
    excluded_fold: Option<usize>,
    targets: Option<&HashMap<String, f32>>,
) -> Result<LogisticRegression, Box<dyn Error>> {
    let width = tracks.names.len();
    let mut matrix = Vec::new();
    let mut labels = Vec::new();
    for pair in pairs {
        if excluded_fold
            .is_some_and(|fold| fold_of(&pair.left_id) == fold || fold_of(&pair.right_id) == fold)
        {
            continue;
        }
        let left = &tracks.values[&pair.left_id];
        let right = &tracks.values[&pair.right_id];
        let left_target = targets.map_or(left.target, |values| values[&pair.left_id]);
        let right_target = targets.map_or(right.target, |values| values[&pair.right_id]);
        if (left_target - right_target).abs() <= f32::EPSILON {
            continue;
        }
        if targets.is_none() && pair.decision == "tie" {
            let difference = left
                .features
                .iter()
                .zip(&right.features)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            for _ in 0..tie_repeats {
                for sign in [1.0_f32, -1.0] {
                    matrix.extend(difference.iter().map(|value| sign * value));
                    labels.push(0);
                    matrix.extend(difference.iter().map(|value| sign * value));
                    labels.push(1);
                }
            }
            continue;
        }
        let (high, low) = if left_target > right_target {
            (left, right)
        } else {
            (right, left)
        };
        let difference = high
            .features
            .iter()
            .zip(&low.features)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        matrix.extend_from_slice(&difference);
        labels.push(1);
        matrix.extend(difference.iter().map(|value| -*value));
        labels.push(0);
    }
    let rows = labels.len();
    let x = DenseMatrix::new(matrix, rows, width)?;
    let y = BinaryTargets::new(labels)?;
    Ok(LogisticRegression::fit(
        &x.as_view(),
        &y,
        LogisticRegressionParams::default()
            .with_c(c)
            .with_fit_intercept(false)
            .with_max_iter(100)
            .with_tol(1.0e-6),
    )?)
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn fold_of(id: &str) -> usize {
    stable_hash(id) as usize % 5
}

fn raw(model: &LogisticRegression, features: &[f32], center: Option<&[f32]>) -> f32 {
    model
        .coefficients()
        .iter()
        .zip(features)
        .enumerate()
        .map(|(index, (weight, value))| {
            weight * (value - center.map_or(0.0, |values| values[index]))
        })
        .sum()
}

fn ranks(values: &[f32]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|a, b| {
        values[*a]
            .partial_cmp(&values[*b])
            .unwrap_or(Ordering::Equal)
    });
    let mut result = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end - 1) as f64 / 2.0;
        for index in &order[start..end] {
            result[*index] = rank;
        }
        start = end;
    }
    result
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    let mut numerator = 0.0;
    let mut da2 = 0.0;
    let mut db2 = 0.0;
    for (a, b) in a.iter().zip(b) {
        let da = a - mean_a;
        let db = b - mean_b;
        numerator += da * db;
        da2 += da * da;
        db2 += db * db;
    }
    if da2 == 0.0 || db2 == 0.0 {
        0.0
    } else {
        numerator / (da2.sqrt() * db2.sqrt())
    }
}

fn spearman(truth: &[f32], predicted: &[f32]) -> f64 {
    correlation(&ranks(truth), &ranks(predicted))
}

fn cross_validate(
    tracks: &Tracks,
    pairs: &[PairRow],
    c: f32,
    tie_repeats: usize,
) -> Result<(f64, f64), Box<dyn Error>> {
    let mut correct = 0;
    let mut total = 0;
    let mut truth = Vec::new();
    let mut predicted = Vec::new();
    for fold in 0..5 {
        let model = fit_pairwise(tracks, pairs, c, tie_repeats, Some(fold), None)?;
        for pair in pairs {
            if pair.decision == "tie" {
                continue;
            }
            if fold_of(&pair.left_id) != fold || fold_of(&pair.right_id) != fold {
                continue;
            }
            let left = &tracks.values[&pair.left_id];
            let right = &tracks.values[&pair.right_id];
            correct += usize::from(
                (raw(&model, &left.features, None) > raw(&model, &right.features, None))
                    == (left.target > right.target),
            );
            total += 1;
        }
        let mut ids = tracks
            .values
            .values()
            .filter(|track| fold_of(&track.id) == fold)
            .collect::<Vec<_>>();
        ids.sort_by(|a, b| a.id.cmp(&b.id));
        for track in ids {
            truth.push(track.target);
            predicted.push(raw(&model, &track.features, None));
        }
    }
    Ok((correct as f64 / total as f64, spearman(&truth, &predicted)))
}

fn center(tracks: &Tracks) -> Vec<f32> {
    let mut result = vec![0.0; tracks.names.len()];
    for track in tracks.values.values() {
        for (sum, value) in result.iter_mut().zip(&track.features) {
            *sum += value;
        }
    }
    for value in &mut result {
        *value /= tracks.values.len() as f32;
    }
    result
}

fn calibrate(model: &LogisticRegression, tracks: &Tracks, center: &[f32]) -> Calibration {
    let points = tracks
        .values
        .values()
        .map(|track| {
            let target = track.target.clamp(1.0e-4, 1.0 - 1.0e-4);
            (
                raw(model, &track.features, Some(center)),
                (target / (1.0 - target)).ln(),
            )
        })
        .collect::<Vec<_>>();
    let mean_x = points.iter().map(|(x, _)| x).sum::<f32>() / points.len() as f32;
    let mean_y = points.iter().map(|(_, y)| y).sum::<f32>() / points.len() as f32;
    let covariance = points
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f32>();
    let variance = points
        .iter()
        .map(|(x, _)| (x - mean_x).powi(2))
        .sum::<f32>();
    let slope = (covariance / variance.max(f32::EPSILON)).max(f32::EPSILON);
    Calibration {
        slope,
        intercept: mean_y - slope * mean_x,
    }
}

fn score(
    model: &LogisticRegression,
    features: &[f32],
    center: &[f32],
    calibration: Calibration,
) -> f32 {
    let value = calibration.slope * raw(model, features, Some(center)) + calibration.intercept;
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn pair_metrics(
    pairs: &[PairRow],
    tracks: &Tracks,
    scores: &HashMap<String, f32>,
    tie_band: f32,
) -> PairMetrics {
    let mut metrics = PairMetrics {
        decisive_correct: 0,
        decisive_total: 0,
        tie_correct: 0,
        tie_total: 0,
        hard_correct: 0,
        hard_total: 0,
        all_correct: 0,
        all_total: 0,
    };
    for pair in pairs {
        let delta = scores[&pair.left_id] - scores[&pair.right_id];
        let prediction = if delta.abs() <= tie_band {
            "tie"
        } else if delta > 0.0 {
            "left"
        } else {
            "right"
        };
        let correct = prediction == pair.decision;
        metrics.all_total += 1;
        metrics.all_correct += usize::from(correct);
        if pair.decision == "tie" {
            metrics.tie_total += 1;
            metrics.tie_correct += usize::from(correct);
        } else {
            metrics.decisive_total += 1;
            metrics.decisive_correct += usize::from(correct);
            if pair.category == "hard" {
                metrics.hard_total += 1;
                metrics.hard_correct += usize::from(correct);
            }
        }
        assert!(
            tracks.values.contains_key(&pair.left_id) && tracks.values.contains_key(&pair.right_id)
        );
    }
    metrics
}

fn directional_accuracy(pairs: &[PairRow], scores: &HashMap<String, f32>) -> f64 {
    let mut correct = 0;
    let mut total = 0;
    for pair in pairs.iter().filter(|pair| pair.decision != "tie") {
        let delta = scores[&pair.left_id] - scores[&pair.right_id];
        correct += usize::from((delta > 0.0) == (pair.decision == "left"));
        total += 1;
    }
    correct as f64 / total as f64
}

fn shuffled_targets(tracks: &Tracks, seed: u64) -> HashMap<String, f32> {
    let mut ids = tracks.values.keys().cloned().collect::<Vec<_>>();
    ids.sort();
    let mut values = ids
        .iter()
        .map(|id| tracks.values[id].target)
        .collect::<Vec<_>>();
    let mut state = seed;
    for index in (1..values.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        values.swap(index, state as usize % (index + 1));
    }
    ids.into_iter().zip(values).collect()
}

fn evaluate_scores(tracks: &Tracks, scores: &HashMap<String, f32>) -> (f64, f64, f32) {
    let mut ids = tracks.values.keys().collect::<Vec<_>>();
    ids.sort();
    let truth = ids
        .iter()
        .map(|id| tracks.values[*id].target)
        .collect::<Vec<_>>();
    let predicted = ids.iter().map(|id| scores[*id]).collect::<Vec<_>>();
    let mae = truth
        .iter()
        .zip(&predicted)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / truth.len() as f64;
    let min = predicted.iter().copied().fold(f32::INFINITY, f32::min);
    let max = predicted.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (spearman(&truth, &predicted), mae, max - min)
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = env::args().nth(1).ok_or("usage: feasibility <pool-dir>")?;
    let root = Path::new(&root);
    let tie_repeats = env::var("SONARA_TIE_REPEATS")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(1);
    let train = load_tracks(
        &root.join("fit_train_features.tsv"),
        &root.join("fit_train_labels.jsonl"),
    )?;
    let development = load_tracks(
        &root.join("evaluation_development_features.tsv"),
        &root.join("evaluation_development_labels.jsonl"),
    )?;
    let anchors = load_tracks(
        &root.join("anchor_features.tsv"),
        &root.join("rank_anchor_labels.jsonl"),
    )?;
    if train.names != development.names || train.names != anchors.names || train.names.len() != 39 {
        return Err("feature schema mismatch".into());
    }
    let train_pairs = read_pairs(&root.join("train_pairs.jsonl"))?;
    let development_pairs = read_pairs(&root.join("development_pairs.jsonl"))?;

    let mut best: Option<(f32, f64, f64)> = None;
    for c in [0.01, 0.1, 1.0, 10.0] {
        let (accuracy, rho) = cross_validate(&train, &train_pairs, c, tie_repeats)?;
        println!("pilot CV c={c}: pair_accuracy={accuracy:.6} spearman={rho:.6}");
        if best.is_none_or(|(_, best_accuracy, best_rho)| accuracy + rho > best_accuracy + best_rho)
        {
            best = Some((c, accuracy, rho));
        }
    }
    let (c, cv_accuracy, cv_spearman) = best.expect("non-empty grid");
    let model = fit_pairwise(&train, &train_pairs, c, tie_repeats, None, None)?;
    let feature_center = center(&train);
    let calibration = calibrate(&model, &train, &feature_center);
    let development_scores = development
        .values
        .iter()
        .map(|(id, track)| {
            (
                id.clone(),
                score(&model, &track.features, &feature_center, calibration),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut best_band = None;
    for step in 1..=20 {
        let band = step as f32 / 100.0;
        let metrics = pair_metrics(&development_pairs, &development, &development_scores, band);
        println!("development tie_band={band:.3}: {metrics:?}");
        let gates = usize::from(metrics.decisive_correct >= 52)
            + usize::from(metrics.hard_correct >= 20)
            + usize::from(metrics.tie_correct >= 12);
        let candidate = (gates, metrics.all_correct, metrics.tie_correct);
        if best_band.is_none_or(|(_, current, _)| candidate > current) {
            best_band = Some((band, candidate, metrics));
        }
    }
    let (tie_band, _, metrics) = best_band.expect("tie grid");
    let (dev_spearman, dev_mae, dev_range) = evaluate_scores(&development, &development_scores);

    let mut shuffled_accuracy = 0.0;
    let mut shuffled_rho = 0.0;
    for repeat in 0..10 {
        let shuffled = shuffled_targets(&train, 0x5eed + repeat);
        let shuffled_model =
            fit_pairwise(&train, &train_pairs, c, tie_repeats, None, Some(&shuffled))?;
        let shuffled_tracks = Tracks {
            names: train.names.clone(),
            values: train
                .values
                .iter()
                .map(|(id, track)| {
                    (
                        id.clone(),
                        Track {
                            id: id.clone(),
                            target: shuffled[id],
                            features: track.features.clone(),
                        },
                    )
                })
                .collect(),
        };
        let shuffled_calibration = calibrate(&shuffled_model, &shuffled_tracks, &feature_center);
        let shuffled_scores = development
            .values
            .iter()
            .map(|(id, track)| {
                (
                    id.clone(),
                    score(
                        &shuffled_model,
                        &track.features,
                        &feature_center,
                        shuffled_calibration,
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        shuffled_accuracy += directional_accuracy(&development_pairs, &shuffled_scores);
        shuffled_rho += evaluate_scores(&development, &shuffled_scores).0;
    }
    shuffled_accuracy /= 10.0;
    shuffled_rho /= 10.0;

    let anchor_scores = anchors
        .values
        .iter()
        .map(|(id, track)| {
            (
                id.clone(),
                score(&model, &track.features, &feature_center, calibration),
            )
        })
        .collect::<HashMap<_, _>>();
    let heavy = ["heavy-1", "heavy-2", "heavy-3"];
    let dance = ["dance-1", "dance-2", "dance-3"];
    let mut anchor_correct = 0;
    let mut anchor_min_margin = f32::INFINITY;
    for left in heavy {
        for right in dance {
            let margin = anchor_scores[left] - anchor_scores[right];
            anchor_correct += usize::from(margin > 0.0);
            anchor_min_margin = anchor_min_margin.min(margin);
        }
    }

    println!(
        "selected c={c} tie_repeats={tie_repeats} cv_pair_accuracy={cv_accuracy:.6} cv_spearman={cv_spearman:.6}"
    );
    println!("calibration={calibration:?} tie_band={tie_band:.3}");
    println!("development pairs={metrics:?}");
    println!("development spearman={dev_spearman:.6} mae={dev_mae:.6} range={dev_range:.6}");
    println!("shuffled decisive_accuracy={shuffled_accuracy:.6} spearman={shuffled_rho:.6}");
    println!("anchors correct={anchor_correct}/9 min_margin={anchor_min_margin:.6}");
    let go = metrics.decisive_correct >= 52
        && metrics.hard_correct >= 20
        && metrics.tie_correct >= 12
        && dev_spearman >= 0.65
        && dev_mae <= 0.15
        && dev_range >= 0.65
        && (0.45..=0.55).contains(&shuffled_accuracy)
        && shuffled_rho.abs() <= 0.10
        && anchor_correct == 9
        && anchor_min_margin >= 0.15;
    println!(
        "SONAGRAM PAIRWISE FEASIBILITY: {}",
        if go { "GO" } else { "NO-GO" }
    );

    let output = json!({
        "feature_names": train.names,
        "coefficients": model.coefficients(),
        "feature_center": feature_center,
        "calibration": {"slope": calibration.slope, "intercept": calibration.intercept},
        "regularization_c": c,
        "tie_repeats": tie_repeats,
        "tie_band": tie_band,
        "metrics": {
            "cv_pair_accuracy": cv_accuracy, "cv_spearman": cv_spearman,
            "development_decisive_correct": metrics.decisive_correct,
            "development_decisive_total": metrics.decisive_total,
            "development_hard_correct": metrics.hard_correct,
            "development_hard_total": metrics.hard_total,
            "development_tie_correct": metrics.tie_correct,
            "development_tie_total": metrics.tie_total,
            "development_spearman": dev_spearman, "development_mae": dev_mae,
            "development_range": dev_range, "shuffled_accuracy": shuffled_accuracy,
            "shuffled_spearman": shuffled_rho, "anchor_correct": anchor_correct,
            "anchor_min_margin": anchor_min_margin,
        },
        "go": go,
    });
    fs::write(
        root.join("candidate.json"),
        serde_json::to_vec_pretty(&output)?,
    )?;
    fs::write(
        root.join("linear_candidate.ferricml"),
        model.to_artifact(FEATURE_SCHEMA_SHA256)?,
    )?;
    Ok(())
}
