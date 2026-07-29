use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::ensemble::{
    HistGradientBoostingRegressor, HistGradientBoostingRegressorParams,
};
use ferricml::ranking::{
    kendall_tau_b, spearman_correlation, PairIndex, PairOutcome, PairwiseLinearRanker,
    PairwiseLinearRankerParams, PairwiseObservation,
};
use serde::{Deserialize, Serialize};

const WIDTH: usize = 39;
const FOLDS: usize = 5;
const TIE_BAND: f32 = 0.07;
const FEATURE_SCHEMA: [u8; 32] = [
    0x6a, 0xc0, 0x3f, 0x40, 0x12, 0x6b, 0x70, 0x25, 0xfd, 0x28, 0x90, 0xf8, 0xa8, 0xf4,
    0x0c, 0x95, 0x06, 0x53, 0x11, 0x7d, 0xb4, 0x7a, 0xdf, 0x13, 0xb6, 0xe9, 0xac, 0x37,
    0x6d, 0xb4, 0xb7, 0x85,
];

#[derive(Clone)]
struct Track {
    id: String,
    target: f32,
    features: Vec<f32>,
    fold: usize,
}

struct Dataset {
    feature_names: Vec<String>,
    tracks: Vec<Track>,
    by_id: HashMap<String, usize>,
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

#[derive(Clone, Copy, Debug, Serialize)]
struct Calibration {
    slope: f32,
    intercept: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Metrics {
    spearman: f64,
    kendall: f64,
    mae: f64,
    decisive_correct: usize,
    decisive_total: usize,
    hard_correct: usize,
    hard_total: usize,
    tie_correct: usize,
    tie_total: usize,
    score_range: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct HgbConfig {
    learning_rate: f32,
    max_iter: usize,
    max_leaf_nodes: usize,
    min_samples_leaf: usize,
    l2_regularization: f32,
    max_bins: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct CandidateConfig {
    c: f32,
    tie_weight: f32,
    hgb: HgbConfig,
    linear_weight: f32,
    harshness_correction: f32,
    output_scale: f32,
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn read_labels(path: &Path) -> Result<HashMap<String, f32>, Box<dyn Error>> {
    let mut labels = HashMap::new();
    for line in fs::read_to_string(path)?.lines().filter(|line| !line.is_empty()) {
        let row: LabelRow = serde_json::from_str(line)?;
        if !row.insufficient {
            labels.insert(row.sample_id, row.target.ok_or("usable label without target")?);
        }
    }
    Ok(labels)
}

fn read_features(path: &Path) -> Result<(Vec<String>, HashMap<String, Vec<f32>>), Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or("missing feature header")?
        .split('\t')
        .collect::<Vec<_>>();
    if header.len() != WIDTH + 2 || header[0] != "sample_id" || header[1] != "support" {
        return Err("invalid feature header".into());
    }
    let names = header[2..].iter().map(|name| (*name).to_owned()).collect();
    let mut features = HashMap::new();
    for line in lines {
        let columns = line.split('\t').collect::<Vec<_>>();
        if columns.len() != header.len() {
            return Err("invalid feature row width".into());
        }
        features.insert(
            columns[0].to_owned(),
            columns[2..]
                .iter()
                .map(|value| value.parse())
                .collect::<Result<Vec<f32>, _>>()?,
        );
    }
    Ok((names, features))
}

fn read_pairs(path: &Path) -> Result<Vec<PairRow>, Box<dyn Error>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).map_err(Into::into))
        .collect()
}

fn group_key(path: &Path) -> String {
    path.parent()
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

fn load_dataset(
    feature_path: &Path,
    label_path: &Path,
    private_paths: &HashMap<String, String>,
) -> Result<Dataset, Box<dyn Error>> {
    let labels = read_labels(label_path)?;
    let (feature_names, features) = read_features(feature_path)?;
    let mut ids = labels.keys().cloned().collect::<Vec<_>>();
    ids.sort();
    let mut tracks = Vec::with_capacity(ids.len());
    for id in ids {
        let values = features.get(&id).ok_or("label missing feature row")?;
        let path = Path::new(private_paths.get(&id).ok_or("label missing private path")?);
        let target = labels[&id];
        tracks.push(Track {
            id,
            target,
            features: values.clone(),
            fold: stable_hash(&group_key(path)) as usize % FOLDS,
        });
    }
    let by_id = tracks
        .iter()
        .enumerate()
        .map(|(index, track)| (track.id.clone(), index))
        .collect();
    Ok(Dataset {
        feature_names,
        tracks,
        by_id,
    })
}

fn subset(dataset: &Dataset, excluded_fold: Option<usize>) -> Result<(Vec<usize>, DenseMatrix, RegressionTargets), Box<dyn Error>> {
    let indices = dataset
        .tracks
        .iter()
        .enumerate()
        .filter(|(_, track)| excluded_fold != Some(track.fold))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let values = indices
        .iter()
        .flat_map(|&index| dataset.tracks[index].features.iter().copied())
        .collect::<Vec<_>>();
    let targets = indices
        .iter()
        .map(|&index| dataset.tracks[index].target)
        .collect::<Vec<_>>();
    Ok((
        indices,
        DenseMatrix::new(values, targets.len(), WIDTH)?,
        RegressionTargets::new(targets)?,
    ))
}

fn observations(
    dataset: &Dataset,
    pairs: &[PairRow],
    indices: &[usize],
    tie_weight: f32,
) -> Result<Vec<PairwiseObservation>, Box<dyn Error>> {
    let local = indices
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect::<HashMap<_, _>>();
    let mut result = Vec::new();
    for pair in pairs {
        let Some(&left_global) = dataset.by_id.get(&pair.left_id) else { continue };
        let Some(&right_global) = dataset.by_id.get(&pair.right_id) else { continue };
        let (Some(&left), Some(&right)) = (local.get(&left_global), local.get(&right_global)) else {
            continue;
        };
        let outcome = match pair.decision.as_str() {
            "left" => PairOutcome::LeftPreferred,
            "right" => PairOutcome::RightPreferred,
            "tie" => PairOutcome::Tie,
            _ => return Err("invalid pair decision".into()),
        };
        result.push(PairwiseObservation::new(
            PairIndex::new(left, right)?,
            outcome,
            if outcome == PairOutcome::Tie { tie_weight } else { 1.0 },
        )?);
    }
    Ok(result)
}

fn fit_ranker(
    dataset: &Dataset,
    pairs: &[PairRow],
    excluded_fold: Option<usize>,
    c: f32,
    tie_weight: f32,
) -> Result<(Vec<usize>, DenseMatrix, PairwiseLinearRanker, Calibration), Box<dyn Error>> {
    let (indices, matrix, _) = subset(dataset, excluded_fold)?;
    let observations = observations(dataset, pairs, &indices, tie_weight)?;
    let ranker = PairwiseLinearRanker::fit(
        &matrix.as_view(),
        &observations,
        PairwiseLinearRankerParams::default()
            .with_c(c)
            .with_max_iter(200)
            .with_tol(1.0e-6),
    )?;
    let raw = ranker.score_items(&matrix.as_view())?;
    let targets = indices
        .iter()
        .map(|&index| dataset.tracks[index].target)
        .collect::<Vec<_>>();
    let calibration = calibrate(&raw, &targets);
    Ok((indices, matrix, ranker, calibration))
}

fn calibrate(raw: &[f32], targets: &[f32]) -> Calibration {
    let points = raw.iter().copied().zip(targets.iter().copied()).map(|(x, y)| {
        let y = y.clamp(1.0e-4, 1.0 - 1.0e-4);
        (x, (y / (1.0 - y)).ln())
    }).collect::<Vec<_>>();
    let mean_x = points.iter().map(|(x, _)| x).sum::<f32>() / points.len() as f32;
    let mean_y = points.iter().map(|(_, y)| y).sum::<f32>() / points.len() as f32;
    let covariance = points.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum::<f32>();
    let variance = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f32>();
    let slope = (covariance / variance.max(f32::EPSILON)).max(f32::EPSILON);
    Calibration { slope, intercept: mean_y - slope * mean_x }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn calibrated(raw: f32, calibration: Calibration) -> f32 {
    sigmoid(calibration.slope * raw + calibration.intercept)
}

fn hgb_params(config: HgbConfig) -> HistGradientBoostingRegressorParams {
    HistGradientBoostingRegressorParams::default()
        .with_learning_rate(config.learning_rate)
        .with_max_iter(config.max_iter)
        .with_max_leaf_nodes(config.max_leaf_nodes)
        .with_min_samples_leaf(config.min_samples_leaf)
        .with_l2_regularization(config.l2_regularization)
        .with_max_bins(config.max_bins)
}

fn evaluate(dataset: &Dataset, pairs: &[PairRow], scores: &[f32]) -> Result<Metrics, Box<dyn Error>> {
    let truth = dataset.tracks.iter().map(|track| f64::from(track.target)).collect::<Vec<_>>();
    let predicted = scores.iter().map(|&score| f64::from(score)).collect::<Vec<_>>();
    let mae = truth.iter().zip(&predicted).map(|(left, right)| (left - right).abs()).sum::<f64>() / truth.len() as f64;
    let mut metrics = Metrics {
        spearman: spearman_correlation(&truth, &predicted)?,
        kendall: kendall_tau_b(&truth, &predicted)?,
        mae,
        decisive_correct: 0,
        decisive_total: 0,
        hard_correct: 0,
        hard_total: 0,
        tie_correct: 0,
        tie_total: 0,
        score_range: scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - scores.iter().copied().fold(f32::INFINITY, f32::min),
    };
    for pair in pairs {
        let Some(&left) = dataset.by_id.get(&pair.left_id) else { continue };
        let Some(&right) = dataset.by_id.get(&pair.right_id) else { continue };
        let delta = scores[left] - scores[right];
        let prediction = if delta.abs() <= TIE_BAND { "tie" } else if delta > 0.0 { "left" } else { "right" };
        if pair.decision == "tie" {
            metrics.tie_total += 1;
            metrics.tie_correct += usize::from(prediction == "tie");
        } else {
            metrics.decisive_total += 1;
            metrics.decisive_correct += usize::from(prediction == pair.decision);
            if pair.category == "hard" {
                metrics.hard_total += 1;
                metrics.hard_correct += usize::from(prediction == pair.decision);
            }
        }
    }
    Ok(metrics)
}

fn ranker_oof(dataset: &Dataset, pairs: &[PairRow], c: f32, tie_weight: f32) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut scores = vec![f32::NAN; dataset.tracks.len()];
    for fold in 0..FOLDS {
        let (_, _, ranker, calibration) = fit_ranker(dataset, pairs, Some(fold), c, tie_weight)?;
        for (index, track) in dataset.tracks.iter().enumerate().filter(|(_, track)| track.fold == fold) {
            scores[index] = calibrated(ranker.score_one(&track.features)?, calibration);
        }
    }
    Ok(scores)
}

fn hgb_oof(dataset: &Dataset, config: HgbConfig) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut scores = vec![f32::NAN; dataset.tracks.len()];
    for fold in 0..FOLDS {
        let (_, matrix, targets) = subset(dataset, Some(fold))?;
        let model = HistGradientBoostingRegressor::fit(&matrix.as_view(), &targets, hgb_params(config))?;
        for (index, track) in dataset.tracks.iter().enumerate().filter(|(_, track)| track.fold == fold) {
            scores[index] = model.predict_one(&track.features)?;
        }
    }
    Ok(scores)
}

fn blend(
    dataset: &Dataset,
    linear: &[f32],
    hgb: &[f32],
    linear_weight: f32,
    correction: f32,
    output_scale: f32,
) -> Vec<f32> {
    dataset.tracks.iter().enumerate().map(|(index, track)| {
        let blended = linear_weight * linear[index] + (1.0 - linear_weight) * hgb[index];
        (0.5 + output_scale * (blended - 0.5)
            + correction * (track.features[26] - 0.5))
            .clamp(0.0, 1.0)
    }).collect()
}

fn objective(metrics: Metrics) -> f64 {
    let tie_accuracy = metrics.tie_correct as f64 / metrics.tie_total.max(1) as f64;
    metrics.spearman + metrics.kendall - 0.5 * metrics.mae + 0.15 * tie_accuracy
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env::args().nth(1).ok_or("usage: ferricml-aggression-upgrade <pool-dir>")?);
    let private_paths: HashMap<String, String> = serde_json::from_slice(&fs::read(root.join("private_paths.json"))?)?;
    let train = load_dataset(&root.join("fit_train_features.tsv"), &root.join("fit_train_labels.jsonl"), &private_paths)?;
    let development = load_dataset(&root.join("evaluation_development_features.tsv"), &root.join("evaluation_development_labels.jsonl"), &private_paths)?;
    if train.feature_names != development.feature_names || train.feature_names.len() != WIDTH {
        return Err("feature schema mismatch".into());
    }
    let train_pairs = read_pairs(&root.join("train_pairs.jsonl"))?;
    let development_pairs = read_pairs(&root.join("development_pairs.jsonl"))?;

    let mut best_ranker = None;
    for c in [0.01, 0.1, 1.0, 10.0] {
        for tie_weight in [16.0, 32.0, 64.0] {
            let scores = ranker_oof(&train, &train_pairs, c, tie_weight)?;
            let metrics = evaluate(&train, &train_pairs, &scores)?;
            println!("ranker c={c} tie_weight={tie_weight}: {metrics:?}");
            if best_ranker.as_ref().is_none_or(|(_, _, _, current)| objective(metrics) > objective(*current)) {
                best_ranker = Some((c, tie_weight, scores, metrics));
            }
        }
    }
    let (c, tie_weight, linear_oof, ranker_metrics) = best_ranker.ok_or("empty ranker grid")?;
    println!("selected ranker c={c} tie_weight={tie_weight}: {ranker_metrics:?}");
    let (train_indices, train_matrix, ranker, calibration) =
        fit_ranker(&train, &train_pairs, None, c, tie_weight)?;
    debug_assert_eq!(train_indices.len(), train.tracks.len());
    let train_targets = RegressionTargets::new(
        train.tracks.iter().map(|track| track.target).collect(),
    )?;
    let dev_linear = development
        .tracks
        .iter()
        .map(|track| calibrated(ranker.score_one(&track.features).unwrap(), calibration))
        .collect::<Vec<_>>();

    let mut hgb_grid = Vec::new();
    for learning_rate in [0.05, 0.1] {
        for max_iter in [32, 64, 96, 128] {
            for max_leaf_nodes in [7, 15] {
                for min_samples_leaf in [10, 20] {
                    hgb_grid.push(HgbConfig {
                        learning_rate,
                        max_iter,
                        max_leaf_nodes,
                        min_samples_leaf,
                        l2_regularization: 0.1,
                        max_bins: 64,
                    });
                }
            }
        }
    }
    let mut best: Option<(CandidateConfig, Metrics, Metrics)> = None;
    for hgb in hgb_grid {
        let hgb_oof = hgb_oof(&train, hgb)?;
        let full_hgb = HistGradientBoostingRegressor::fit(
            &train_matrix.as_view(),
            &train_targets,
            hgb_params(hgb),
        )?;
        let dev_hgb = development
            .tracks
            .iter()
            .map(|track| full_hgb.predict_one(&track.features).unwrap())
            .collect::<Vec<_>>();
        for linear_weight in [0.0, 0.1, 0.2, 0.3, 0.5] {
            for harshness_correction in [0.0, 0.05, 0.1] {
                for output_scale in [0.75, 0.85, 0.95, 1.0] {
                    let scores = blend(
                        &train,
                        &linear_oof,
                        &hgb_oof,
                        linear_weight,
                        harshness_correction,
                        output_scale,
                    );
                    let metrics = evaluate(&train, &train_pairs, &scores)?;
                    let config = CandidateConfig {
                        c,
                        tie_weight,
                        hgb,
                        linear_weight,
                        harshness_correction,
                        output_scale,
                    };
                    let development_scores = blend(
                        &development,
                        &dev_linear,
                        &dev_hgb,
                        linear_weight,
                        harshness_correction,
                        output_scale,
                    );
                    let development_metrics =
                        evaluate(&development, &development_pairs, &development_scores)?;
                    let development_guard = development_metrics.decisive_correct >= 52
                        && development_metrics.hard_correct >= 20
                        && development_metrics.tie_correct >= 12
                        && development_metrics.spearman >= 0.870_894
                        && development_metrics.mae <= 0.106_6;
                    if development_guard
                        && best.as_ref().is_none_or(|(current_config, current, _)| {
                            let complexity = config.hgb.max_iter * config.hgb.max_leaf_nodes;
                            let current_complexity = current_config.hgb.max_iter
                                * current_config.hgb.max_leaf_nodes;
                            complexity < current_complexity
                                || (complexity == current_complexity
                                    && objective(metrics) > objective(*current))
                        })
                    {
                        best = Some((config, metrics, development_metrics));
                    }
                }
            }
        }
        println!("completed HGB {hgb:?}");
    }
    let (config, cv_metrics, guarded_development_metrics) =
        best.ok_or("no FerricML candidate passed existing development guards")?;
    println!(
        "selected candidate {config:?}: CV {cv_metrics:?}; guarded development {guarded_development_metrics:?}"
    );
    let hgb = HistGradientBoostingRegressor::fit(&train_matrix.as_view(), &train_targets, hgb_params(config.hgb))?;
    let dev_hgb = development.tracks.iter().map(|track| hgb.predict_one(&track.features).unwrap()).collect::<Vec<_>>();
    let development_scores = blend(
        &development,
        &dev_linear,
        &dev_hgb,
        config.linear_weight,
        config.harshness_correction,
        config.output_scale,
    );
    let development_metrics = evaluate(&development, &development_pairs, &development_scores)?;
    println!("development {development_metrics:?}");

    let ranker_artifact = ranker.to_artifact(FEATURE_SCHEMA)?;
    let hgb_artifact = hgb.to_artifact(FEATURE_SCHEMA)?;
    fs::write(root.join("ferric_ranker_candidate.ferricml"), &ranker_artifact)?;
    fs::write(root.join("ferric_hgb_candidate.ferricml"), &hgb_artifact)?;
    fs::write(
        root.join("ferric_candidate.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "feature_names": train.feature_names,
            "feature_schema_sha256": FEATURE_SCHEMA.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
            "config": config,
            "calibration": calibration,
            "cv_metrics": cv_metrics,
            "development_metrics": development_metrics,
            "ranker_artifact_bytes": ranker_artifact.len(),
            "hgb_artifact_bytes": hgb_artifact.len(),
        }))?,
    )?;
    let (_, all_features) = read_features(&root.join("train_features.tsv"))?;
    let mut all_ids = all_features.keys().cloned().collect::<Vec<_>>();
    all_ids.sort();
    let all_scores = all_ids
        .into_iter()
        .map(|id| {
            let features = &all_features[&id];
            let linear = calibrated(ranker.score_one(features).unwrap(), calibration);
            let tree = hgb.predict_one(features).unwrap();
            let score = (0.5
                + config.output_scale
                    * (config.linear_weight * linear
                        + (1.0 - config.linear_weight) * tree
                        - 0.5)
                + config.harshness_correction * (features[26] - 0.5))
                .clamp(0.0, 1.0);
            (id, score)
        })
        .collect::<HashMap<_, _>>();
    fs::write(
        root.join("ferric_all_scores.json"),
        serde_json::to_vec_pretty(&all_scores)?,
    )?;
    Ok(())
}
