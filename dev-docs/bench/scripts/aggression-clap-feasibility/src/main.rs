use std::cmp::Ordering;
use std::env;
use std::error::Error;
use std::fs;

use ferricml::data::{DenseMatrix, RegressionTargets};
use ferricml::ensemble::{MaxFeatures, RandomForestRegressor, RandomForestRegressorParams};

#[derive(Clone)]
struct Row {
    id: String,
    group: String,
    target: f32,
    features: Vec<f32>,
}

struct Dataset {
    names: Vec<String>,
    rows: Vec<Row>,
}

struct OpenDataset {
    data: Dataset,
    decisions: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
struct Config {
    trees: usize,
    depth: usize,
    leaf: usize,
    features: MaxFeatures,
}

#[derive(Clone, Copy, Debug)]
struct RegressionMetrics {
    spearman: f64,
    pearson: f64,
    mae: f64,
    min: f32,
    max: f32,
}

#[derive(Clone, Copy, Debug)]
struct PairMetrics {
    decisive_correct: usize,
    decisive_total: usize,
    all_correct: usize,
    pairs: usize,
}

fn parse_features(values: &[&str]) -> Result<Vec<f32>, Box<dyn Error>> {
    values
        .iter()
        .map(|value| value.parse().map_err(Into::into))
        .collect()
}

fn load_pilot(path: &str) -> Result<Dataset, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<_> = lines
        .next()
        .ok_or("missing pilot header")?
        .split('\t')
        .collect();
    if header.len() < 5 || header[..4] != ["sample_id", "group_id", "sha256", "target"] {
        return Err("invalid pilot TSV header".into());
    }
    let names = header[4..]
        .iter()
        .map(|value| (*value).to_owned())
        .collect();
    let mut rows = Vec::new();
    for line in lines {
        let values: Vec<_> = line.split('\t').collect();
        if values.len() != header.len() {
            return Err(format!("invalid pilot TSV row width: {}", values.len()).into());
        }
        rows.push(Row {
            id: values[0].to_owned(),
            group: values[1].to_owned(),
            target: values[3].parse()?,
            features: parse_features(&values[4..])?,
        });
    }
    Ok(Dataset { names, rows })
}

fn load_open(path: &str) -> Result<OpenDataset, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<_> = lines
        .next()
        .ok_or("missing open header")?
        .split('\t')
        .collect();
    if header.len() < 6 || header[..5] != ["pair_id", "source", "sha256", "decision", "target"] {
        return Err("invalid open TSV header".into());
    }
    let names = header[5..]
        .iter()
        .map(|value| (*value).to_owned())
        .collect();
    let mut rows = Vec::new();
    let mut decisions = Vec::new();
    for line in lines {
        let values: Vec<_> = line.split('\t').collect();
        if values.len() != header.len() {
            return Err(format!("invalid open TSV row width: {}", values.len()).into());
        }
        rows.push(Row {
            id: values[0].to_owned(),
            group: values[1].to_owned(),
            target: values[4].parse()?,
            features: parse_features(&values[5..])?,
        });
        decisions.push(values[3].to_owned());
    }
    Ok(OpenDataset {
        data: Dataset { names, rows },
        decisions,
    })
}

fn params(config: Config, seed: u64) -> RandomForestRegressorParams {
    RandomForestRegressorParams::default()
        .with_n_estimators(config.trees)
        .with_max_depth(Some(config.depth))
        .with_min_samples_leaf(config.leaf)
        .with_min_samples_split(config.leaf.saturating_mul(2).max(2))
        .with_max_features(config.features)
        .with_random_state(seed)
}

fn fit_predict(
    rows: &[Row],
    train: &[usize],
    test_rows: &[Row],
    config: Config,
    seed: u64,
    override_targets: Option<&[f32]>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let columns = rows.first().ok_or("empty training data")?.features.len();
    let x = train
        .iter()
        .flat_map(|index| rows[*index].features.iter().copied())
        .collect::<Vec<_>>();
    let y = train
        .iter()
        .map(|index| override_targets.map_or(rows[*index].target, |values| values[*index]))
        .collect();
    let matrix = DenseMatrix::new(x, train.len(), columns)?;
    let targets = RegressionTargets::new(y)?;
    let model = RandomForestRegressor::fit(&matrix.as_view(), &targets, params(config, seed))?;
    test_rows
        .iter()
        .map(|row| {
            model
                .predict_one(&row.features)
                .map(|v| v.clamp(0.0, 1.0))
                .map_err(Into::into)
        })
        .collect()
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn cross_validate(
    rows: &[Row],
    folds: usize,
    config: Config,
    seed: u64,
    override_targets: Option<&[f32]>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut predictions = vec![f32::NAN; rows.len()];
    for fold in 0..folds {
        let train = (0..rows.len())
            .filter(|index| stable_hash(&rows[*index].group) as usize % folds != fold)
            .collect::<Vec<_>>();
        let test = (0..rows.len())
            .filter(|index| stable_hash(&rows[*index].group) as usize % folds == fold)
            .collect::<Vec<_>>();
        let held_out = test
            .iter()
            .map(|index| rows[*index].clone())
            .collect::<Vec<_>>();
        let values = fit_predict(
            rows,
            &train,
            &held_out,
            config,
            seed + fold as u64,
            override_targets,
        )?;
        for (index, value) in test.into_iter().zip(values) {
            predictions[index] = value;
        }
    }
    if predictions.iter().any(|value| !value.is_finite()) {
        return Err("cross-validation left missing predictions".into());
    }
    Ok(predictions)
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
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    for (left, right) in a.iter().zip(b) {
        let da = left - mean_a;
        let db = right - mean_b;
        numerator += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if denom_a == 0.0 || denom_b == 0.0 {
        return 0.0;
    }
    numerator / (denom_a.sqrt() * denom_b.sqrt())
}

fn regression_metrics(rows: &[Row], predictions: &[f32]) -> RegressionMetrics {
    let truth = rows.iter().map(|row| row.target).collect::<Vec<_>>();
    let truth_f64 = truth
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let predicted_f64 = predictions
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    RegressionMetrics {
        spearman: correlation(&ranks(&truth), &ranks(predictions)),
        pearson: correlation(&truth_f64, &predicted_f64),
        mae: truth_f64
            .iter()
            .zip(&predicted_f64)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / truth.len() as f64,
        min: predictions.iter().copied().fold(f32::INFINITY, f32::min),
        max: predictions
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
    }
}

fn pair_metrics(open: &OpenDataset, predictions: &[f32], tie_band: f32) -> PairMetrics {
    let mut pairs = open
        .data
        .rows
        .iter()
        .map(|row| row.id.clone())
        .collect::<Vec<_>>();
    pairs.sort();
    pairs.dedup();
    let mut result = PairMetrics {
        decisive_correct: 0,
        decisive_total: 0,
        all_correct: 0,
        pairs: pairs.len(),
    };
    for pair in pairs {
        let indices = open
            .data
            .rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| (row.id == pair).then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(indices.len(), 2);
        assert_eq!(open.decisions[indices[0]], open.decisions[indices[1]]);
        let truth = match open.decisions[indices[0]].as_str() {
            "left" => 1,
            "right" => -1,
            "tie" => 0,
            value => panic!("unknown decision {value}"),
        };
        let delta = predictions[indices[0]] - predictions[indices[1]];
        let predicted = if delta.abs() < tie_band {
            0
        } else if delta > 0.0 {
            1
        } else {
            -1
        };
        result.all_correct += usize::from(truth == predicted);
        if truth != 0 {
            result.decisive_total += 1;
            result.decisive_correct += usize::from(truth == predicted);
        }
    }
    result
}

fn shuffled_targets(rows: &[Row], seed: u64) -> Vec<f32> {
    let mut values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
    let mut state = seed;
    for index in (1..values.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        values.swap(index, (state as usize) % (index + 1));
    }
    values
}

fn better(candidate: RegressionMetrics, best: RegressionMetrics) -> bool {
    candidate.spearman > best.spearman + 1e-12
        || ((candidate.spearman - best.spearman).abs() <= 1e-12 && candidate.mae < best.mae)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: feasibility <pilot-features.tsv> <open-features.tsv>".into());
    }
    let pilot = load_pilot(&args[1])?;
    let open = load_open(&args[2])?;
    if pilot.rows.len() < 256 || open.data.rows.len() != 48 {
        return Err(format!(
            "expected at least 256 pilot and exactly 48 open rows, got {} and {}",
            pilot.rows.len(),
            open.data.rows.len()
        )
        .into());
    }
    if pilot.names != open.data.names {
        return Err("pilot/open feature schema mismatch".into());
    }

    let mut best: Option<(Config, RegressionMetrics)> = None;
    for trees in [32, 64, 128] {
        for depth in [3, 4, 5, 6, 8, 10] {
            for leaf in [1, 2, 4, 6, 8] {
                for features in [MaxFeatures::All, MaxFeatures::Sqrt] {
                    let config = Config {
                        trees,
                        depth,
                        leaf,
                        features,
                    };
                    let predictions = cross_validate(&pilot.rows, 5, config, 20260722, None)?;
                    let metric = regression_metrics(&pilot.rows, &predictions);
                    if best
                        .as_ref()
                        .is_none_or(|(_, current)| better(metric, *current))
                    {
                        best = Some((config, metric));
                    }
                }
            }
        }
    }
    let (config, pilot_cv) = best.expect("non-empty grid");
    let all = (0..pilot.rows.len()).collect::<Vec<_>>();
    let open_predictions = fit_predict(&pilot.rows, &all, &open.data.rows, config, 20260722, None)?;
    let open_regression = regression_metrics(&open.data.rows, &open_predictions);
    let open_pairs = pair_metrics(&open, &open_predictions, 0.08);

    let shuffled = shuffled_targets(&pilot.rows, 0x5eed);
    let shuffled_cv = cross_validate(&pilot.rows, 5, config, 20260722, Some(&shuffled))?;
    let shuffled_cv_metrics = regression_metrics(&pilot.rows, &shuffled_cv);
    let shuffled_open_predictions = fit_predict(
        &pilot.rows,
        &all,
        &open.data.rows,
        config,
        20260722,
        Some(&shuffled),
    )?;
    let shuffled_open = regression_metrics(&open.data.rows, &shuffled_open_predictions);

    println!(
        "features ({}): {}",
        pilot.names.len(),
        pilot.names.join(",")
    );
    println!("best config (selected on pilot CV only): {config:?}");
    println!("pilot group-CV: {pilot_cv:?}");
    println!("open regression: {open_regression:?}");
    println!("open explicit pairs (tie band 0.08): {open_pairs:?}");
    println!("shuffled-label pilot CV: {shuffled_cv_metrics:?}");
    println!("shuffled-label open: {shuffled_open:?}");
    let dynamic_range = open_regression.max - open_regression.min;
    let go = open_pairs.decisive_total == 17
        && open_pairs.decisive_correct >= 14
        && open_regression.spearman >= 0.70
        && dynamic_range >= 0.55
        && pilot_cv.spearman >= 0.50
        && shuffled_cv_metrics.spearman.abs() < 0.20;
    println!("OPEN FEASIBILITY: {}", if go { "GO" } else { "NO-GO" });
    Ok(())
}
