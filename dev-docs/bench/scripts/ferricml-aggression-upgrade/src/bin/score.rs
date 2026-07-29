use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use ferricml::ensemble::HistGradientBoostingRegressor;
use ferricml::ranking::PairwiseLinearRanker;

const SCHEMA: [u8; 32] = [
    0x6a, 0xc0, 0x3f, 0x40, 0x12, 0x6b, 0x70, 0x25, 0xfd, 0x28, 0x90, 0xf8, 0xa8, 0xf4,
    0x0c, 0x95, 0x06, 0x53, 0x11, 0x7d, 0xb4, 0x7a, 0xdf, 0x13, 0xb6, 0xe9, 0xac, 0x37,
    0x6d, 0xb4, 0xb7, 0x85,
];

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let pool = PathBuf::from(env::args().nth(1).ok_or("usage: score <pool> <features> <output>")?);
    let features_path = PathBuf::from(env::args().nth(2).ok_or("missing features")?);
    let output_path = PathBuf::from(env::args().nth(3).ok_or("missing output")?);
    let candidate: serde_json::Value = serde_json::from_slice(&fs::read(pool.join("ferric_candidate.json"))?)?;
    let config = &candidate["config"];
    let calibration = &candidate["calibration"];
    let slope = calibration["slope"].as_f64().unwrap() as f32;
    let intercept = calibration["intercept"].as_f64().unwrap() as f32;
    let linear_weight = config["linear_weight"].as_f64().unwrap() as f32;
    let output_scale = config["output_scale"].as_f64().unwrap() as f32;
    let correction = config["harshness_correction"].as_f64().unwrap() as f32;
    let ranker = PairwiseLinearRanker::from_artifact(
        &fs::read(pool.join("ferric_ranker_candidate.ferricml"))?,
        SCHEMA,
    )?;
    let hgb = HistGradientBoostingRegressor::from_artifact(
        &fs::read(pool.join("ferric_hgb_candidate.ferricml"))?,
        SCHEMA,
    )?;
    let text = fs::read_to_string(features_path)?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("missing feature header")?.split('\t').collect::<Vec<_>>();
    if header.len() != 41 {
        return Err("unexpected feature width".into());
    }
    let mut scores = HashMap::new();
    for line in lines {
        let columns = line.split('\t').collect::<Vec<_>>();
        let values = columns[2..].iter().map(|value| value.parse()).collect::<Result<Vec<f32>, _>>()?;
        let linear = sigmoid(slope * ranker.score_one(&values)? + intercept);
        let tree = hgb.predict_one(&values)?;
        let score = (0.5
            + output_scale * (linear_weight * linear + (1.0 - linear_weight) * tree - 0.5)
            + correction * (values[26] - 0.5))
            .clamp(0.0, 1.0);
        scores.insert(columns[0].to_owned(), score);
    }
    fs::write(output_path, serde_json::to_vec_pretty(&scores)?)?;
    Ok(())
}

