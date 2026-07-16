//! Bring-your-own genre classification: a small feed-forward model over the
//! hand-crafted similarity embedding.
//!
//! sonara ships **no** genre model. This module is a *socket*: a user trains a
//! tiny classifier (see the pure-numpy trainer in `python/sonara/genre.py`) over
//! sonara's 48-dimensional similarity embedding
//! ([`crate::similarity::EMBEDDING_DIM`]), exports it as JSON, and hands the path
//! back to `analyze_file` / `analyze_signal` via
//! [`AnalysisConfig::genre_model`](crate::analyze::AnalysisConfig). When a model
//! is set, analysis computes the embedding, runs the classifier, and populates
//! `genre` + `genre_confidence`.
//!
//! ## Model JSON format
//!
//! ```json
//! {
//!   "format_version": 1,
//!   "embedding_version": 2,
//!   "labels": ["rock", "electronic"],
//!   "layers": [
//!     {"weights": [[...],[...]], "bias": [...], "activation": "relu"},
//!     {"weights": [[...]], "bias": [...], "activation": "softmax"}
//!   ]
//! }
//! ```
//!
//! Each layer maps an input vector `x` (length `in_dim`) to
//! `out = activation(W·x + b)`, where `W` is stored **row-major** with `out_dim`
//! rows of `in_dim` columns and `b` has length `out_dim`. Supported activations:
//! `"relu"`, `"softmax"`, `"identity"`. The first layer's `in_dim` must equal
//! [`crate::similarity::EMBEDDING_DIM`] (48); the last layer's activation must be
//! `"softmax"` and its `out_dim` must equal `labels.len()`.
//!
//! ## Versioning
//!
//! [`GENRE_MODEL_FORMAT_VERSION`] identifies the JSON schema above. The model
//! also carries `embedding_version`, which must match
//! [`crate::similarity::SIMILARITY_VERSION`] **at use time** (checked in
//! `analyze`, not at load) — a model file can be inspected even when stale, but
//! classifying on a mismatched embedding layout is silently wrong and therefore
//! refused.

use std::path::Path;

use crate::error::{Result, SonaraError};
use crate::similarity::EMBEDDING_DIM;
use crate::types::Float;

/// Version of the genre-model JSON schema this build understands.
///
/// **Bump rule:** increment ONLY when the on-disk JSON layout changes in a way
/// that is not backward compatible — e.g. renaming/removing a top-level field,
/// changing the meaning of `layers`/`weights`, or adding a new *required* field.
/// Purely additive, optional fields do NOT require a bump. A loaded model whose
/// `format_version` differs from this constant is rejected by [`from_json_str`].
pub const GENRE_MODEL_FORMAT_VERSION: u32 = 1;

/// A layer activation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Softmax,
    Identity,
}

impl Activation {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "relu" => Some(Self::Relu),
            "softmax" => Some(Self::Softmax),
            "identity" => Some(Self::Identity),
            _ => None,
        }
    }
}

/// A single dense layer: `out = activation(W·x + b)`.
///
/// `weights` is row-major, `out_dim` rows of `in_dim` columns; `bias` has length
/// `out_dim`.
#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Vec<Vec<Float>>,
    pub bias: Vec<Float>,
    pub activation: Activation,
}

impl Layer {
    #[inline]
    fn in_dim(&self) -> usize {
        self.weights.first().map(|r| r.len()).unwrap_or(0)
    }
    #[inline]
    fn out_dim(&self) -> usize {
        self.weights.len()
    }

    /// Apply the layer to `x` (length must equal `in_dim`).
    fn forward(&self, x: &[Float]) -> Vec<Float> {
        let mut out: Vec<Float> = self
            .weights
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| {
                let mut acc = b;
                for (w, xi) in row.iter().zip(x.iter()) {
                    acc += w * xi;
                }
                acc
            })
            .collect();
        match self.activation {
            Activation::Relu => {
                for v in out.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            Activation::Identity => {}
            Activation::Softmax => softmax_in_place(&mut out),
        }
        out
    }
}

/// Numerically stable softmax (subtract max before exponentiating).
fn softmax_in_place(v: &mut [Float]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().copied().fold(Float::NEG_INFINITY, Float::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// A loaded, validated genre classifier over the similarity embedding.
#[derive(Debug, Clone)]
pub struct GenreModel {
    pub labels: Vec<String>,
    pub layers: Vec<Layer>,
    /// The `embedding_version` the model was trained against. Compared to
    /// [`crate::similarity::SIMILARITY_VERSION`] at use time (in `analyze`).
    pub embedding_version: u32,
}

impl GenreModel {
    /// Classify an embedding vector, returning `(label, confidence)`.
    ///
    /// `confidence` is the softmax probability of the winning label, in
    /// `(0, 1]`. The embedding length should equal [`EMBEDDING_DIM`]; a shorter
    /// or longer vector is zero-padded / truncated to the first layer's `in_dim`
    /// so inference never panics.
    pub fn predict(&self, embedding: &[Float]) -> (String, Float) {
        // Fit the input to the first layer's in_dim defensively.
        let in_dim = self.layers.first().map(|l| l.in_dim()).unwrap_or(0);
        let mut x: Vec<Float> = vec![0.0; in_dim];
        for (slot, &v) in x.iter_mut().zip(embedding.iter()) {
            *slot = if v.is_finite() { v } else { 0.0 };
        }
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        // Last layer is softmax → argmax gives the predicted label.
        let (idx, &conf) = x
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let label = self
            .labels
            .get(idx)
            .cloned()
            .unwrap_or_else(|| String::from("unknown"));
        (label, conf)
    }
}

/// Parse and validate a genre model from a JSON string.
pub fn from_json_str(s: &str) -> Result<GenreModel> {
    let value = json::parse(s).map_err(|e| SonaraError::ModelError(format!("invalid model JSON: {e}")))?;
    build_model(&value)
}

/// Load and validate a genre model from a JSON file on disk.
pub fn load(path: &Path) -> Result<GenreModel> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| SonaraError::ModelError(format!("could not read model file {}: {e}", path.display())))?;
    from_json_str(&s)
}

// ============================================================
// Validation: JSON value → GenreModel
// ============================================================

fn err(msg: impl Into<String>) -> SonaraError {
    SonaraError::ModelError(msg.into())
}

fn build_model(v: &json::Value) -> Result<GenreModel> {
    let obj = v.as_object().ok_or_else(|| err("model root must be a JSON object"))?;

    let format_version = obj
        .lookup("format_version")
        .and_then(json::Value::as_u32)
        .ok_or_else(|| err("missing/invalid `format_version` (expected integer)"))?;
    if format_version != GENRE_MODEL_FORMAT_VERSION {
        return Err(err(format!(
            "unsupported model format_version {format_version}; this build understands version {GENRE_MODEL_FORMAT_VERSION}"
        )));
    }

    let embedding_version = obj
        .lookup("embedding_version")
        .and_then(json::Value::as_u32)
        .ok_or_else(|| err("missing/invalid `embedding_version` (expected integer)"))?;

    let labels_val = obj.lookup("labels").ok_or_else(|| err("missing `labels`"))?;
    let labels_arr = labels_val.as_array().ok_or_else(|| err("`labels` must be an array"))?;
    if labels_arr.is_empty() {
        return Err(err("`labels` must be non-empty"));
    }
    let labels: Vec<String> = labels_arr
        .iter()
        .map(|l| l.as_str().map(str::to_string).ok_or_else(|| err("every `labels` entry must be a string")))
        .collect::<Result<_>>()?;

    let layers_val = obj.lookup("layers").ok_or_else(|| err("missing `layers`"))?;
    let layers_arr = layers_val.as_array().ok_or_else(|| err("`layers` must be an array"))?;
    if layers_arr.is_empty() {
        return Err(err("`layers` must be non-empty"));
    }

    let mut layers: Vec<Layer> = Vec::with_capacity(layers_arr.len());
    for (li, lv) in layers_arr.iter().enumerate() {
        let lobj = lv.as_object().ok_or_else(|| err(format!("layer {li} must be an object")))?;

        let w_val = lobj.lookup("weights").ok_or_else(|| err(format!("layer {li} missing `weights`")))?;
        let w_rows = w_val.as_array().ok_or_else(|| err(format!("layer {li} `weights` must be an array")))?;
        if w_rows.is_empty() {
            return Err(err(format!("layer {li} `weights` must be non-empty")));
        }
        let mut weights: Vec<Vec<Float>> = Vec::with_capacity(w_rows.len());
        let mut row_len: Option<usize> = None;
        for (ri, rv) in w_rows.iter().enumerate() {
            let cols = rv.as_array().ok_or_else(|| err(format!("layer {li} weights row {ri} must be an array")))?;
            if cols.is_empty() {
                return Err(err(format!("layer {li} weights row {ri} must be non-empty")));
            }
            match row_len {
                None => row_len = Some(cols.len()),
                Some(n) if n != cols.len() => {
                    return Err(err(format!(
                        "layer {li} weight rows are ragged: row {ri} has {} cols, expected {n}",
                        cols.len()
                    )));
                }
                _ => {}
            }
            let row: Vec<Float> = cols
                .iter()
                .map(|c| c.as_f32().ok_or_else(|| err(format!("layer {li} weights must be numbers"))))
                .collect::<Result<_>>()?;
            weights.push(row);
        }

        let b_val = lobj.lookup("bias").ok_or_else(|| err(format!("layer {li} missing `bias`")))?;
        let b_arr = b_val.as_array().ok_or_else(|| err(format!("layer {li} `bias` must be an array")))?;
        let bias: Vec<Float> = b_arr
            .iter()
            .map(|c| c.as_f32().ok_or_else(|| err(format!("layer {li} bias must be numbers"))))
            .collect::<Result<_>>()?;
        if bias.len() != weights.len() {
            return Err(err(format!(
                "layer {li} bias length {} must equal out_dim {} (number of weight rows)",
                bias.len(),
                weights.len()
            )));
        }

        let act_str = lobj
            .lookup("activation")
            .and_then(json::Value::as_str)
            .ok_or_else(|| err(format!("layer {li} missing/invalid `activation` (expected string)")))?;
        let activation = Activation::from_str(act_str)
            .ok_or_else(|| err(format!("layer {li} unsupported activation '{act_str}' (use relu/softmax/identity)")))?;

        layers.push(Layer { weights, bias, activation });
    }

    // First layer must consume the embedding.
    let first_in = layers[0].in_dim();
    if first_in != EMBEDDING_DIM {
        return Err(err(format!(
            "first layer in_dim {first_in} must equal the embedding dimensionality {EMBEDDING_DIM}"
        )));
    }
    // Dimensions chain: out_dim[i] == in_dim[i+1].
    for i in 0..layers.len() - 1 {
        let out = layers[i].out_dim();
        let next_in = layers[i + 1].in_dim();
        if out != next_in {
            return Err(err(format!(
                "layer {i} out_dim {out} does not match layer {} in_dim {next_in}",
                i + 1
            )));
        }
    }
    // Last layer: softmax over exactly `labels.len()` outputs.
    let last = layers.last().unwrap();
    if last.activation != Activation::Softmax {
        return Err(err("last layer activation must be `softmax`"));
    }
    if last.out_dim() != labels.len() {
        return Err(err(format!(
            "last layer out_dim {} must equal labels.len() {}",
            last.out_dim(),
            labels.len()
        )));
    }

    Ok(GenreModel { labels, layers, embedding_version })
}

// ============================================================
// Minimal strict JSON parser (private)
// ============================================================
//
// Hand-rolled recursive-descent parser: sonara keeps zero serde dependency. It
// supports exactly what the model format needs — objects, arrays, strings,
// numbers, booleans and null — with standard string escapes
// (`\" \\ \/ \b \f \n \r \t \uXXXX`). Any malformed input, or trailing garbage
// after the top-level value, is rejected with an error (never a panic).
mod json {
    use crate::types::Float;

    #[derive(Debug, Clone)]
    pub enum Value {
        Null,
        // `null` and `true`/`false` are parsed for JSON completeness (so they are
        // accepted syntactically and rejected by the *type* checks in
        // `build_model`, not by a parse error). The model schema never reads a
        // boolean payload, hence the allow.
        #[allow(dead_code)]
        Bool(bool),
        Num(f64),
        Str(String),
        Array(Vec<Value>),
        Object(Vec<(String, Value)>),
    }

    impl Value {
        pub fn as_object(&self) -> Option<&[(String, Value)]> {
            match self {
                Value::Object(o) => Some(o),
                _ => None,
            }
        }
        pub fn as_array(&self) -> Option<&[Value]> {
            match self {
                Value::Array(a) => Some(a),
                _ => None,
            }
        }
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Value::Str(s) => Some(s),
                _ => None,
            }
        }
        pub fn as_f32(&self) -> Option<Float> {
            match self {
                Value::Num(n) => Some(*n as Float),
                _ => None,
            }
        }
        pub fn as_u32(&self) -> Option<u32> {
            match self {
                // Accept only integral, non-negative numbers in range.
                Value::Num(n) if n.fract() == 0.0 && *n >= 0.0 && *n <= u32::MAX as f64 => Some(*n as u32),
                _ => None,
            }
        }
    }

    /// Convenience: look up a key in an object slice. (Named `lookup`, not
    /// `get`, so it never collides with the inherent slice `get(index)`.)
    pub trait Lookup {
        fn lookup(&self, key: &str) -> Option<&Value>;
    }
    impl Lookup for [(String, Value)] {
        fn lookup(&self, key: &str) -> Option<&Value> {
            self.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }
    }

    struct Parser<'a> {
        b: &'a [u8],
        i: usize,
    }

    pub fn parse(s: &str) -> Result<Value, String> {
        let mut p = Parser { b: s.as_bytes(), i: 0 };
        p.skip_ws();
        let v = p.parse_value()?;
        p.skip_ws();
        if p.i != p.b.len() {
            return Err(format!("trailing characters after JSON value at byte {}", p.i));
        }
        Ok(v)
    }

    impl<'a> Parser<'a> {
        fn peek(&self) -> Option<u8> {
            self.b.get(self.i).copied()
        }

        fn skip_ws(&mut self) {
            while let Some(c) = self.peek() {
                if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                    self.i += 1;
                } else {
                    break;
                }
            }
        }

        fn parse_value(&mut self) -> Result<Value, String> {
            self.skip_ws();
            match self.peek() {
                Some(b'{') => self.parse_object(),
                Some(b'[') => self.parse_array(),
                Some(b'"') => Ok(Value::Str(self.parse_string()?)),
                Some(b't') | Some(b'f') => self.parse_bool(),
                Some(b'n') => self.parse_null(),
                Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
                Some(c) => Err(format!("unexpected character '{}' at byte {}", c as char, self.i)),
                None => Err("unexpected end of input".to_string()),
            }
        }

        fn expect(&mut self, c: u8) -> Result<(), String> {
            if self.peek() == Some(c) {
                self.i += 1;
                Ok(())
            } else {
                Err(format!("expected '{}' at byte {}", c as char, self.i))
            }
        }

        fn parse_object(&mut self) -> Result<Value, String> {
            self.expect(b'{')?;
            let mut out: Vec<(String, Value)> = Vec::new();
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.i += 1;
                return Ok(Value::Object(out));
            }
            loop {
                self.skip_ws();
                if self.peek() != Some(b'"') {
                    return Err(format!("expected object key string at byte {}", self.i));
                }
                let key = self.parse_string()?;
                self.skip_ws();
                self.expect(b':')?;
                let val = self.parse_value()?;
                out.push((key, val));
                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        self.i += 1;
                    }
                    Some(b'}') => {
                        self.i += 1;
                        break;
                    }
                    _ => return Err(format!("expected ',' or '}}' in object at byte {}", self.i)),
                }
            }
            Ok(Value::Object(out))
        }

        fn parse_array(&mut self) -> Result<Value, String> {
            self.expect(b'[')?;
            let mut out: Vec<Value> = Vec::new();
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.i += 1;
                return Ok(Value::Array(out));
            }
            loop {
                let val = self.parse_value()?;
                out.push(val);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        self.i += 1;
                    }
                    Some(b']') => {
                        self.i += 1;
                        break;
                    }
                    _ => return Err(format!("expected ',' or ']' in array at byte {}", self.i)),
                }
            }
            Ok(Value::Array(out))
        }

        fn parse_string(&mut self) -> Result<String, String> {
            self.expect(b'"')?;
            let mut s = String::new();
            loop {
                match self.peek() {
                    None => return Err("unterminated string".to_string()),
                    Some(b'"') => {
                        self.i += 1;
                        break;
                    }
                    Some(b'\\') => {
                        self.i += 1;
                        match self.peek() {
                            Some(b'"') => s.push('"'),
                            Some(b'\\') => s.push('\\'),
                            Some(b'/') => s.push('/'),
                            Some(b'b') => s.push('\u{0008}'),
                            Some(b'f') => s.push('\u{000C}'),
                            Some(b'n') => s.push('\n'),
                            Some(b'r') => s.push('\r'),
                            Some(b't') => s.push('\t'),
                            Some(b'u') => {
                                let cp = self.parse_hex4()?;
                                // Basic BMP handling; surrogate pairs are not
                                // needed by the model format, but decode a lone
                                // code point (reject unpaired surrogates).
                                match char::from_u32(cp as u32) {
                                    Some(ch) => s.push(ch),
                                    None => return Err(format!("invalid \\u escape at byte {}", self.i)),
                                }
                                continue;
                            }
                            _ => return Err(format!("invalid escape at byte {}", self.i)),
                        }
                        self.i += 1;
                    }
                    Some(c) if c < 0x20 => {
                        return Err(format!("control character in string at byte {}", self.i));
                    }
                    Some(_) => {
                        // Copy a full UTF-8 scalar. Determine its byte length.
                        let start = self.i;
                        let len = utf8_len(self.b[start]);
                        if start + len > self.b.len() {
                            return Err("truncated UTF-8 in string".to_string());
                        }
                        match std::str::from_utf8(&self.b[start..start + len]) {
                            Ok(chunk) => s.push_str(chunk),
                            Err(_) => return Err("invalid UTF-8 in string".to_string()),
                        }
                        self.i += len;
                    }
                }
            }
            Ok(s)
        }

        fn parse_hex4(&mut self) -> Result<u16, String> {
            // self.i points at 'u'; consume it and 4 hex digits.
            self.i += 1;
            if self.i + 4 > self.b.len() {
                return Err("truncated \\u escape".to_string());
            }
            let mut cp: u16 = 0;
            for _ in 0..4 {
                let c = self.b[self.i];
                let d = match c {
                    b'0'..=b'9' => c - b'0',
                    b'a'..=b'f' => c - b'a' + 10,
                    b'A'..=b'F' => c - b'A' + 10,
                    _ => return Err(format!("invalid hex digit in \\u escape at byte {}", self.i)),
                };
                cp = cp * 16 + d as u16;
                self.i += 1;
            }
            Ok(cp)
        }

        fn parse_bool(&mut self) -> Result<Value, String> {
            if self.b[self.i..].starts_with(b"true") {
                self.i += 4;
                Ok(Value::Bool(true))
            } else if self.b[self.i..].starts_with(b"false") {
                self.i += 5;
                Ok(Value::Bool(false))
            } else {
                Err(format!("invalid literal at byte {}", self.i))
            }
        }

        fn parse_null(&mut self) -> Result<Value, String> {
            if self.b[self.i..].starts_with(b"null") {
                self.i += 4;
                Ok(Value::Null)
            } else {
                Err(format!("invalid literal at byte {}", self.i))
            }
        }

        fn parse_number(&mut self) -> Result<Value, String> {
            let start = self.i;
            if self.peek() == Some(b'-') {
                self.i += 1;
            }
            let mut saw_digit = false;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    saw_digit = true;
                    self.i += 1;
                } else {
                    break;
                }
            }
            if self.peek() == Some(b'.') {
                self.i += 1;
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        saw_digit = true;
                        self.i += 1;
                    } else {
                        break;
                    }
                }
            }
            if matches!(self.peek(), Some(b'e') | Some(b'E')) {
                self.i += 1;
                if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                    self.i += 1;
                }
                let mut saw_exp = false;
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        saw_exp = true;
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                if !saw_exp {
                    return Err(format!("malformed number exponent at byte {}", self.i));
                }
            }
            if !saw_digit {
                return Err(format!("malformed number at byte {}", start));
            }
            let text = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "invalid number bytes".to_string())?;
            text.parse::<f64>()
                .map(Value::Num)
                .map_err(|_| format!("could not parse number '{text}'"))
        }
    }

    /// Byte length of a UTF-8 scalar given its leading byte.
    fn utf8_len(b: u8) -> usize {
        if b < 0x80 {
            1
        } else if b >> 5 == 0b110 {
            2
        } else if b >> 4 == 0b1110 {
            3
        } else if b >> 3 == 0b11110 {
            4
        } else {
            1
        }
    }
}

use json::Lookup as _;

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_version_pinned() {
        // Bump deliberately (with a changelog note), never accidentally.
        assert_eq!(GENRE_MODEL_FORMAT_VERSION, 1);
    }

    /// A minimal valid 2-class linear (softmax-only) model over 48 dims.
    /// Row 0 (class "a") is all zeros; row 1 (class "b") is all zeros with a
    /// bias favoring "b" — so class "b" always wins regardless of input.
    fn two_class_json(embedding_version: u32) -> String {
        let zeros48 = "0,".repeat(48);
        let zeros48 = zeros48.trim_end_matches(',');
        format!(
            r#"{{
              "format_version": 1,
              "embedding_version": {embedding_version},
              "labels": ["a", "b"],
              "layers": [
                {{"weights": [[{zeros48}],[{zeros48}]], "bias": [0.0, 2.0], "activation": "softmax"}}
              ]
            }}"#
        )
    }

    #[test]
    fn test_roundtrip_valid_model() {
        let m = from_json_str(&two_class_json(crate::similarity::SIMILARITY_VERSION)).unwrap();
        assert_eq!(m.labels, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(m.layers.len(), 1);
        assert_eq!(m.layers[0].in_dim(), EMBEDDING_DIM);
        assert_eq!(m.layers[0].out_dim(), 2);
        assert_eq!(m.layers[0].activation, Activation::Softmax);
        assert_eq!(m.embedding_version, crate::similarity::SIMILARITY_VERSION);
    }

    #[test]
    fn test_predict_golden_two_class() {
        // Zero weights + bias [0, 2] → logits [0, 2] regardless of input.
        // softmax([0, 2]) = [1/(1+e^2), e^2/(1+e^2)] ≈ [0.1192, 0.8808].
        let m = from_json_str(&two_class_json(crate::similarity::SIMILARITY_VERSION)).unwrap();
        let (label, conf) = m.predict(&vec![0.5; EMBEDDING_DIM]);
        assert_eq!(label, "b");
        let expected = (2.0_f32).exp() / (1.0 + (2.0_f32).exp());
        assert!((conf - expected).abs() < 1e-5, "conf {conf} vs {expected}");
        assert!(conf > 0.5 && conf <= 1.0);
        // Input-independent: any embedding gives the same winner + confidence.
        let (l2, c2) = m.predict(&vec![-3.0; EMBEDDING_DIM]);
        assert_eq!(l2, "b");
        assert!((c2 - conf).abs() < 1e-6);
    }

    #[test]
    fn test_predict_two_layer_relu_chain() {
        // 48 -> 2 (relu) -> 2 (softmax). Hidden identity-ish, output favors idx 0.
        let zeros48 = "0,".repeat(48);
        let zeros48 = zeros48.trim_end_matches(',');
        let json = format!(
            r#"{{
              "format_version": 1,
              "embedding_version": {v},
              "labels": ["x", "y"],
              "layers": [
                {{"weights": [[{z}],[{z}]], "bias": [1.0, 0.0], "activation": "relu"}},
                {{"weights": [[3.0, 0.0],[0.0, 0.0]], "bias": [0.0, 0.0], "activation": "softmax"}}
              ]
            }}"#,
            v = crate::similarity::SIMILARITY_VERSION,
            z = zeros48
        );
        let m = from_json_str(&json).unwrap();
        // Hidden = relu([1, 0]) = [1, 0]. Output logits = [3*1, 0] = [3, 0].
        let (label, conf) = m.predict(&vec![0.0; EMBEDDING_DIM]);
        assert_eq!(label, "x");
        let expected = (3.0_f32).exp() / ((3.0_f32).exp() + 1.0);
        assert!((conf - expected).abs() < 1e-5);
    }

    // ---- JSON parser: malformed inputs must Err (never panic) ----

    fn assert_model_err(json: &str) {
        match from_json_str(json) {
            Err(SonaraError::ModelError(_)) => {}
            Err(other) => panic!("expected ModelError, got {other:?}"),
            Ok(_) => panic!("expected error for input: {json}"),
        }
    }

    #[test]
    fn test_reject_truncated() {
        assert_model_err(r#"{"format_version": 1, "labels": ["a"#);
    }

    #[test]
    fn test_reject_trailing_garbage() {
        let mut j = two_class_json(crate::similarity::SIMILARITY_VERSION);
        j.push_str(" extra");
        assert_model_err(&j);
    }

    #[test]
    fn test_reject_wrong_types() {
        // labels as a string, not an array.
        assert_model_err(
            r#"{"format_version": 1, "embedding_version": 2, "labels": "rock", "layers": []}"#,
        );
        // weights entries not numbers.
        assert_model_err(
            r#"{"format_version": 1, "embedding_version": 2, "labels": ["a"],
                "layers": [{"weights": [["x"]], "bias": [0.0], "activation": "softmax"}]}"#,
        );
    }

    #[test]
    fn test_reject_bad_activation() {
        let zeros48 = "0,".repeat(48);
        let zeros48 = zeros48.trim_end_matches(',');
        let json = format!(
            r#"{{"format_version": 1, "embedding_version": {v}, "labels": ["a","b"],
                "layers": [{{"weights": [[{z}],[{z}]], "bias": [0.0,0.0], "activation": "sigmoid"}}]}}"#,
            v = crate::similarity::SIMILARITY_VERSION,
            z = zeros48
        );
        assert_model_err(&json);
    }

    #[test]
    fn test_reject_last_layer_not_softmax() {
        let zeros48 = "0,".repeat(48);
        let zeros48 = zeros48.trim_end_matches(',');
        let json = format!(
            r#"{{"format_version": 1, "embedding_version": {v}, "labels": ["a","b"],
                "layers": [{{"weights": [[{z}],[{z}]], "bias": [0.0,0.0], "activation": "relu"}}]}}"#,
            v = crate::similarity::SIMILARITY_VERSION,
            z = zeros48
        );
        assert_model_err(&json);
    }

    #[test]
    fn test_reject_dim_mismatch_first_layer() {
        // First layer in_dim = 3, not EMBEDDING_DIM.
        assert_model_err(
            r#"{"format_version": 1, "embedding_version": 2, "labels": ["a","b"],
                "layers": [{"weights": [[0.0,0.0,0.0],[0.0,0.0,0.0]], "bias": [0.0,0.0], "activation": "softmax"}]}"#,
        );
    }

    #[test]
    fn test_reject_labels_out_dim_mismatch() {
        let zeros48 = "0,".repeat(48);
        let zeros48 = zeros48.trim_end_matches(',');
        // 3 labels but only 2 output rows.
        let json = format!(
            r#"{{"format_version": 1, "embedding_version": {v}, "labels": ["a","b","c"],
                "layers": [{{"weights": [[{z}],[{z}]], "bias": [0.0,0.0], "activation": "softmax"}}]}}"#,
            v = crate::similarity::SIMILARITY_VERSION,
            z = zeros48
        );
        assert_model_err(&json);
    }

    #[test]
    fn test_reject_empty_labels() {
        assert_model_err(
            r#"{"format_version": 1, "embedding_version": 2, "labels": [],
                "layers": [{"weights": [[0.0]], "bias": [0.0], "activation": "softmax"}]}"#,
        );
    }

    #[test]
    fn test_reject_missing_embedding_version() {
        let zeros48 = "0,".repeat(48);
        let zeros48 = zeros48.trim_end_matches(',');
        let json = format!(
            r#"{{"format_version": 1, "labels": ["a","b"],
                "layers": [{{"weights": [[{z}],[{z}]], "bias": [0.0,0.0], "activation": "softmax"}}]}}"#,
            z = zeros48
        );
        assert_model_err(&json);
    }

    #[test]
    fn test_reject_unsupported_format_version() {
        let zeros48 = "0,".repeat(48);
        let zeros48 = zeros48.trim_end_matches(',');
        let json = format!(
            r#"{{"format_version": 999, "embedding_version": {v}, "labels": ["a","b"],
                "layers": [{{"weights": [[{z}],[{z}]], "bias": [0.0,0.0], "activation": "softmax"}}]}}"#,
            v = crate::similarity::SIMILARITY_VERSION,
            z = zeros48
        );
        assert_model_err(&json);
    }

    #[test]
    fn test_reject_ragged_weights() {
        // Two rows of different length.
        assert_model_err(
            r#"{"format_version": 1, "embedding_version": 2, "labels": ["a","b"],
                "layers": [{"weights": [[0.0,0.0],[0.0]], "bias": [0.0,0.0], "activation": "softmax"}]}"#,
        );
    }

    #[test]
    fn test_reject_bias_length_mismatch() {
        let zeros48 = "0,".repeat(48);
        let zeros48 = zeros48.trim_end_matches(',');
        let json = format!(
            r#"{{"format_version": 1, "embedding_version": {v}, "labels": ["a","b"],
                "layers": [{{"weights": [[{z}],[{z}]], "bias": [0.0], "activation": "softmax"}}]}}"#,
            v = crate::similarity::SIMILARITY_VERSION,
            z = zeros48
        );
        assert_model_err(&json);
    }

    #[test]
    fn test_parser_handles_escapes_and_exponents() {
        // A label with an escaped quote, and exponent-form numbers in weights.
        let mut rows = String::new();
        for r in 0..2 {
            let vals = (0..48)
                .map(|c| if c == 0 && r == 1 { "1e0".to_string() } else { "0.0".to_string() })
                .collect::<Vec<_>>()
                .join(",");
            rows.push_str(&format!("[{vals}]"));
            if r == 0 {
                rows.push(',');
            }
        }
        let json = format!(
            r#"{{"format_version": 1, "embedding_version": {v},
                "labels": ["a\"b", "c/d"],
                "layers": [{{"weights": [{rows}], "bias": [0.0, 2.5], "activation": "softmax"}}]}}"#,
            v = crate::similarity::SIMILARITY_VERSION,
        );
        let m = from_json_str(&json).unwrap();
        assert_eq!(m.labels[0], "a\"b");
        assert_eq!(m.labels[1], "c/d");
    }
}
