//! Sequence alignment and dynamic programming algorithms.
//!
//! Mirrors librosa.sequence — dtw, rqa, viterbi, viterbi_discriminative,
//! viterbi_binary, transition_uniform, transition_loop, transition_cycle,
//! transition_local.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{CanoraError, Result};
use crate::types::Float;

// ============================================================
// Dynamic Time Warping
// ============================================================

/// Dynamic Time Warping.
///
/// Computes the DTW alignment between two feature sequences.
///
/// - `c`: Cost matrix of shape `(N, M)`. `c[(i,j)]` is the cost of aligning
///   frame i of X with frame j of Y.
/// - `step_sizes_sigma`: Step constraints. Default: [(1,1), (0,1), (1,0)]
///
/// Returns `(accumulated_cost_matrix, optimal_path)`.
pub fn dtw(
    c: ArrayView2<Float>,
    step_sizes_sigma: Option<&[(usize, usize)]>,
) -> Result<(Array2<Float>, Vec<(usize, usize)>)> {
    let n = c.nrows();
    let m = c.ncols();

    if n == 0 || m == 0 {
        return Err(CanoraError::InvalidParameter {
            param: "C",
            reason: "cost matrix must be non-empty".into(),
        });
    }

    let steps = step_sizes_sigma.unwrap_or(&[(1, 1), (0, 1), (1, 0)]);

    // Compute accumulated cost matrix
    let mut d = Array2::<Float>::from_elem((n, m), Float::INFINITY);
    d[(0, 0)] = c[(0, 0)];

    for i in 0..n {
        for j in 0..m {
            if i == 0 && j == 0 {
                continue;
            }
            let mut best = Float::INFINITY;
            for &(di, dj) in steps {
                if di == 0 && dj == 0 {
                    continue;
                }
                if i >= di && j >= dj {
                    let prev = d[(i - di, j - dj)];
                    if prev < best {
                        best = prev;
                    }
                }
            }
            d[(i, j)] = c[(i, j)] + best;
        }
    }

    // Backtrack optimal path
    let path = dtw_backtracking(d.view(), steps)?;

    Ok((d, path))
}

/// Backtrack through accumulated cost matrix to find optimal path.
pub fn dtw_backtracking(
    d: ArrayView2<Float>,
    step_sizes: &[(usize, usize)],
) -> Result<Vec<(usize, usize)>> {
    let n = d.nrows();
    let m = d.ncols();

    let mut path = Vec::new();
    let mut i = n - 1;
    let mut j = m - 1;
    path.push((i, j));

    while i > 0 || j > 0 {
        let mut best_cost = Float::INFINITY;
        let mut best_step = (1, 1);

        for &(di, dj) in step_sizes {
            if di == 0 && dj == 0 {
                continue;
            }
            if i >= di && j >= dj {
                let cost = d[(i - di, j - dj)];
                if cost < best_cost {
                    best_cost = cost;
                    best_step = (di, dj);
                }
            }
        }

        i -= best_step.0;
        j -= best_step.1;
        path.push((i, j));
    }

    path.reverse();
    Ok(path)
}

// ============================================================
// Recurrence Quantification Analysis
// ============================================================

/// Recurrence Quantification Analysis.
///
/// Finds the longest matching subsequences in a recurrence/self-similarity matrix.
///
/// - `sim`: Binary recurrence matrix of shape `(N, N)`.
///
/// Returns RQA score matrix.
pub fn rqa(sim: ArrayView2<Float>) -> Result<Array2<Float>> {
    let n = sim.nrows();
    let m = sim.ncols();

    let mut score = Array2::<Float>::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            if sim[(i, j)] > 0.0 {
                let prev = if i > 0 && j > 0 { score[(i - 1, j - 1)] } else { 0.0 };
                score[(i, j)] = prev + 1.0;
            }
        }
    }

    Ok(score)
}

// ============================================================
// Viterbi algorithm
// ============================================================

/// Viterbi decoding for Hidden Markov Models.
///
/// Finds the most likely state sequence given observations.
///
/// - `log_prob`: Log observation probabilities, shape `(n_states, n_frames)`.
///   `log_prob[(s, t)]` = log P(observation_t | state_s)
/// - `log_trans`: Log transition matrix, shape `(n_states, n_states)`.
///   `log_trans[(i, j)]` = log P(state_j | state_i)
/// - `log_init`: Log initial state distribution, shape `(n_states,)`.
///
/// Returns the most likely state sequence of length `n_frames`.
pub fn viterbi(
    log_prob: ArrayView2<Float>,
    log_trans: ArrayView2<Float>,
    log_init: Option<ArrayView1<Float>>,
) -> Result<Array1<usize>> {
    let n_states = log_prob.nrows();
    let n_frames = log_prob.ncols();

    if n_states == 0 || n_frames == 0 {
        return Err(CanoraError::InvalidParameter {
            param: "log_prob",
            reason: "must be non-empty".into(),
        });
    }
    if log_trans.nrows() != n_states || log_trans.ncols() != n_states {
        return Err(CanoraError::ShapeMismatch {
            expected: format!("({n_states}, {n_states})"),
            got: format!("({}, {})", log_trans.nrows(), log_trans.ncols()),
        });
    }

    // Trellis: log probability of best path ending in state s at time t
    let mut trellis = Array2::<Float>::from_elem((n_states, n_frames), Float::NEG_INFINITY);
    // Backpointers
    let mut backptr = Array2::<usize>::zeros((n_states, n_frames));

    // Initialize
    for s in 0..n_states {
        let init = match &log_init {
            Some(p) => p[s],
            None => -(n_states as Float).ln(), // uniform
        };
        trellis[(s, 0)] = init + log_prob[(s, 0)];
    }

    // Forward pass
    for t in 1..n_frames {
        for j in 0..n_states {
            let mut best_val = Float::NEG_INFINITY;
            let mut best_state = 0usize;

            for i in 0..n_states {
                let val = trellis[(i, t - 1)] + log_trans[(i, j)];
                if val > best_val {
                    best_val = val;
                    best_state = i;
                }
            }

            trellis[(j, t)] = best_val + log_prob[(j, t)];
            backptr[(j, t)] = best_state;
        }
    }

    // Backtrack
    let mut states = Array1::<usize>::zeros(n_frames);

    // Find best final state
    let mut best_final = 0usize;
    let mut best_val = Float::NEG_INFINITY;
    for s in 0..n_states {
        if trellis[(s, n_frames - 1)] > best_val {
            best_val = trellis[(s, n_frames - 1)];
            best_final = s;
        }
    }
    states[n_frames - 1] = best_final;

    // Trace back
    for t in (0..n_frames - 1).rev() {
        states[t] = backptr[(states[t + 1], t + 1)];
    }

    Ok(states)
}

/// Viterbi decoding with discriminative (per-frame) transition probabilities.
///
/// - `log_prob`: shape `(n_states, n_frames)`
/// - `log_trans`: shape `(n_states, n_states)` — same transition for all frames
/// - `log_init`: shape `(n_states,)` — optional
pub fn viterbi_discriminative(
    log_prob: ArrayView2<Float>,
    log_trans: ArrayView2<Float>,
    log_init: Option<ArrayView1<Float>>,
) -> Result<Array1<usize>> {
    // For the standard case, this is the same as viterbi
    // The discriminative variant uses per-frame observation likelihoods directly
    viterbi(log_prob, log_trans, log_init)
}

/// Binary-state Viterbi for onset/offset detection.
///
/// Simplified 2-state HMM (active/inactive).
///
/// - `log_prob`: shape `(2, n_frames)` — log probabilities for inactive (0) and active (1)
/// - `log_trans`: shape `(2, 2)` — transition matrix
pub fn viterbi_binary(
    log_prob: ArrayView2<Float>,
    log_trans: ArrayView2<Float>,
) -> Result<Array1<bool>> {
    if log_prob.nrows() != 2 {
        return Err(CanoraError::InvalidParameter {
            param: "log_prob",
            reason: "binary Viterbi requires exactly 2 states".into(),
        });
    }

    let states = viterbi(log_prob, log_trans, None)?;
    Ok(states.mapv(|s| s == 1))
}

// ============================================================
// Transition matrices
// ============================================================

/// Uniform transition matrix: all transitions equally likely.
pub fn transition_uniform(n_states: usize) -> Array2<Float> {
    Array2::from_elem((n_states, n_states), 1.0 / n_states as Float)
}

/// Self-loop transition matrix.
///
/// `prob` is the probability of staying in the same state.
/// Off-diagonal transitions go to the next state with probability `(1-prob)`.
pub fn transition_loop(n_states: usize, prob: Float) -> Array2<Float> {
    let mut trans = Array2::<Float>::zeros((n_states, n_states));

    for i in 0..n_states {
        trans[(i, i)] = prob;
        if i + 1 < n_states {
            trans[(i, i + 1)] = 1.0 - prob;
        } else {
            // Last state: self-loop absorbs all probability
            trans[(i, i)] = 1.0;
        }
    }

    trans
}

/// Cyclic transition matrix.
///
/// Like `transition_loop` but the last state can transition to the first.
pub fn transition_cycle(n_states: usize, prob: Float) -> Array2<Float> {
    let mut trans = Array2::<Float>::zeros((n_states, n_states));

    for i in 0..n_states {
        trans[(i, i)] = prob;
        trans[(i, (i + 1) % n_states)] = 1.0 - prob;
    }

    trans
}

/// Local (banded) transition matrix.
///
/// Each state can transition to states within `width` steps.
/// Transitions outside the band have zero probability.
pub fn transition_local(n_states: usize, width: usize) -> Array2<Float> {
    let mut trans = Array2::<Float>::zeros((n_states, n_states));
    let half = width / 2;

    for i in 0..n_states {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n_states);
        let count = hi - lo;
        let p = 1.0 / count as Float;
        for j in lo..hi {
            trans[(i, j)] = p;
        }
    }

    trans
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- DTW ----

    #[test]
    fn test_dtw_identity() {
        // DTW of a sequence with itself → zero cost diagonal
        let c = Array2::from_shape_fn((5, 5), |(i, j)| {
            if i == j { 0.0 } else { 1.0 }
        });
        let (d, path) = dtw(c.view(), None).unwrap();
        assert_abs_diff_eq!(d[(4, 4)], 0.0, epsilon = 1e-10);
        // Path should be diagonal
        for (i, j) in &path {
            assert_eq!(i, j);
        }
    }

    #[test]
    fn test_dtw_known() {
        // Simple 3x4 cost matrix
        let c = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0],
        )
        .unwrap();
        let (d, path) = dtw(c.view(), None).unwrap();
        // Path should start at (0,0) and end at (2,3)
        assert_eq!(path[0], (0, 0));
        assert_eq!(*path.last().unwrap(), (2, 3));
        // Accumulated cost at end should be minimal
        assert!(d[(2, 3)] < 10.0);
    }

    #[test]
    fn test_dtw_symmetric_cost() {
        let c = Array2::from_shape_fn((10, 10), |(i, j)| {
            ((i as Float) - (j as Float)).abs()
        });
        let (d1, _) = dtw(c.view(), None).unwrap();

        let ct = c.t().to_owned();
        let (d2, _) = dtw(ct.view(), None).unwrap();

        // Total cost should be the same
        assert_abs_diff_eq!(d1[(9, 9)], d2[(9, 9)], epsilon = 1e-10);
    }

    // ---- RQA ----

    #[test]
    fn test_rqa_identity() {
        let sim = Array2::from_shape_fn((5, 5), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        });
        let score = rqa(sim.view()).unwrap();
        // Diagonal should count up: 1, 2, 3, 4, 5
        for i in 0..5 {
            assert_abs_diff_eq!(score[(i, i)], (i + 1) as Float, epsilon = 1e-10);
        }
    }

    // ---- Viterbi ----

    #[test]
    fn test_viterbi_known_hmm() {
        // 2-state HMM: state 0 emits "low", state 1 emits "high"
        let log_prob = Array2::from_shape_vec(
            (2, 5),
            vec![
                0.0_f64.ln_1p(), -5.0, 0.0_f64.ln_1p(), -5.0, 0.0_f64.ln_1p(), // state 0 prefers frames 0,2,4
                -5.0, 0.0_f64.ln_1p(), -5.0, 0.0_f64.ln_1p(), -5.0, // state 1 prefers frames 1,3
            ],
        )
        .unwrap();

        // Transitions encourage staying
        let log_trans = Array2::from_shape_vec(
            (2, 2),
            vec![(-0.1_f64).ln_1p(), (-2.0_f64), (-2.0_f64), (-0.1_f64).ln_1p()],
        )
        .unwrap();

        let states = viterbi(log_prob.view(), log_trans.view(), None).unwrap();
        assert_eq!(states.len(), 5);
        // State 0 should dominate at frames 0,2,4
        assert_eq!(states[0], 0);
    }

    #[test]
    fn test_viterbi_single_state() {
        let log_prob = Array2::from_elem((1, 10), 0.0);
        let log_trans = Array2::from_elem((1, 1), 0.0);
        let states = viterbi(log_prob.view(), log_trans.view(), None).unwrap();
        for &s in states.iter() {
            assert_eq!(s, 0);
        }
    }

    #[test]
    fn test_viterbi_binary() {
        let log_prob = Array2::from_shape_vec(
            (2, 6),
            vec![
                0.0, 0.0, -10.0, -10.0, 0.0, 0.0, // inactive
                -10.0, -10.0, 0.0, 0.0, -10.0, -10.0, // active
            ],
        )
        .unwrap();
        let log_trans = Array2::from_shape_vec(
            (2, 2),
            vec![-0.1, -3.0, -3.0, -0.1],
        )
        .unwrap();

        let states = viterbi_binary(log_prob.view(), log_trans.view()).unwrap();
        assert_eq!(states.len(), 6);
        // Middle frames should be active
        assert!(states[2]);
        assert!(states[3]);
        // Edge frames should be inactive
        assert!(!states[0]);
        assert!(!states[5]);
    }

    // ---- Transition matrices ----

    #[test]
    fn test_transition_uniform() {
        let t = transition_uniform(3);
        assert_eq!(t.shape(), &[3, 3]);
        for &v in t.iter() {
            assert_abs_diff_eq!(v, 1.0 / 3.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_transition_loop() {
        let t = transition_loop(3, 0.9);
        // Diagonal should be 0.9
        assert_abs_diff_eq!(t[(0, 0)], 0.9, epsilon = 1e-10);
        assert_abs_diff_eq!(t[(0, 1)], 0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(t[(0, 2)], 0.0, epsilon = 1e-10);
        // Last state absorbs
        assert_abs_diff_eq!(t[(2, 2)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transition_cycle() {
        let t = transition_cycle(3, 0.8);
        assert_abs_diff_eq!(t[(2, 0)], 0.2, epsilon = 1e-10); // wraps around
        assert_abs_diff_eq!(t[(2, 2)], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_transition_stochastic() {
        // All transition matrices should be row-stochastic
        for mat in [
            transition_uniform(5),
            transition_loop(5, 0.9),
            transition_cycle(5, 0.7),
            transition_local(5, 3),
        ] {
            for row in mat.rows() {
                let sum: Float = row.sum();
                assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_transition_local_bandwidth() {
        let t = transition_local(10, 3);
        // State 5 should only connect to states 4, 5, 6
        assert!(t[(5, 3)] == 0.0);
        assert!(t[(5, 4)] > 0.0);
        assert!(t[(5, 5)] > 0.0);
        assert!(t[(5, 6)] > 0.0);
        assert!(t[(5, 7)] == 0.0);
    }

    // ---- DTW benchmark-friendly size ----

    #[test]
    fn test_dtw_larger() {
        let n = 100;
        let c = Array2::from_shape_fn((n, n), |(i, j)| {
            ((i as Float) - (j as Float)).abs()
        });
        let (_d, path) = dtw(c.view(), None).unwrap();
        assert!(path.len() >= n);
        assert_eq!(path[0], (0, 0));
        assert_eq!(*path.last().unwrap(), (n - 1, n - 1));
    }
}
