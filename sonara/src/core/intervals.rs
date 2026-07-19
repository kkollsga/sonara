//! Frequency interval utilities.
//!
//! Interval frequency generation — interval_frequencies,
//! pythagorean_intervals, plimit_intervals.

use ndarray::Array1;

use crate::types::Float;

/// Generate frequencies from just intonation intervals.
///
/// - `n_bins`: Number of frequency bins
/// - `fmin`: Minimum frequency (Hz)
/// - `intervals`: Frequency ratios relative to the root
/// - `bins_per_octave`: How many intervals per octave
pub fn interval_frequencies(
    n_bins: usize,
    fmin: Float,
    intervals: &[Float],
    bins_per_octave: usize,
) -> Array1<Float> {
    Array1::from_shape_fn(n_bins, |i| {
        let octave = i / bins_per_octave;
        let degree = i % bins_per_octave;
        let interval = if degree < intervals.len() {
            intervals[degree]
        } else {
            1.0
        };
        fmin * 2.0_f32.powi(octave as i32) * interval
    })
}

/// Generate Pythagorean (3-limit) just intonation intervals.
///
/// Returns ratios for one octave, built from the circle of fifths.
pub fn pythagorean_intervals(bins_per_octave: usize) -> Array1<Float> {
    let mut intervals: Vec<Float> = Vec::with_capacity(bins_per_octave);

    for i in 0..bins_per_octave {
        // Position on circle of fifths
        let fifths = if i <= bins_per_octave / 2 {
            i as i32
        } else {
            i as i32 - bins_per_octave as i32
        };

        // Ratio: (3/2)^fifths, folded to [1, 2)
        let mut ratio = (3.0 / 2.0_f32).powi(fifths);
        while ratio >= 2.0 {
            ratio /= 2.0;
        }
        while ratio < 1.0 {
            ratio *= 2.0;
        }
        intervals.push(ratio);
    }

    // Sort by ratio
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Array1::from_vec(intervals)
}

/// Generate p-limit just intonation intervals.
///
/// `p` is the prime limit (e.g., 5 for 5-limit, 7 for 7-limit).
/// Returns intervals sorted within one octave.
pub fn plimit_intervals(p: usize, bins_per_octave: usize) -> Array1<Float> {
    let primes: Vec<usize> = sieve_primes(p);

    // Generate all ratios as products of prime powers
    let max_power: i32 = 6;
    let mut candidates = std::collections::HashSet::new();

    // Exhaustive search: enumerate all combinations of prime powers
    let n_primes = primes.len();
    let powers_range: Vec<i32> = (-max_power..=max_power).collect();

    // For up to 3 primes, enumerate directly
    fn fold_octave(mut v: Float) -> Float {
        while v >= 2.0 {
            v /= 2.0;
        }
        while v < 1.0 {
            v *= 2.0;
        }
        v
    }

    match n_primes {
        0 => {}
        1 => {
            for &a in &powers_range {
                let r = (primes[0] as Float).powi(a);
                let f = fold_octave(r);
                candidates.insert((f * 1e10).round() as i64);
            }
        }
        2 => {
            for &a in &powers_range {
                for &b in &powers_range {
                    let r = (primes[0] as Float).powi(a) * (primes[1] as Float).powi(b);
                    if r > 0.0 && r.is_finite() {
                        let f = fold_octave(r);
                        candidates.insert((f * 1e10).round() as i64);
                    }
                }
            }
        }
        _ => {
            for &a in &powers_range {
                for &b in &powers_range {
                    for &c in &powers_range {
                        let mut r = (primes[0] as Float).powi(a) * (primes[1] as Float).powi(b);
                        if n_primes > 2 {
                            r *= (primes[2] as Float).powi(c);
                        }
                        if r > 0.0 && r.is_finite() && r < 1e12 && r > 1e-12 {
                            let f = fold_octave(r);
                            candidates.insert((f * 1e10).round() as i64);
                        }
                    }
                }
            }
        }
    }

    let mut folded: Vec<Float> = candidates.into_iter().map(|v| v as Float / 1e10).collect();
    folded.sort_by(|a, b| a.partial_cmp(b).unwrap());
    folded.truncate(bins_per_octave);

    Array1::from_vec(folded)
}

fn sieve_primes(limit: usize) -> Vec<usize> {
    let mut primes = Vec::new();
    for n in 2..=limit {
        if primes.iter().all(|&p| n % p != 0) {
            primes.push(n);
        }
    }
    primes
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pythagorean_12() {
        let intervals = pythagorean_intervals(12);
        assert_eq!(intervals.len(), 12);
        // First should be 1.0 (unison)
        assert_abs_diff_eq!(intervals[0], 1.0, epsilon = 1e-5);
        // All should be in [1, 2)
        for &v in intervals.iter() {
            assert!(v >= 1.0 && v < 2.0);
        }
        // Should contain the perfect fifth 3/2 = 1.5
        assert!(intervals.iter().any(|&v| (v - 1.5).abs() < 1e-10));
    }

    #[test]
    fn test_plimit_5() {
        // plimit_intervals(primes=[3, 5], bins_per_octave=7)
        // Expected: [1.0, 1.125, 1.25, 1.333, 1.5, 1.667, 1.875]
        let intervals = plimit_intervals(5, 7);
        assert_eq!(intervals.len(), 7);
        // All should be in [1, 2)
        for &v in intervals.iter() {
            assert!(v >= 0.99 && v < 2.01, "interval {v} out of [1, 2)");
        }
        // First should be 1.0 (unison)
        assert!(
            (intervals[0] - 1.0).abs() < 0.01,
            "first interval should be ~1.0"
        );
    }

    #[test]
    fn test_interval_frequencies() {
        let intervals = vec![1.0, 9.0 / 8.0, 5.0 / 4.0, 4.0 / 3.0, 3.0 / 2.0];
        let freqs = interval_frequencies(10, 100.0, &intervals, 5);
        assert_eq!(freqs.len(), 10);
        assert_abs_diff_eq!(freqs[0], 100.0, epsilon = 1e-5);
        assert_abs_diff_eq!(freqs[4], 150.0, epsilon = 1e-5); // 100 * 3/2
        assert_abs_diff_eq!(freqs[5], 200.0, epsilon = 1e-5); // 100 * 2 * 1
    }
}
