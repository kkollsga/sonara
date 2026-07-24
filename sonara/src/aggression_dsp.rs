//! Canonical-rate DSP used only by the bundled aggression model.
//!
//! This deliberately mirrors the arithmetic of the generic fused analyzer for
//! model inputs while omitting public products the caller will discard.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use ndarray::{s, Array1, Array2, ArrayView1};
use rayon::prelude::*;

use super::{
    aggression_evenly_sample, aggression_grid_regularity, aggression_interval_cv,
    aggression_quantile, aggression_quantile_sorted, aggression_sorted,
    aggression_window_summaries, HOP_LENGTH, N_CONTRAST_BANDS, PARALLEL_THRESHOLD,
};
use crate::aggression::{AggressionAnalysis, AggressionEvidence, AGGRESSION_SAMPLE_RATE};
use crate::core::{convert, fft, spectrum};
use crate::dsp::windows;
use crate::error::{Result, SonaraError};
use crate::filters;
use crate::perceptual;
use crate::types::{Float, WindowSpec};
use crate::util::utils;

const N_FFT: usize = 2048;
const N_MELS: usize = 128;
const N_BINS: usize = N_FFT / 2 + 1;
const MAX_PEAKS: usize = 50;
const EVIDENCE_CONTRAST_BANDS: usize = 5;

struct AggressionCache {
    sparse_mel: Vec<(usize, Vec<Float>)>,
    freqs: Array1<Float>,
    win_padded: Array1<Float>,
    dct_rows: [Vec<Float>; 2],
    contrast_bands: [(usize, usize); EVIDENCE_CONTRAST_BANDS],
}

static AGGRESSION_CACHE: OnceLock<AggressionCache> = OnceLock::new();
static ACTIVE_ANALYSES: AtomicUsize = AtomicUsize::new(0);

struct ActiveAnalysis;

impl ActiveAnalysis {
    fn enter() -> (Self, bool) {
        let was_idle = ACTIVE_ANALYSES.fetch_add(1, Ordering::AcqRel) == 0;
        (Self, was_idle)
    }
}

impl Drop for ActiveAnalysis {
    fn drop(&mut self) {
        ACTIVE_ANALYSES.fetch_sub(1, Ordering::AcqRel);
    }
}

struct AggressionScratch {
    fft_input: Vec<Float>,
    fft_output: Vec<num_complex::Complex<Float>>,
    power: Vec<Float>,
    magnitude: Vec<Float>,
    band_values: Vec<Float>,
    peak_frequencies: Vec<Float>,
    peak_magnitudes: Vec<Float>,
    peak_indices: Vec<usize>,
}

impl AggressionScratch {
    fn new() -> Self {
        Self {
            fft_input: vec![0.0; N_FFT],
            fft_output: vec![num_complex::Complex::new(0.0, 0.0); N_BINS],
            power: vec![0.0; N_BINS],
            magnitude: vec![0.0; N_BINS],
            band_values: Vec::with_capacity(N_BINS),
            peak_frequencies: Vec::with_capacity(MAX_PEAKS * 2),
            peak_magnitudes: Vec::with_capacity(MAX_PEAKS * 2),
            peak_indices: Vec::with_capacity(MAX_PEAKS * 2),
        }
    }
}

thread_local! {
    static AGGRESSION_SCRATCH: RefCell<AggressionScratch> = RefCell::new(AggressionScratch::new());
}

struct AggressionFrame {
    mel: [Float; N_MELS],
    centroid: Float,
    rms: Float,
    bandwidth: Float,
    contrast: [Float; EVIDENCE_CONTRAST_BANDS],
    dissonance: Float,
    crest_db: Float,
    high_energy_ratio: Float,
    high_flatness: Float,
    high_total: Float,
    peak_ratio: Float,
}

fn cache() -> &'static AggressionCache {
    AGGRESSION_CACHE.get_or_init(|| {
        let sample_rate = AGGRESSION_SAMPLE_RATE as Float;
        let mel = filters::mel(
            sample_rate,
            N_FFT,
            N_MELS,
            0.0,
            sample_rate / 2.0,
            false,
            "slaney",
        );
        let sparse_mel = (0..N_MELS)
            .map(|index| {
                let row = mel.row(index);
                let first = row.iter().position(|&value| value > 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&value| value > 0.0).unwrap_or(0);
                if first > last {
                    (0, Vec::new())
                } else {
                    (first, row.slice(s![first..=last]).to_vec())
                }
            })
            .collect();
        let freqs = convert::fft_frequencies(sample_rate, N_FFT);
        let window = windows::get_window(&WindowSpec::Named("hann".into()), N_FFT, true)
            .expect("hann window");
        let win_padded = utils::pad_center(window.view(), N_FFT).expect("pad_center");
        let dct_rows = [0, 2].map(|coefficient| {
            (0..N_MELS)
                .map(|mel_index| {
                    let norm = if coefficient == 0 {
                        (1.0 / N_MELS as Float).sqrt()
                    } else {
                        (2.0 / N_MELS as Float).sqrt()
                    };
                    norm * (std::f32::consts::PI
                        * coefficient as Float
                        * (2 * mel_index + 1) as Float
                        / (2.0 * N_MELS as Float))
                        .cos()
                })
                .collect()
        });

        let fmin: Float = 200.0;
        let fmax = sample_rate / 2.0;
        let mut edges = [0.0; N_CONTRAST_BANDS + 1];
        edges[0] = fmin;
        for (index, edge) in edges.iter_mut().enumerate().skip(1) {
            *edge = fmin * (fmax / fmin).powf(index as Float / N_CONTRAST_BANDS as Float);
        }
        let frequencies = freqs.as_slice().unwrap();
        let contrast_bands = std::array::from_fn(|index| {
            let start = frequencies
                .iter()
                .position(|&frequency| frequency >= edges[index])
                .unwrap_or(0);
            let end = frequencies
                .iter()
                .position(|&frequency| frequency >= edges[index + 1])
                .unwrap_or(N_BINS);
            (start, end)
        });

        AggressionCache {
            sparse_mel,
            freqs,
            win_padded,
            dct_rows,
            contrast_bands,
        }
    })
}

pub(super) fn analyze_signal(y: ArrayView1<'_, Float>) -> Result<AggressionAnalysis> {
    let (_active, may_parallelize) = ActiveAnalysis::enter();
    let cache = cache();
    let sample_rate = AGGRESSION_SAMPLE_RATE;
    let sample_rate_float = sample_rate as Float;
    let duration_sec = y.len() as Float / sample_rate_float;

    let pad = N_FFT / 2;
    let mut padded = Array1::<Float>::zeros(y.len() + 2 * pad);
    padded.slice_mut(s![pad..pad + y.len()]).assign(&y);
    if padded.len() < N_FFT {
        return Err(SonaraError::InsufficientData {
            needed: N_FFT,
            got: padded.len(),
        });
    }
    let samples = padded.as_slice().unwrap();
    let window = cache.win_padded.as_slice().unwrap();
    let frequencies = cache.freqs.as_slice().unwrap();
    let frame_count = 1 + (padded.len() - N_FFT) / HOP_LENGTH;

    let compute_frame = |frame_index: usize, scratch: &mut AggressionScratch| {
        let start = frame_index * HOP_LENGTH;
        for index in 0..N_FFT {
            scratch.fft_input[index] = samples[start + index] * window[index];
        }
        fft::rfft(&mut scratch.fft_input, &mut scratch.fft_output).expect("FFT failed");

        let mut centroid_numerator = 0.0;
        let mut magnitude_sum = 0.0;
        for index in 0..N_BINS {
            let bin_power = scratch.fft_output[index].norm_sqr();
            let bin_magnitude = bin_power.sqrt();
            scratch.power[index] = bin_power;
            scratch.magnitude[index] = bin_magnitude;
            centroid_numerator += frequencies[index] * bin_magnitude;
            magnitude_sum += bin_magnitude;
        }
        let centroid = if magnitude_sum > 0.0 {
            centroid_numerator / magnitude_sum
        } else {
            0.0
        };

        let mut sum_squared = 0.0;
        for index in 0..N_FFT {
            sum_squared += samples[start + index] * samples[start + index];
        }
        let rms = (sum_squared / N_FFT as Float).sqrt();
        const EPSILON: Float = 1.0e-12;
        let peak = samples[start..start + N_FFT]
            .iter()
            .copied()
            .map(Float::abs)
            .fold(0.0, Float::max);
        let crest_db = 20.0 * ((peak + EPSILON) / (rms + EPSILON)).log10();

        let bandwidth = if magnitude_sum > 0.0 {
            let mut numerator = 0.0;
            for index in 0..N_BINS {
                let deviation = frequencies[index] - centroid;
                numerator += scratch.magnitude[index] * deviation * deviation;
            }
            (numerator / magnitude_sum).sqrt()
        } else {
            0.0
        };

        let minimum_power: Float = 1.0e-10;
        let mut total = 0.0;
        let mut high_total = 0.0;
        let mut high_log_sum = 0.0;
        let mut high_arithmetic_sum = 0.0;
        let mut high_count = 0_usize;
        let mut strongest = [0.0; 8];
        for index in 0..N_BINS {
            let value = scratch.power[index].max(minimum_power);
            let log_value = value.ln();
            let bin_power = scratch.power[index];
            total += bin_power;
            if frequencies[index] >= 4_000.0 {
                high_total += bin_power;
                high_log_sum += log_value;
                high_arithmetic_sum += value;
                high_count += 1;
            }
            if bin_power > strongest[0] {
                strongest[0] = bin_power;
                strongest.sort_by(Float::total_cmp);
            }
        }
        let high_mean = high_arithmetic_sum / high_count.max(1) as Float;
        let high_energy_ratio = high_total / (total + EPSILON);
        let high_flatness = if high_mean > 0.0 {
            (high_log_sum / high_count.max(1) as Float).exp() / high_mean
        } else {
            0.0
        };
        let peak_ratio = strongest.iter().sum::<Float>() / (total + EPSILON);

        let mel = std::array::from_fn(|mel_index| {
            let (start_bin, weights) = &cache.sparse_mel[mel_index];
            let mut sum = 0.0;
            for (offset, &weight) in weights.iter().enumerate() {
                sum += weight * scratch.power[start_bin + offset];
            }
            sum
        });

        let mut contrast = [0.0; EVIDENCE_CONTRAST_BANDS];
        for (band, &(start_bin, end_bin)) in cache.contrast_bands.iter().enumerate() {
            if start_bin >= end_bin {
                continue;
            }
            let length = end_bin - start_bin;
            scratch.band_values.clear();
            scratch
                .band_values
                .extend((start_bin..end_bin).map(|index| scratch.magnitude[index].max(1.0e-10)));
            let quantile_index = ((length as Float * 0.02) as usize).min(length - 1);
            scratch
                .band_values
                .select_nth_unstable_by(quantile_index, Float::total_cmp);
            let valley = scratch.band_values[quantile_index];
            let peak_index = (length - 1).saturating_sub(quantile_index);
            scratch
                .band_values
                .select_nth_unstable_by(peak_index, Float::total_cmp);
            contrast[band] = scratch.band_values[peak_index].log10() - valley.log10();
        }

        scratch.peak_frequencies.clear();
        scratch.peak_magnitudes.clear();
        for index in 1..N_BINS - 1 {
            if scratch.magnitude[index] <= scratch.magnitude[index - 1]
                || scratch.magnitude[index] <= scratch.magnitude[index + 1]
            {
                continue;
            }
            if frequencies[index] < 40.0 || frequencies[index] > 5_000.0 {
                continue;
            }
            let alpha = scratch.magnitude[index - 1];
            let beta = scratch.magnitude[index];
            let gamma = scratch.magnitude[index + 1];
            let denominator = alpha - 2.0 * beta + gamma;
            let (frequency, peak_magnitude) = if denominator.abs() > 1.0e-10 {
                let interpolation = 0.5 * (alpha - gamma) / denominator;
                let fractional_bin = index as Float + interpolation;
                let frequency = if fractional_bin >= 0.0 && (fractional_bin as usize) < N_BINS - 1 {
                    let lower = fractional_bin as usize;
                    let fraction = fractional_bin - lower as Float;
                    frequencies[lower] * (1.0 - fraction) + frequencies[lower + 1] * fraction
                } else {
                    frequencies[index]
                };
                (frequency, beta - 0.25 * (alpha - gamma) * interpolation)
            } else {
                (frequencies[index], beta)
            };
            scratch.peak_frequencies.push(frequency);
            scratch.peak_magnitudes.push(peak_magnitude);
        }
        scratch.peak_indices.clear();
        scratch
            .peak_indices
            .extend(0..scratch.peak_frequencies.len());
        scratch.peak_indices.sort_unstable_by(|&left, &right| {
            scratch.peak_magnitudes[right].total_cmp(&scratch.peak_magnitudes[left])
        });
        scratch.peak_indices.truncate(MAX_PEAKS);

        let mut dissonance = 0.0;
        if scratch.peak_indices.len() >= 2 {
            let mut dissonance_sum = 0.0;
            let mut weight_sum = 0.0;
            for left in 0..scratch.peak_indices.len() {
                for right in left + 1..scratch.peak_indices.len() {
                    let left_index = scratch.peak_indices[left];
                    let right_index = scratch.peak_indices[right];
                    let left_frequency = scratch.peak_frequencies[left_index];
                    let right_frequency = scratch.peak_frequencies[right_index];
                    let minimum_frequency = left_frequency.min(right_frequency);
                    let frequency_difference = (left_frequency - right_frequency).abs();
                    let scale = 0.24 / (0.0207 * minimum_frequency + 18.96);
                    let roughness = (-3.5144 * scale * frequency_difference).exp()
                        - (-5.7564 * scale * frequency_difference).exp();
                    let weight =
                        scratch.peak_magnitudes[left_index] * scratch.peak_magnitudes[right_index];
                    dissonance_sum += weight * roughness.max(0.0);
                    weight_sum += weight;
                }
            }
            if weight_sum > 0.0 {
                dissonance = (dissonance_sum / weight_sum).clamp(0.0, 1.0);
            }
        }

        AggressionFrame {
            mel,
            centroid,
            rms,
            bandwidth,
            contrast,
            dissonance,
            crest_db,
            high_energy_ratio,
            high_flatness,
            high_total,
            peak_ratio,
        }
    };

    let frames = if may_parallelize && frame_count >= PARALLEL_THRESHOLD {
        (0..frame_count)
            .into_par_iter()
            .map(|frame_index| {
                AGGRESSION_SCRATCH
                    .with(|scratch| compute_frame(frame_index, &mut scratch.borrow_mut()))
            })
            .collect::<Vec<_>>()
    } else {
        (0..frame_count)
            .map(|frame_index| {
                AGGRESSION_SCRATCH
                    .with(|scratch| compute_frame(frame_index, &mut scratch.borrow_mut()))
            })
            .collect::<Vec<_>>()
    };

    let mut mel_spectrogram = Array2::<Float>::zeros((N_MELS, frame_count));
    let mut centroids = vec![0.0; frame_count];
    let mut rms_frames = vec![0.0; frame_count];
    let mut bandwidths = vec![0.0; frame_count];
    let mut contrast_sum = [0.0; EVIDENCE_CONTRAST_BANDS];
    let mut dissonance = vec![0.0; frame_count];
    let mut crest = vec![0.0; frame_count];
    let mut high_energy = vec![0.0; frame_count];
    let mut high_flatness = vec![0.0; frame_count];
    let mut high_total = vec![0.0; frame_count];
    let mut peak_ratio = vec![0.0; frame_count];
    for (frame_index, frame) in frames.into_iter().enumerate() {
        centroids[frame_index] = frame.centroid;
        rms_frames[frame_index] = frame.rms;
        bandwidths[frame_index] = frame.bandwidth;
        dissonance[frame_index] = frame.dissonance;
        crest[frame_index] = frame.crest_db;
        high_energy[frame_index] = frame.high_energy_ratio;
        high_flatness[frame_index] = frame.high_flatness;
        high_total[frame_index] = frame.high_total;
        peak_ratio[frame_index] = frame.peak_ratio;
        for band in 0..EVIDENCE_CONTRAST_BANDS {
            contrast_sum[band] += frame.contrast[band];
        }
        for (mel_index, value) in frame.mel.into_iter().enumerate() {
            mel_spectrogram[(mel_index, frame_index)] = value;
        }
    }

    let spectrogram_db = spectrum::power_to_db(mel_spectrogram.view(), 1.0, 1.0e-10, Some(80.0));
    let onset_frame_count = frame_count.saturating_sub(1);
    let mut onset_envelope = Array1::<Float>::zeros(onset_frame_count);
    for frame_index in 0..onset_frame_count {
        let mut sum = 0.0;
        for mel_index in 0..N_MELS {
            sum += (spectrogram_db[(mel_index, frame_index + 1)]
                - spectrogram_db[(mel_index, frame_index)])
                .max(0.0);
        }
        onset_envelope[frame_index] = sum / N_MELS as Float;
    }
    let left_padding = 1 + N_FFT / (2 * HOP_LENGTH);
    let mut padded_onset = Array1::<Float>::zeros(onset_frame_count + left_padding);
    for frame_index in 0..onset_frame_count {
        padded_onset[left_padding + frame_index] = onset_envelope[frame_index];
    }

    let (tempo, beats) = crate::beat::beat_track_detailed(
        None,
        Some(padded_onset.view()),
        sample_rate,
        HOP_LENGTH,
        120.0,
        100.0,
        true,
        None,
        None,
    )?;
    let bpm = tempo.tempo;
    let onset_frames = crate::onset::onset_detect(
        None,
        Some(padded_onset.view()),
        sample_rate,
        HOP_LENGTH,
        false,
        0.07,
        0,
    )?;

    let mut mfcc = [0.0; 2];
    for frame_index in 0..frame_count {
        for (target, row) in mfcc.iter_mut().zip(&cache.dct_rows) {
            let mut sum = 0.0;
            for mel_index in 0..N_MELS {
                sum += row[mel_index] * spectrogram_db[(mel_index, frame_index)];
            }
            *target += sum;
        }
    }
    for value in &mut mfcc {
        *value /= frame_count.max(1) as Float;
    }
    for value in &mut contrast_sum {
        *value /= frame_count.max(1) as Float;
    }

    let rms_mean = rms_frames.iter().sum::<Float>() / rms_frames.len() as Float;
    let nonzero_rms = rms_frames
        .iter()
        .copied()
        .filter(|&value| value > 1.0e-10)
        .collect::<Vec<_>>();
    let dynamic_range_db = if nonzero_rms.len() > 10 {
        let mut sorted = nonzero_rms;
        sorted.sort_by(Float::total_cmp);
        let low = sorted[sorted.len() * 5 / 100];
        let high = sorted[sorted.len() * 95 / 100];
        if low > 0.0 {
            20.0 * (high / low).log10()
        } else {
            0.0
        }
    } else {
        0.0
    };
    let centroid_mean = centroids.iter().sum::<Float>() / centroids.len().max(1) as Float;
    let bandwidth_mean = bandwidths.iter().sum::<Float>() / bandwidths.len().max(1) as Float;
    let onset_density_embedding = onset_frames.len() as Float / duration_sec;
    let energy = perceptual::energy(
        rms_mean,
        centroid_mean,
        onset_density_embedding,
        bandwidth_mean,
    );
    let danceability = perceptual::danceability_heuristic(bpm, &beats, onset_density_embedding);

    let onset_p90 = aggression_quantile(padded_onset.as_slice().unwrap(), 0.90);
    let onset_normalized = padded_onset
        .iter()
        .map(|value| (value / onset_p90.max(1.0e-12)).clamp(0.0, 4.0))
        .collect::<Vec<_>>();
    let onset_sorted = aggression_sorted(&onset_normalized);
    let onset_p50 = aggression_quantile_sorted(&onset_sorted, 0.50);
    let onset_threshold = (onset_p50 + 0.25).max(0.30);
    let aggression_onsets = onset_normalized
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value >= onset_threshold).then_some(index))
        .collect::<Vec<_>>();
    let aggression_onset_density = aggression_onsets.len() as Float / duration_sec;
    let high_flux = high_total
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).max(0.0) / pair[1].max(1.0e-12))
        .collect::<Vec<_>>();
    let rms_sorted = aggression_sorted(&rms_frames);
    let rms_p10 = aggression_quantile_sorted(&rms_sorted, 0.10);
    let rms_p90 = aggression_quantile_sorted(&rms_sorted, 0.90);
    let signal_rms = (y.iter().map(|value| value * value).sum::<Float>() / y.len() as Float).sqrt();
    let non_silent_threshold = 0.10 * rms_p90.max(1.0e-12);
    let non_silent = rms_frames
        .iter()
        .filter(|value| **value >= non_silent_threshold)
        .count() as Float
        / rms_frames.len().max(1) as Float;
    let peak_sorted = aggression_sorted(&peak_ratio);
    let peak_p50 = aggression_quantile_sorted(&peak_sorted, 0.50);
    let high_flatness_sorted = aggression_sorted(&high_flatness);
    let high_flatness_p50 = aggression_quantile_sorted(&high_flatness_sorted, 0.50);
    let content_support = if signal_rms <= 1.0e-6 {
        0.0
    } else {
        non_silent * (0.5 * peak_p50 + 0.5 * (1.0 - high_flatness_p50))
    }
    .clamp(0.0, 1.0);
    let (window_force_top2, window_harshness_top2, window_impact_persistence, window_impact_top2) =
        aggression_window_summaries(
            &crest,
            &high_energy,
            &high_flatness,
            &peak_ratio,
            &onset_normalized,
            sample_rate,
            HOP_LENGTH,
        );
    let crest_sorted = aggression_sorted(&crest);
    let dissonance_sorted = aggression_sorted(&aggression_evenly_sample(&dissonance, 48));
    let high_energy_sorted = aggression_sorted(&high_energy);
    let high_flux_sorted = aggression_sorted(&high_flux);

    crate::aggression::score_evidence(&AggressionEvidence {
        crest_p50: aggression_quantile_sorted(&crest_sorted, 0.50),
        crest_p90: aggression_quantile_sorted(&crest_sorted, 0.90),
        dissonance_p50: aggression_quantile_sorted(&dissonance_sorted, 0.50),
        dissonance_p90: aggression_quantile_sorted(&dissonance_sorted, 0.90),
        mfcc_0: mfcc[0],
        mfcc_2: mfcc[1],
        contrast: contrast_sum,
        centroid: centroid_mean,
        bandwidth: bandwidth_mean,
        bpm,
        onset_density_embedding,
        danceability,
        grid_regularity: aggression_grid_regularity(&beats),
        dynamic_range_db,
        energy,
        high_energy_p50: aggression_quantile_sorted(&high_energy_sorted, 0.50),
        high_energy_p90: aggression_quantile_sorted(&high_energy_sorted, 0.90),
        high_flatness_p50,
        high_flux_p90: aggression_quantile_sorted(&high_flux_sorted, 0.90),
        onset_density: aggression_onset_density,
        onset_interval_cv: aggression_interval_cv(&aggression_onsets),
        onset_strength_p50: onset_p50,
        onset_strength_p90: aggression_quantile_sorted(&onset_sorted, 0.90),
        rms_dynamic_ratio: rms_p90 / rms_p10.max(1.0e-12),
        spectral_peak_ratio: peak_p50,
        window_force_top2,
        window_harshness_top2,
        window_impact_persistence,
        window_impact_top2,
        content_support,
    })
}
