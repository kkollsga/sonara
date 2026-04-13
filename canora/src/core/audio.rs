//! Audio loading, resampling, and time-domain processing.
//!
//! Mirrors librosa.core.audio — load, stream, to_mono, resample,
//! get_duration, get_samplerate, autocorrelate, lpc, zero_crossings,
//! clicks, tone, chirp, mu_compress, mu_expand.

use std::f64::consts::PI;
use std::path::Path;

#[cfg(test)]
use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::error::{CanoraError, Result};
use crate::types::{AudioBuffer, Float};

// ============================================================
// Audio I/O
// ============================================================

/// Load an audio file as a floating-point time series.
///
/// Audio is automatically resampled to the given rate (default 22050 Hz).
/// Use `sr=None` (pass 0) to preserve the native sample rate.
///
/// - `path`: Path to audio file (WAV, FLAC, OGG, MP3, etc.)
/// - `sr`: Target sample rate. 0 means use native.
/// - `mono`: Convert to mono if true.
/// - `offset`: Start reading after this many seconds.
/// - `duration`: Only read this many seconds (0 = all).
pub fn load(
    path: &Path,
    sr: u32,
    mono: bool,
    offset: Float,
    duration: Float,
) -> Result<(Array1<Float>, u32)> {
    // Try hound first for WAV files (fastest path)
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let (samples, native_sr, n_channels) = if ext.eq_ignore_ascii_case("wav") {
        load_wav(path)?
    } else {
        load_symphonia(path)?
    };

    let native_sr = native_sr;

    // Apply offset and duration
    let offset_samples = (offset * native_sr as Float) as usize;
    let total_samples = samples.len() / n_channels;

    let end_sample = if duration > 0.0 {
        let dur_samples = (duration * native_sr as Float) as usize;
        (offset_samples + dur_samples).min(total_samples)
    } else {
        total_samples
    };

    if offset_samples >= total_samples {
        return Err(CanoraError::InvalidParameter {
            param: "offset",
            reason: format!(
                "offset {offset}s exceeds file duration {}s",
                total_samples as Float / native_sr as Float
            ),
        });
    }

    // Deinterleave if multi-channel
    let mut audio = if n_channels == 1 {
        Array1::from_vec(samples[offset_samples..end_sample].to_vec())
    } else {
        let n_samples = end_sample - offset_samples;
        let mut channel_data = Array2::<Float>::zeros((n_channels, n_samples));
        for i in 0..n_samples {
            for ch in 0..n_channels {
                channel_data[(ch, i)] = samples[(offset_samples + i) * n_channels + ch];
            }
        }
        if mono {
            to_mono(channel_data.view())
        } else {
            // Return first channel for now (multi-channel support in later phases)
            channel_data.row(0).to_owned()
        }
    };

    // Resample if needed
    let target_sr = if sr == 0 { native_sr } else { sr };
    if target_sr != native_sr {
        audio = resample(audio.view(), native_sr, target_sr)?;
    }

    Ok((audio, target_sr))
}

/// Load WAV file using hound (fast path).
fn load_wav(path: &Path) -> Result<(Vec<Float>, u32, usize)> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| CanoraError::AudioFile(format!("{}: {}", path.display(), e)))?;

    let spec = reader.spec();
    let sr = spec.sample_rate;
    let n_channels = spec.channels as usize;

    let samples: Vec<Float> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.map(|v| v as Float))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CanoraError::Decode(e.to_string()))?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let scale = 1.0 / (1u64 << (bits - 1)) as Float;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as Float * scale))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| CanoraError::Decode(e.to_string()))?
        }
    };

    Ok((samples, sr, n_channels))
}

/// Load audio file using symphonia (supports mp3, flac, ogg, etc.).
fn load_symphonia(path: &Path) -> Result<(Vec<Float>, u32, usize)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)
        .map_err(|e| CanoraError::AudioFile(format!("{}: {}", path.display(), e)))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| CanoraError::Decode(format!("probe failed: {e}")))?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| CanoraError::Decode("no audio track found".into()))?;
    let track_id = track.id;
    let sr = track
        .codec_params
        .sample_rate
        .ok_or_else(|| CanoraError::Decode("no sample rate".into()))?;
    let n_channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| CanoraError::Decode(format!("codec init failed: {e}")))?;

    // Pre-allocate output with estimated capacity (avoid repeated growth)
    let estimated_samples = track
        .codec_params
        .n_frames
        .map(|f| f as usize * n_channels)
        .unwrap_or(sr as usize * 300 * n_channels); // fallback: 5 min
    let mut samples = Vec::with_capacity(estimated_samples);

    // Reuse SampleBuffer across packets (avoid per-packet allocation)
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(CanoraError::Decode(e.to_string())),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .map_err(|e| CanoraError::Decode(e.to_string()))?;

        let spec = *decoded.spec();
        let capacity = decoded.capacity();

        // Reuse or create SampleBuffer (only allocates when capacity grows)
        if sample_buf.is_none() || sample_buf.as_ref().unwrap().capacity() < capacity {
            sample_buf = Some(SampleBuffer::<f32>::new(capacity as u64, spec));
        }
        let buf = sample_buf.as_mut().unwrap();
        buf.copy_interleaved_ref(decoded);

        samples.extend(buf.samples().iter().map(|&s| s as Float));
    }

    Ok((samples, sr, n_channels))
}

/// Convert a multi-channel signal to mono by averaging channels.
pub fn to_mono(y: ndarray::ArrayView2<Float>) -> Array1<Float> {
    y.mean_axis(Axis(0)).unwrap()
}

/// Resample a signal from `orig_sr` to `target_sr`.
///
/// Uses rubato's sinc interpolation for high-quality resampling.
pub fn resample(
    y: ArrayView1<Float>,
    orig_sr: u32,
    target_sr: u32,
) -> Result<Array1<Float>> {
    if orig_sr == target_sr {
        return Ok(y.to_owned());
    }

    use rubato::{Fft, FixedSync, Resampler};
    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;

    let chunk_size = 1024;
    let mut resampler = Fft::<f64>::new(
        orig_sr as usize,
        target_sr as usize,
        chunk_size,
        1, // sub_chunks
        1, // mono
        FixedSync::Input,
    )
    .map_err(|e| CanoraError::Fft(format!("resampler init: {e}")))?;

    let input_len = y.len();
    let input_vec: Vec<Float> = y.to_vec();
    let input_data = vec![input_vec];
    let input = SequentialSliceOfVecs::new(&input_data, 1, input_len)
        .map_err(|e| CanoraError::Fft(format!("resampler input buffer: {e}")))?;

    let output_len = resampler.process_all_needed_output_len(input_len);
    let mut output_data = vec![vec![0.0f64; output_len]];
    let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 1, output_len)
        .map_err(|e| CanoraError::Fft(format!("resampler output buffer: {e}")))?;

    let (_nbr_in, nbr_out) = resampler
        .process_all_into_buffer(&input, &mut output, input_len, None)
        .map_err(|e| CanoraError::Fft(format!("resample: {e}")))?;

    output_data[0].truncate(nbr_out);
    Ok(Array1::from_vec(output_data.into_iter().next().unwrap()))
}

/// Get the duration of an audio file in seconds.
pub fn get_duration(path: &Path) -> Result<Float> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("wav") {
        let reader = hound::WavReader::open(path)
            .map_err(|e| CanoraError::AudioFile(format!("{}: {}", path.display(), e)))?;
        let spec = reader.spec();
        let n_samples = reader.len() as Float / spec.channels as Float;
        Ok(n_samples / spec.sample_rate as Float)
    } else {
        // Fallback: load and measure (not ideal for large files)
        let (samples, sr, _) = load_symphonia(path)?;
        Ok(samples.len() as Float / sr as Float)
    }
}

/// Get the native sample rate of an audio file.
pub fn get_samplerate(path: &Path) -> Result<u32> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("wav") {
        let reader = hound::WavReader::open(path)
            .map_err(|e| CanoraError::AudioFile(format!("{}: {}", path.display(), e)))?;
        Ok(reader.spec().sample_rate)
    } else {
        let (_, sr, _) = load_symphonia(path)?;
        Ok(sr)
    }
}

// ============================================================
// Time-domain processing
// ============================================================

/// Stream audio from a file in fixed-length blocks.
///
/// Returns a Vec of audio blocks. Each block is an AudioBuffer of `block_length` samples.
/// For true streaming, the caller should process blocks as they arrive.
pub fn stream(
    path: &Path,
    block_length: usize,
    frame_length: usize,
    hop_length: usize,
    sr: u32,
    mono: bool,
) -> Result<Vec<AudioBuffer>> {
    let (y, _) = load(path, sr, mono, 0.0, 0.0)?;
    let step = block_length * hop_length;
    let mut blocks = Vec::new();
    let mut pos = 0;
    while pos < y.len() {
        let end = (pos + step + frame_length).min(y.len());
        blocks.push(y.slice(ndarray::s![pos..end]).to_owned());
        pos += step;
    }
    Ok(blocks)
}

/// Compute the autocorrelation of a 1-D signal.
///
/// Uses the FFT method: autocorrelation = ifft(|fft(x)|²).
pub fn autocorrelate(y: ArrayView1<Float>, max_size: Option<usize>) -> Result<Array1<Float>> {
    let n = y.len();
    let fft_size = (2 * n).next_power_of_two();

    // Zero-pad to fft_size
    let mut padded = vec![0.0; fft_size];
    for i in 0..n {
        padded[i] = y[i];
    }

    // FFT
    let mut spectrum = crate::core::fft::rfft_alloc(&mut padded)?;

    // |X|² (power spectrum)
    for c in spectrum.iter_mut() {
        *c = num_complex::Complex::new(c.norm_sqr(), 0.0);
    }

    // IFFT
    let autocorr = crate::core::fft::irfft_alloc(&mut spectrum, fft_size)?;

    // Trim and normalize
    let max_lag = max_size.unwrap_or(n).min(n);
    let mut result = Array1::<Float>::zeros(max_lag);
    let scale = 1.0 / fft_size as Float;
    for i in 0..max_lag {
        result[i] = autocorr[i] * scale;
    }

    Ok(result)
}

/// Linear Prediction Coefficients via the Burg method.
///
/// Returns `order + 1` coefficients where the first is always 1.0.
pub fn lpc(y: ArrayView1<Float>, order: usize) -> Result<Array1<Float>> {
    let n = y.len();
    if n <= order {
        return Err(CanoraError::InsufficientData {
            needed: order + 1,
            got: n,
        });
    }

    // Levinson-Durbin recursion on autocorrelation
    let acf = autocorrelate(y, Some(order + 1))?;

    if acf[0].abs() < 1e-30 {
        // Signal is essentially zero
        let mut coeffs = Array1::<Float>::zeros(order + 1);
        coeffs[0] = 1.0;
        return Ok(coeffs);
    }

    let mut a = vec![0.0; order + 1];
    let mut a_prev = vec![0.0; order + 1];
    a[0] = 1.0;
    a_prev[0] = 1.0;

    let mut error = acf[0];

    for i in 1..=order {
        // Compute reflection coefficient
        let mut lambda = 0.0;
        for j in 0..i {
            lambda -= a_prev[j] * acf[i - j];
        }
        lambda /= error;

        // Update coefficients
        for j in 0..=i {
            a[j] = a_prev[j] + lambda * a_prev[i - j];
        }

        // Update error
        error *= 1.0 - lambda * lambda;

        if error.abs() < 1e-30 {
            break;
        }

        // Copy for next iteration
        a_prev[..=i].copy_from_slice(&a[..=i]);
    }

    Ok(Array1::from_vec(a))
}

/// Count zero crossings in a signal.
///
/// Returns a boolean array where `true` indicates a zero crossing.
pub fn zero_crossings(y: ArrayView1<Float>, threshold: Float) -> Array1<bool> {
    let n = y.len();
    if n < 2 {
        return Array1::from_elem(n, false);
    }

    Array1::from_shape_fn(n, |i| {
        if i == 0 {
            false
        } else {
            let a = if y[i - 1].abs() <= threshold { 0.0 } else { y[i - 1] };
            let b = if y[i].abs() <= threshold { 0.0 } else { y[i] };
            (a > 0.0 && b <= 0.0) || (a <= 0.0 && b > 0.0)
        }
    })
}

// ============================================================
// Signal generation
// ============================================================

/// Generate click (impulse) signals at specified times.
pub fn clicks(
    times: &[Float],
    sr: u32,
    length: usize,
    click_freq: Float,
    click_duration: Float,
) -> Array1<Float> {
    let mut y = Array1::<Float>::zeros(length);
    let click_samples = (click_duration * sr as Float).ceil() as usize;

    for &t in times {
        let start = (t * sr as Float).round() as usize;
        for i in 0..click_samples {
            let idx = start + i;
            if idx < length {
                y[idx] += (2.0 * PI * click_freq * i as Float / sr as Float).sin();
            }
        }
    }

    y
}

/// Generate a pure tone (sine wave).
pub fn tone(frequency: Float, sr: u32, length: usize) -> Array1<Float> {
    Array1::from_shape_fn(length, |i| {
        (2.0 * PI * frequency * i as Float / sr as Float).sin()
    })
}

/// Generate a chirp (swept-frequency sine wave).
///
/// Linear sweep from `fmin` to `fmax` over `length` samples.
pub fn chirp(fmin: Float, fmax: Float, sr: u32, length: usize) -> Array1<Float> {
    let duration = length as Float / sr as Float;
    Array1::from_shape_fn(length, |i| {
        let t = i as Float / sr as Float;
        let phase = 2.0 * PI * (fmin * t + (fmax - fmin) * t * t / (2.0 * duration));
        phase.sin()
    })
}

// ============================================================
// Companding
// ============================================================

/// Mu-law compression.
///
/// `y = sign(x) * ln(1 + mu*|x|) / ln(1 + mu)`
pub fn mu_compress(x: ArrayView1<Float>, mu: Float) -> Array1<Float> {
    let log_mu = (1.0 + mu).ln();
    x.mapv(|v| v.signum() * (1.0 + mu * v.abs()).ln() / log_mu)
}

/// Mu-law expansion (inverse of mu_compress).
///
/// `x = sign(y) * (1/mu) * ((1 + mu)^|y| - 1)`
pub fn mu_expand(y: ArrayView1<Float>, mu: Float) -> Array1<Float> {
    y.mapv(|v| v.signum() * ((1.0 + mu).powf(v.abs()) - 1.0) / mu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_to_mono_stereo() {
        let stereo = Array2::from_shape_vec(
            (2, 4),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let mono = to_mono(stereo.view());
        assert_eq!(mono.len(), 4);
        assert_abs_diff_eq!(mono[0], 3.0, epsilon = 1e-14); // (1+5)/2
        assert_abs_diff_eq!(mono[1], 4.0, epsilon = 1e-14); // (2+6)/2
    }

    #[test]
    fn test_tone_440hz() {
        let y = tone(440.0, 22050, 22050);
        assert_eq!(y.len(), 22050);
        // Check it starts at zero
        assert_abs_diff_eq!(y[0], 0.0, epsilon = 1e-14);
        // Check energy is reasonable (RMS of sine = 1/sqrt(2))
        let rms = (y.mapv(|v| v * v).sum() / y.len() as Float).sqrt();
        assert_abs_diff_eq!(rms, 1.0 / 2.0_f64.sqrt(), epsilon = 0.01);
    }

    #[test]
    fn test_chirp_basic() {
        let y = chirp(100.0, 1000.0, 22050, 22050);
        assert_eq!(y.len(), 22050);
        // Values should be in [-1, 1]
        for &v in y.iter() {
            assert!(v >= -1.01 && v <= 1.01);
        }
    }

    #[test]
    fn test_mu_compress_expand_roundtrip() {
        let x = array![-0.8, -0.3, 0.0, 0.5, 1.0];
        let mu = 255.0;
        let compressed = mu_compress(x.view(), mu);
        let expanded = mu_expand(compressed.view(), mu);
        for i in 0..x.len() {
            assert_abs_diff_eq!(x[i], expanded[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mu_compress_zero() {
        let x = array![0.0];
        let compressed = mu_compress(x.view(), 255.0);
        assert_abs_diff_eq!(compressed[0], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_zero_crossings_sine() {
        let n = 1000;
        let y = Array1::from_shape_fn(n, |i| {
            (2.0 * PI * 10.0 * i as Float / n as Float).sin()
        });
        let zc = zero_crossings(y.view(), 0.0);
        let count: usize = zc.iter().filter(|&&v| v).count();
        // 10 cycles → ~20 zero crossings
        assert!(count >= 18 && count <= 22, "got {count} crossings");
    }

    #[test]
    fn test_zero_crossings_threshold() {
        let y = array![0.0, 0.01, -0.01, 0.0];
        // With high threshold, small values are treated as zero → fewer crossings
        let zc = zero_crossings(y.view(), 0.1);
        let count: usize = zc.iter().filter(|&&v| v).count();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_autocorrelate_sine() {
        let n = 2048;
        let freq = 100.0;
        let sr = 22050.0;
        let y = Array1::from_shape_fn(n, |i| {
            (2.0 * PI * freq * i as Float / sr).sin()
        });
        let acf = autocorrelate(y.view(), Some(n)).unwrap();
        // Autocorrelation of sine peaks at 0 and at period
        assert!(acf[0] > 0.0); // Peak at lag 0
        let period_samples = (sr / freq).round() as usize;
        // Should have a peak near the period
        let peak_region = &acf.as_slice().unwrap()[period_samples - 5..period_samples + 5];
        let local_max = peak_region.iter().copied().fold(Float::NEG_INFINITY, Float::max);
        assert!(local_max > acf[0] * 0.5, "autocorrelation should peak near period");
    }

    #[test]
    fn test_lpc_basic() {
        let y = Array1::from_shape_fn(256, |i| (i as Float * 0.1).sin());
        let coeffs = lpc(y.view(), 4).unwrap();
        assert_eq!(coeffs.len(), 5); // order + 1
        assert_abs_diff_eq!(coeffs[0], 1.0, epsilon = 1e-10); // first coeff is always 1
    }

    #[test]
    fn test_lpc_insufficient_data() {
        let y = array![1.0, 2.0];
        assert!(lpc(y.view(), 5).is_err());
    }

    #[test]
    fn test_clicks_basic() {
        let y = clicks(&[0.0, 0.5], 22050, 22050, 1000.0, 0.01);
        assert_eq!(y.len(), 22050);
        // Should have energy near the click times
        assert!(y.slice(s![0..220]).iter().any(|&v| v.abs() > 0.01));
        let mid = 22050 / 2;
        assert!(y.slice(s![mid..mid + 220]).iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn test_resample_identity() {
        let y = Array1::from_shape_fn(1024, |i| (i as Float * 0.1).sin());
        let resampled = resample(y.view(), 22050, 22050).unwrap();
        assert_eq!(resampled.len(), y.len());
        for i in 0..y.len() {
            assert_abs_diff_eq!(y[i], resampled[i], epsilon = 1e-14);
        }
    }
}
