use pyo3::prelude::*;

mod analyze;
mod beat;
mod core;
mod effects;
mod error;
mod feature;
mod filters;
mod onset;
mod tonal;
mod util;

#[pymodule]
fn _sonara(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version — sourced from sonara crate's Cargo.toml via env! at compile time
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register submodules
    analyze::register(m)?;
    beat::register(m)?;
    core::register(m)?;
    effects::register(m)?;
    feature::register(m)?;
    filters::register(m)?;
    onset::register(m)?;
    tonal::register(m)?;
    util::register(m)?;

    // ========================================================
    // Top-level re-exports
    // ========================================================

    // --- Core: Audio I/O ---
    m.add_function(wrap_pyfunction!(core::audio::py_load, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_to_mono, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_resample, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_get_duration, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_get_samplerate, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_autocorrelate, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_lpc, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_zero_crossings, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_clicks, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_tone, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_chirp, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_mu_compress, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_mu_expand, m)?)?;
    m.add_function(wrap_pyfunction!(core::audio::py_stream_with_resample, m)?)?;

    // --- Core: Conversions (all 50+) ---
    m.add_function(wrap_pyfunction!(core::convert::py_hz_to_mel, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_mel_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_hz_to_midi, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_midi_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_note_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_note_to_midi, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_midi_to_note, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_hz_to_note, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_hz_to_octs, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_octs_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_a4_to_tuning, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_tuning_to_a4, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_hz_to_svara_h, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_hz_to_svara_c, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_midi_to_svara_h, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_midi_to_svara_c, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_note_to_svara_h, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_note_to_svara_c, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_hz_to_fjs, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_fft_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_mel_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_cqt_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_tempo_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_fourier_tempo_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_frames_to_samples, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_frames_to_time, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_samples_to_frames, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_samples_to_time, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_time_to_frames, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_time_to_samples, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_blocks_to_frames, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_blocks_to_samples, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_blocks_to_time, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_a_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_b_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_c_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_d_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_z_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_frequency_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_multi_frequency_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_samples_like, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_times_like, m)?)?;
    // Notation
    m.add_function(wrap_pyfunction!(core::convert::py_key_to_notes, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_key_to_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_mela_to_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_mela_to_svara, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_thaat_to_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_list_mela, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_list_thaat, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_fifths_to_note, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_interval_to_fjs, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_interval_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_pythagorean_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(core::convert::py_plimit_intervals, m)?)?;

    // --- Core: Spectrum ---
    m.add_function(wrap_pyfunction!(core::spectrum::py_stft, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_istft, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_power_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_amplitude_to_db, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_db_to_power, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_db_to_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_magphase, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_phase_vocoder, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_pcen, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_perceptual_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_griffinlim, m)?)?;
    // CQT
    m.add_function(wrap_pyfunction!(core::spectrum::py_cqt, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_vqt, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_hybrid_cqt, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_pseudo_cqt, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_icqt, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_griffinlim_cqt, m)?)?;
    // Pitch
    m.add_function(wrap_pyfunction!(core::spectrum::py_yin, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_pyin, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_estimate_tuning, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_pitch_tuning, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_piptrack, m)?)?;
    // Harmonic
    m.add_function(wrap_pyfunction!(core::spectrum::py_salience, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_interp_harmonics, m)?)?;
    m.add_function(wrap_pyfunction!(core::spectrum::py_f0_harmonics, m)?)?;

    // --- Effects ---
    m.add_function(wrap_pyfunction!(effects::py_trim, m)?)?;
    m.add_function(wrap_pyfunction!(effects::py_split, m)?)?;
    m.add_function(wrap_pyfunction!(effects::py_split_with_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(effects::py_melody_separate, m)?)?;

    // --- Onset / Beat ---
    m.add_function(wrap_pyfunction!(onset::py_onset_detect, m)?)?;
    m.add_function(wrap_pyfunction!(onset::py_onset_strength, m)?)?;
    m.add_function(wrap_pyfunction!(onset::py_onset_strength_method, m)?)?;
    m.add_function(wrap_pyfunction!(beat::py_beat_track, m)?)?;
    m.add_function(wrap_pyfunction!(beat::py_tempo_curve, m)?)?;
    m.add_function(wrap_pyfunction!(beat::py_tempo_variability, m)?)?;

    // --- Filters ---
    m.add_function(wrap_pyfunction!(filters::py_mel, m)?)?;

    // --- Tonal ---
    m.add_function(wrap_pyfunction!(tonal::py_hpcp, m)?)?;
    m.add_function(wrap_pyfunction!(tonal::py_chords_from_beats, m)?)?;
    m.add_function(wrap_pyfunction!(tonal::py_chords_from_frames, m)?)?;
    m.add_function(wrap_pyfunction!(tonal::py_chord_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(tonal::py_dissonance, m)?)?;
    m.add_function(wrap_pyfunction!(tonal::py_dissonance_from_peaks, m)?)?;

    // --- Features ---
    m.add_function(wrap_pyfunction!(feature::spectral::py_melspectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(feature::spectral::py_mfcc, m)?)?;
    m.add_function(wrap_pyfunction!(feature::spectral::py_chroma_stft, m)?)?;
    m.add_function(wrap_pyfunction!(feature::spectral::py_spectral_centroid, m)?)?;
    m.add_function(wrap_pyfunction!(feature::spectral::py_rms, m)?)?;

    Ok(())
}
