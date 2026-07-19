//! Music notation utilities.
//!
//! Key/scale mapping, Indian notation (mela/thaat/svara), and just intonation
//! (FJS, Pythagorean, p-limit intervals).

use crate::error::{Result, SonaraError};
use crate::types::Float;

// ============================================================
// Western key/scale functions
// ============================================================

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];
const NOTE_NAMES_FLAT: [&str; 12] = [
    "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
];

/// Major scale intervals (semitones from root).
const MAJOR_INTERVALS: [usize; 7] = [0, 2, 4, 5, 7, 9, 11];
/// Natural minor scale intervals.
const MINOR_INTERVALS: [usize; 7] = [0, 2, 3, 5, 7, 8, 10];

/// Convert a key string to note names.
///
/// - `key`: e.g., "C:maj", "A:min", "Eb:maj"
///
/// Returns 7 note names.
pub fn key_to_notes(key: &str) -> Result<Vec<String>> {
    let (root, mode) = parse_key(key)?;
    let intervals = match mode.as_str() {
        "maj" | "major" => &MAJOR_INTERVALS,
        "min" | "minor" => &MINOR_INTERVALS,
        _ => {
            return Err(SonaraError::InvalidParameter {
                param: "key",
                reason: format!("unknown mode: '{mode}'"),
            })
        }
    };

    let root_idx = note_name_to_index(&root)?;
    let use_flats =
        root.contains('b') || ["F", "Bb", "Eb", "Ab", "Db", "Gb"].contains(&root.as_str());

    let names = if use_flats {
        &NOTE_NAMES_FLAT
    } else {
        &NOTE_NAMES
    };

    Ok(intervals
        .iter()
        .map(|&i| names[(root_idx + i) % 12].to_string())
        .collect())
}

/// Convert a key string to scale degrees (semitone offsets from root).
pub fn key_to_degrees(key: &str) -> Result<Vec<usize>> {
    let (_root, mode) = parse_key(key)?;
    match mode.as_str() {
        "maj" | "major" => Ok(MAJOR_INTERVALS.to_vec()),
        "min" | "minor" => Ok(MINOR_INTERVALS.to_vec()),
        _ => Err(SonaraError::InvalidParameter {
            param: "key",
            reason: format!("unknown mode: '{mode}'"),
        }),
    }
}

fn parse_key(key: &str) -> Result<(String, String)> {
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() != 2 {
        return Err(SonaraError::InvalidParameter {
            param: "key",
            reason: format!("key must be 'note:mode', got '{key}'"),
        });
    }
    Ok((parts[0].to_string(), parts[1].to_lowercase()))
}

fn note_name_to_index(name: &str) -> Result<usize> {
    for (i, &n) in NOTE_NAMES.iter().enumerate() {
        if n.eq_ignore_ascii_case(name) {
            return Ok(i);
        }
    }
    for (i, &n) in NOTE_NAMES_FLAT.iter().enumerate() {
        if n.eq_ignore_ascii_case(name) {
            return Ok(i);
        }
    }
    Err(SonaraError::InvalidParameter {
        param: "note",
        reason: format!("unknown note name: '{name}'"),
    })
}

// ============================================================
// Indian music (Melakarta / Thaat)
// ============================================================

/// All 72 Melakarta raga names.
const MELAKARTA_NAMES: [&str; 72] = [
    "Kanakangi",
    "Ratnangi",
    "Ganamurti",
    "Vanaspati",
    "Manavati",
    "Tanarupi",
    "Senavati",
    "Hanumatodi",
    "Dhenuka",
    "Natakapriya",
    "Kokilapriya",
    "Rupavati",
    "Gayakapriya",
    "Vakulabharanam",
    "Mayamalavagowla",
    "Chakravakam",
    "Suryakantam",
    "Hatakambari",
    "Jhankaradhvani",
    "Natabhairavi",
    "Keeravani",
    "Kharaharapriya",
    "Gaurimanohari",
    "Varunapriya",
    "Mararanjani",
    "Charukesi",
    "Sarasangi",
    "Harikambhoji",
    "Dheerasankarabharanam",
    "Naganandini",
    "Yagapriya",
    "Ragavardhini",
    "Gangeyabhushani",
    "Vagadheeswari",
    "Shulini",
    "Chalanata",
    "Salagam",
    "Jalarnavam",
    "Jhalavarali",
    "Navaneetam",
    "Pavani",
    "Raghupriya",
    "Gavambodhi",
    "Bhavapriya",
    "Shubhapantuvarali",
    "Shadvidamargini",
    "Suvarnangi",
    "Divyamani",
    "Dhavalambari",
    "Namanarayani",
    "Kamavardhini",
    "Ramapriya",
    "Gamanashrama",
    "Vishwambhari",
    "Shamalangi",
    "Shanmukhapriya",
    "Simhendramadhyamam",
    "Hemavati",
    "Dharmavati",
    "Neetimati",
    "Kantamani",
    "Rishabhapriya",
    "Latangi",
    "Vachaspati",
    "Mechakalyani",
    "Chitrambari",
    "Sucharitra",
    "Jyotiswarupini",
    "Dhatuvardhini",
    "Nasikabhushani",
    "Kosalam",
    "Rasikapriya",
];

/// 10 Thaat names.
const THAAT_NAMES: [&str; 10] = [
    "Bilaval", "Khamaj", "Kafi", "Asavari", "Bhairavi", "Bhairav", "Kalyan", "Marva", "Poorvi",
    "Todi",
];

/// Thaat scale degrees (semitone offsets from Sa).
const THAAT_DEGREES: [[usize; 7]; 10] = [
    [0, 2, 4, 5, 7, 9, 11], // Bilaval (= major)
    [0, 2, 4, 5, 7, 9, 10], // Khamaj
    [0, 2, 3, 5, 7, 9, 10], // Kafi
    [0, 2, 3, 5, 7, 8, 10], // Asavari (= natural minor)
    [0, 1, 3, 5, 7, 8, 10], // Bhairavi
    [0, 1, 4, 5, 7, 9, 11], // Bhairav
    [0, 2, 4, 6, 7, 9, 11], // Kalyan
    [0, 1, 4, 6, 7, 9, 11], // Marva
    [0, 1, 4, 6, 7, 8, 11], // Poorvi
    [0, 1, 3, 6, 7, 8, 11], // Todi
];

/// Convert a melakarta number (1-72) to svara names.
pub fn mela_to_svara(mela: usize) -> Result<Vec<String>> {
    if mela < 1 || mela > 72 {
        return Err(SonaraError::InvalidParameter {
            param: "mela",
            reason: format!("melakarta number must be 1-72, got {mela}"),
        });
    }
    let degrees = mela_to_degrees(mela)?;
    let svara_names = [
        "Sa", "Ri1", "Ri2", "Ga1", "Ga2", "Ma1", "Ma2", "Pa", "Da1", "Da2", "Ni1", "Ni2",
    ];
    Ok(degrees
        .iter()
        .map(|&d| svara_names[d % 12].to_string())
        .collect())
}

/// Convert a melakarta number to scale degrees.
pub fn mela_to_degrees(mela: usize) -> Result<Vec<usize>> {
    if mela < 1 || mela > 72 {
        return Err(SonaraError::InvalidParameter {
            param: "mela",
            reason: format!("melakarta number must be 1-72, got {mela}"),
        });
    }

    let idx = mela - 1;
    let chakra = idx / 6; // 0-11
    let sub = idx % 6; // 0-5

    // Sa is always 0, Pa is always 7
    let mut degrees = vec![0usize; 7];
    degrees[0] = 0; // Sa
    degrees[4] = 7; // Pa

    // Ma: chakras 0-5 use Ma1 (5), chakras 6-11 use Ma2 (6)
    degrees[3] = if chakra < 6 { 5 } else { 6 };

    // Ri and Ga depend on chakra within the half
    let half_chakra = chakra % 6;
    match half_chakra {
        0 => {
            degrees[1] = 1;
            degrees[2] = 2;
        }
        1 => {
            degrees[1] = 1;
            degrees[2] = 3;
        }
        2 => {
            degrees[1] = 1;
            degrees[2] = 4;
        }
        3 => {
            degrees[1] = 2;
            degrees[2] = 3;
        }
        4 => {
            degrees[1] = 2;
            degrees[2] = 4;
        }
        5 => {
            degrees[1] = 3;
            degrees[2] = 4;
        }
        _ => unreachable!(),
    }

    // Da and Ni depend on sub-index
    match sub {
        0 => {
            degrees[5] = 8;
            degrees[6] = 9;
        }
        1 => {
            degrees[5] = 8;
            degrees[6] = 10;
        }
        2 => {
            degrees[5] = 8;
            degrees[6] = 11;
        }
        3 => {
            degrees[5] = 9;
            degrees[6] = 10;
        }
        4 => {
            degrees[5] = 9;
            degrees[6] = 11;
        }
        5 => {
            degrees[5] = 10;
            degrees[6] = 11;
        }
        _ => unreachable!(),
    }

    Ok(degrees)
}

/// Convert a thaat name to scale degrees.
pub fn thaat_to_degrees(thaat: &str) -> Result<Vec<usize>> {
    for (i, &name) in THAAT_NAMES.iter().enumerate() {
        if name.eq_ignore_ascii_case(thaat) {
            return Ok(THAAT_DEGREES[i].to_vec());
        }
    }
    Err(SonaraError::InvalidParameter {
        param: "thaat",
        reason: format!("unknown thaat: '{thaat}'"),
    })
}

/// List all 72 melakarta raga names.
pub fn list_mela() -> Vec<String> {
    MELAKARTA_NAMES.iter().map(|s| s.to_string()).collect()
}

/// List all 10 thaat names.
pub fn list_thaat() -> Vec<String> {
    THAAT_NAMES.iter().map(|s| s.to_string()).collect()
}

// ============================================================
// Circle of fifths / FJS
// ============================================================

/// Convert a position on the circle of fifths to a note name.
///
/// `fifths=0` → "C", `fifths=1` → "G", `fifths=-1` → "F", etc.
pub fn fifths_to_note(fifths: i32) -> String {
    // Circle of fifths starting from C
    let idx = ((fifths * 7) % 12 + 12) % 12;
    NOTE_NAMES[idx as usize].to_string()
}

/// Convert a frequency ratio interval to FJS notation.
///
/// Simplified version — returns a string representation.
pub fn interval_to_fjs(interval: Float) -> String {
    if interval <= 0.0 {
        return String::new();
    }

    // Fold to one octave [1, 2)
    let mut ratio = interval;
    while ratio >= 2.0 {
        ratio /= 2.0;
    }
    while ratio < 1.0 {
        ratio *= 2.0;
    }

    // Common intervals
    let known: [(Float, &str); 7] = [
        (1.0, "P1"),
        (9.0 / 8.0, "M2"),
        (5.0 / 4.0, "M3"),
        (4.0 / 3.0, "P4"),
        (3.0 / 2.0, "P5"),
        (5.0 / 3.0, "M6"),
        (15.0 / 8.0, "M7"),
    ];

    for &(r, name) in &known {
        if (ratio - r).abs() < 0.01 {
            return name.to_string();
        }
    }

    format!("{:.4}", ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_to_notes_c_major() {
        let notes = key_to_notes("C:maj").unwrap();
        assert_eq!(notes, vec!["C", "D", "E", "F", "G", "A", "B"]);
    }

    #[test]
    fn test_key_to_notes_a_minor() {
        let notes = key_to_notes("A:min").unwrap();
        assert_eq!(notes, vec!["A", "B", "C", "D", "E", "F", "G"]);
    }

    #[test]
    fn test_key_to_degrees_major() {
        let degrees = key_to_degrees("C:maj").unwrap();
        assert_eq!(degrees, vec![0, 2, 4, 5, 7, 9, 11]);
    }

    #[test]
    fn test_key_to_degrees_minor() {
        let degrees = key_to_degrees("A:min").unwrap();
        assert_eq!(degrees, vec![0, 2, 3, 5, 7, 8, 10]);
    }

    #[test]
    fn test_thaat_to_degrees_bilaval() {
        let degrees = thaat_to_degrees("Bilaval").unwrap();
        assert_eq!(degrees, vec![0, 2, 4, 5, 7, 9, 11]);
    }

    #[test]
    fn test_thaat_to_degrees_todi() {
        let degrees = thaat_to_degrees("Todi").unwrap();
        assert_eq!(degrees, vec![0, 1, 3, 6, 7, 8, 11]);
    }

    #[test]
    fn test_list_mela() {
        let melas = list_mela();
        assert_eq!(melas.len(), 72);
        assert_eq!(melas[0], "Kanakangi");
    }

    #[test]
    fn test_list_thaat() {
        let thaats = list_thaat();
        assert_eq!(thaats.len(), 10);
        assert_eq!(thaats[0], "Bilaval");
    }

    #[test]
    fn test_mela_to_degrees_range() {
        // All 72 melakartas should work
        for i in 1..=72 {
            let degrees = mela_to_degrees(i).unwrap();
            assert_eq!(degrees.len(), 7);
            assert_eq!(degrees[0], 0); // Sa
            assert_eq!(degrees[4], 7); // Pa
        }
    }

    #[test]
    fn test_mela_to_degrees_invalid() {
        assert!(mela_to_degrees(0).is_err());
        assert!(mela_to_degrees(73).is_err());
    }

    #[test]
    fn test_fifths_to_note() {
        assert_eq!(fifths_to_note(0), "C");
        assert_eq!(fifths_to_note(1), "G");
        assert_eq!(fifths_to_note(-1), "F");
        assert_eq!(fifths_to_note(2), "D");
    }

    #[test]
    fn test_interval_to_fjs_known() {
        assert_eq!(interval_to_fjs(1.0), "P1");
        assert_eq!(interval_to_fjs(1.5), "P5");
        assert_eq!(interval_to_fjs(1.25), "M3");
    }
}
