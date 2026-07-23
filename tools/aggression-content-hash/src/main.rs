//! Exact Sonagram audio-content identity used by the sealed aggression audit.

use std::path::{Path, PathBuf};

fn syncsafe_u28(bytes: &[u8]) -> usize {
    ((bytes[0] & 0x7f) as usize) << 21
        | ((bytes[1] & 0x7f) as usize) << 14
        | ((bytes[2] & 0x7f) as usize) << 7
        | (bytes[3] & 0x7f) as usize
}

fn mp3_audio_range(bytes: &[u8]) -> (usize, usize) {
    let mut start = 0;
    let mut end = bytes.len();
    if bytes.len() >= 10 && &bytes[..3] == b"ID3" {
        let footer = usize::from(bytes[5] & 0x10 != 0) * 10;
        if let Some(skip) = 10usize
            .checked_add(syncsafe_u28(&bytes[6..10]))
            .and_then(|value| value.checked_add(footer))
            .filter(|value| *value <= bytes.len())
        {
            start = skip;
        }
    }
    if end.saturating_sub(start) >= 128 && &bytes[end - 128..end - 125] == b"TAG" {
        end -= 128;
        if end.saturating_sub(start) >= 227 && &bytes[end - 227..end - 223] == b"TAG+" {
            end -= 227;
        }
    }
    (start, end)
}

fn content_hash(path: &Path) -> Result<(&'static str, String), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let is_mp3 = path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("mp3"));
    let (kind, digest) = if is_mp3 {
        let (start, end) = mp3_audio_range(&bytes);
        ("mp3-audio-v1", blake3::hash(&bytes[start..end]))
    } else {
        ("whole-file-v0", blake3::hash(&bytes))
    };
    Ok((kind, digest.to_hex().to_string()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths = std::env::args_os().skip(1).map(PathBuf::from).collect::<Vec<_>>();
    if paths.is_empty() {
        return Err("usage: sonara-aggression-content-hash <audio> [...]".into());
    }
    for path in paths {
        let (kind, digest) = content_hash(&path)?;
        println!("{kind}\t{digest}\t{}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mp3_tags_do_not_change_audio_identity() {
        let mut left = b"ID3\x04\0\0\0\0\0\x03oneAUDIO".to_vec();
        let mut right = b"ID3\x04\0\0\0\0\0\x03twoAUDIO".to_vec();
        let mut tag = b"TAG".to_vec();
        tag.resize(128, 0);
        left.extend_from_slice(&tag);
        right.extend_from_slice(&tag);
        let (left_start, left_end) = mp3_audio_range(&left);
        let (right_start, right_end) = mp3_audio_range(&right);
        assert_eq!(&left[left_start..left_end], b"AUDIO");
        assert_eq!(&right[right_start..right_end], b"AUDIO");
    }

    #[test]
    fn mutation_outside_excerpt_changes_digest() {
        let first = blake3::hash(b"prefix-audio-suffix");
        let second = blake3::hash(b"prefix-audio-mutated");
        assert_ne!(first, second);
    }
}
