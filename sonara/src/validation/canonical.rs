use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::ValidationError;

fn reject_floats(value: &Value) -> Result<(), ValidationError> {
    match value {
        Value::Number(number) if !number.is_i64() && !number.is_u64() => {
            Err(ValidationError::FloatingPointIdentity)
        }
        Value::Array(values) => values.iter().try_for_each(reject_floats),
        Value::Object(values) => values.values().try_for_each(reject_floats),
        _ => Ok(()),
    }
}

/// Serialize to Sonara's canonical compact JSON representation.
///
/// `serde_json::Map` is key-sorted when `preserve_order` is disabled (Sonara
/// does not enable it), so converting through `Value` fixes object order.
pub fn to_canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, ValidationError> {
    let value = serde_json::to_value(value)?;
    reject_floats(&value)?;
    Ok(serde_json::to_vec(&value)?)
}

/// Decode strict typed JSON and require the input bytes themselves to be canonical.
pub fn from_canonical_bytes<T>(bytes: &[u8]) -> Result<T, ValidationError>
where
    T: DeserializeOwned + Serialize,
{
    let value: T = serde_json::from_slice(bytes)?;
    let canonical = to_canonical_bytes(&value)?;
    if canonical != bytes {
        return Err(ValidationError::NonCanonicalJson);
    }
    Ok(value)
}

/// SHA-256 with an explicit domain separator and canonical payload bytes.
pub fn canonical_digest<T: Serialize>(domain: &str, value: &T) -> Result<String, ValidationError> {
    if domain.is_empty() || domain.as_bytes().contains(&0) {
        return Err(ValidationError::InvalidDomain);
    }
    let bytes = to_canonical_bytes(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}
