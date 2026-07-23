use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::canonical_digest;

pub const CAPSULE_FORMAT: &str = "sonara.validation-capsule.v1";
pub const RECEIPT_FORMAT: &str = "sonara.evaluation-receipt.v1";
pub const ENVELOPE_FORMAT: &str = "sonara.custody-envelope.v1";
pub const PROOF_FORMAT: &str = "sonara.custody-proof.v1";
pub const CLAIM_NAMESPACE: &str = "sonara.validation.claim.v1";
pub const COMPLETION_NAMESPACE: &str = "sonara.validation.completion.v1";
pub const CHECKPOINT_NAMESPACE: &str = "sonara.validation.checkpoint.v1";
const CAPSULE_DOMAIN: &str = "sonara-validation-capsule-v1";
const RECEIPT_DOMAIN: &str = "sonara-evaluation-receipt-v1";
const ENVELOPE_DOMAIN: &str = "sonara-custody-envelope-v1";

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("JSON is not the canonical Sonara representation")]
    NonCanonicalJson,
    #[error("floating-point values are forbidden in validation identities")]
    FloatingPointIdentity,
    #[error("invalid digest domain")]
    InvalidDomain,
    #[error("invalid validation field: {0}")]
    InvalidField(&'static str),
    #[error("duplicate validation identity: {0}")]
    DuplicateIdentity(&'static str),
    #[error("private or filesystem path is forbidden in a public validation artifact")]
    PrivatePath,
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_logical(value: &str) -> Result<(), ValidationError> {
    let lower = value.to_ascii_lowercase();
    if value.is_empty()
        || value.starts_with('/')
        || value.starts_with('\\')
        || value.contains("..")
        || value.contains("file://")
        || lower.get(1..3) == Some(":\\")
    {
        return Err(ValidationError::PrivatePath);
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DigestRef {
    pub algorithm: String,
    pub value: String,
}

impl DigestRef {
    pub fn sha256(value: impl Into<String>) -> Self {
        Self {
            algorithm: "sha256".into(),
            value: value.into(),
        }
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.algorithm != "sha256" || !valid_hex(&self.value, 64) {
            return Err(ValidationError::InvalidField("digest"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub id: String,
    pub role: String,
    pub digest: DigestRef,
    pub size_bytes: u64,
}

impl ArtifactRef {
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_logical(&self.id)?;
        validate_logical(&self.role)?;
        self.digest.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedDecimal {
    pub name: String,
    /// Canonical decimal text. Exponents, NaN, and infinities are forbidden.
    pub value: String,
}

impl NamedDecimal {
    pub fn validate(&self) -> Result<(), ValidationError> {
        validate_logical(&self.name)?;
        let bytes = self.value.as_bytes();
        let digits = bytes.iter().filter(|byte| byte.is_ascii_digit()).count();
        let dots = bytes.iter().filter(|byte| **byte == b'.').count();
        let signs = usize::from(bytes.first() == Some(&b'-'));
        if digits == 0
            || dots > 1
            || bytes.len() != digits + dots + signs
            || self.value.starts_with('+')
            || self.value.ends_with('.')
            || (self.value.starts_with('0')
                && self.value.len() > 1
                && !self.value.starts_with("0."))
            || (self.value.starts_with("-0")
                && self.value.len() > 2
                && !self.value.starts_with("-0."))
        {
            return Err(ValidationError::InvalidField("decimal"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateIdentity {
    pub commit: String,
    pub tree: String,
    pub binary_diff: DigestRef,
}

impl CandidateIdentity {
    fn validate(&self) -> Result<(), ValidationError> {
        if !valid_hex(&self.commit, 40) || !valid_hex(&self.tree, 40) {
            return Err(ValidationError::InvalidField("candidate git identity"));
        }
        self.binary_diff.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeIdentity {
    pub model_id: String,
    pub schema_version: u32,
    pub artifacts: Vec<ArtifactRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluatorKind {
    Executable,
    SignedHumanPanel,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorEvidence {
    pub id: String,
    pub kind: EvaluatorKind,
    pub artifacts: Vec<ArtifactRef>,
    pub output_digest: DigestRef,
    pub attestation_digest: DigestRef,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationSuite {
    pub kind: String,
    pub schema_version: u32,
    pub thresholds: Vec<NamedDecimal>,
    pub evaluator: EvaluatorEvidence,
    pub controls: Vec<ArtifactRef>,
    pub tooling: Vec<ArtifactRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationCapsule {
    pub format: String,
    pub feature: String,
    pub model_id: String,
    /// Domain-separated identity of the exact runner command specification.
    pub command_digest: DigestRef,
    pub candidate: CandidateIdentity,
    pub runtime: RuntimeIdentity,
    pub suite: ValidationSuite,
    /// Includes the actual suite freeze/config bytes as a content-addressed artifact.
    pub artifacts: Vec<ArtifactRef>,
}

fn validate_artifacts<'a>(
    artifacts: impl Iterator<Item = &'a ArtifactRef>,
) -> Result<(), ValidationError> {
    let mut ids = HashSet::new();
    let mut identities = HashSet::new();
    for artifact in artifacts {
        artifact.validate()?;
        if !ids.insert(&artifact.id) {
            return Err(ValidationError::DuplicateIdentity("artifact id"));
        }
        if !identities.insert((artifact.role.as_str(), artifact.digest.value.as_str())) {
            return Err(ValidationError::DuplicateIdentity("artifact role/digest"));
        }
    }
    Ok(())
}

impl ValidationCapsule {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.format != CAPSULE_FORMAT
            || self.runtime.schema_version == 0
            || self.suite.schema_version == 0
        {
            return Err(ValidationError::InvalidField("capsule format/schema"));
        }
        validate_logical(&self.feature)?;
        validate_logical(&self.model_id)?;
        self.command_digest.validate()?;
        validate_logical(&self.runtime.model_id)?;
        validate_logical(&self.suite.kind)?;
        self.candidate.validate()?;
        let mut threshold_names = HashSet::new();
        for threshold in &self.suite.thresholds {
            threshold.validate()?;
            if !threshold_names.insert(&threshold.name) {
                return Err(ValidationError::DuplicateIdentity("threshold"));
            }
        }
        validate_logical(&self.suite.evaluator.id)?;
        self.suite.evaluator.output_digest.validate()?;
        self.suite.evaluator.attestation_digest.validate()?;
        validate_artifacts(
            self.runtime
                .artifacts
                .iter()
                .chain(self.suite.evaluator.artifacts.iter())
                .chain(self.suite.controls.iter())
                .chain(self.suite.tooling.iter())
                .chain(self.artifacts.iter()),
        )
    }

    pub fn evaluation_digest(&self) -> Result<DigestRef, ValidationError> {
        self.validate()?;
        Ok(DigestRef::sha256(canonical_digest(CAPSULE_DOMAIN, self)?))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Pass,
    NoGo,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisclosureClass {
    AggregateOnly,
    Public,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationReceipt {
    pub format: String,
    pub evaluation_digest: DigestRef,
    pub claim_digest: DigestRef,
    pub runtime_digest: DigestRef,
    pub runner_digest: DigestRef,
    pub outcome: Outcome,
    pub metrics: Vec<NamedDecimal>,
    pub evidence: Vec<ArtifactRef>,
    pub disclosure: DisclosureClass,
}

impl EvaluationReceipt {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.format != RECEIPT_FORMAT {
            return Err(ValidationError::InvalidField("receipt format"));
        }
        self.evaluation_digest.validate()?;
        self.claim_digest.validate()?;
        self.runtime_digest.validate()?;
        self.runner_digest.validate()?;
        let mut names = HashSet::new();
        for metric in &self.metrics {
            metric.validate()?;
            if !names.insert(&metric.name) {
                return Err(ValidationError::DuplicateIdentity("metric"));
            }
        }
        validate_artifacts(self.evidence.iter())
    }

    pub fn receipt_digest(&self) -> Result<DigestRef, ValidationError> {
        self.validate()?;
        Ok(DigestRef::sha256(canonical_digest(RECEIPT_DOMAIN, self)?))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionPayload {
    pub evaluation_digest: DigestRef,
    pub claim_digest: DigestRef,
    pub receipt_digest: DigestRef,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvelopeKind {
    Claim,
    Completion,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerEnvelope {
    pub format: String,
    pub ledger_id: String,
    pub sequence: u64,
    pub previous_digest: Option<DigestRef>,
    pub kind: EnvelopeKind,
    pub evaluation_digest: DigestRef,
    pub completion: Option<CompletionPayload>,
}

impl LedgerEnvelope {
    pub fn envelope_digest(&self) -> Result<DigestRef, ValidationError> {
        if self.format != ENVELOPE_FORMAT || self.sequence == 0 {
            return Err(ValidationError::InvalidField("envelope format/sequence"));
        }
        validate_logical(&self.ledger_id)?;
        self.evaluation_digest.validate()?;
        if let Some(previous) = &self.previous_digest {
            previous.validate()?;
        }
        Ok(DigestRef::sha256(canonical_digest(ENVELOPE_DOMAIN, self)?))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedEnvelope {
    pub envelope: LedgerEnvelope,
    pub namespace: String,
    pub principal: String,
    pub signature: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimToken {
    pub ledger_id: String,
    pub evaluation_digest: DigestRef,
    pub claim_digest: DigestRef,
    pub sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustRoot {
    pub principal: String,
    pub public_key_openssh: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedCheckpoint {
    pub format: String,
    pub ledger_id: String,
    pub sequence: u64,
    pub head_digest: DigestRef,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedCheckpoint {
    pub checkpoint: ProtectedCheckpoint,
    pub namespace: String,
    pub principal: String,
    pub signature: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustodyProof {
    pub format: String,
    pub ledger_id: String,
    pub trust_root: TrustRoot,
    pub checkpoint: SignedCheckpoint,
    pub claim: SignedEnvelope,
    pub completion: SignedEnvelope,
}

impl CustodyProof {
    pub fn validate_shape(&self) -> Result<(), ValidationError> {
        if self.format != PROOF_FORMAT {
            return Err(ValidationError::InvalidField("proof format"));
        }
        validate_logical(&self.ledger_id)?;
        validate_logical(&self.trust_root.principal)?;
        if self.trust_root.public_key_openssh.is_empty() {
            return Err(ValidationError::InvalidField("trust root"));
        }
        self.checkpoint.checkpoint.head_digest.validate()?;
        if self.claim.namespace != CLAIM_NAMESPACE
            || self.completion.namespace != COMPLETION_NAMESPACE
            || self.checkpoint.namespace != CHECKPOINT_NAMESPACE
        {
            return Err(ValidationError::InvalidField("signature namespace"));
        }
        Ok(())
    }
}
