//! Feature-neutral validation capsules and receipts.
//!
//! This module is opt-in and is not compiled by Sonara's default feature set.

mod canonical;
mod schema;

pub use canonical::{canonical_digest, from_canonical_bytes, to_canonical_bytes};
pub use schema::{
    ArtifactRef, CandidateIdentity, CompletionPayload, CustodyProof, DigestRef, DisclosureClass,
    EnvelopeKind, EvaluationReceipt, EvaluatorEvidence, EvaluatorKind, LedgerEnvelope,
    NamedDecimal, Outcome, RuntimeIdentity, SignedEnvelope, ValidationCapsule, ValidationError,
    ValidationSuite, CLAIM_NAMESPACE, COMPLETION_NAMESPACE,
};
