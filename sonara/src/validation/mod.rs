//! Feature-neutral validation capsules and receipts.
//!
//! This module is opt-in and is not compiled by Sonara's default feature set.

mod canonical;
#[cfg(feature = "validation-ledger")]
mod custody;
#[cfg(feature = "validation-ledger")]
mod ledger;
mod schema;

pub use canonical::{canonical_digest, from_canonical_bytes, to_canonical_bytes};
#[cfg(feature = "validation-ledger")]
pub use custody::{verify_custody_proof, CustodyError, CustodyProvider, SshSigner};
#[cfg(feature = "validation-ledger")]
pub use ledger::SqliteCustody;
pub use schema::{
    ArtifactRef, CandidateIdentity, ClaimToken, CompletionPayload, CustodyProof, DigestRef,
    DisclosureClass, EnvelopeKind, EvaluationReceipt, EvaluatorEvidence, EvaluatorKind,
    LedgerEnvelope, NamedDecimal, Outcome, ProtectedCheckpoint, RuntimeIdentity, SignedCheckpoint,
    SignedEnvelope, TrustRoot, ValidationCapsule, ValidationError, ValidationSuite,
    CHECKPOINT_NAMESPACE, CLAIM_NAMESPACE, COMPLETION_NAMESPACE,
};
