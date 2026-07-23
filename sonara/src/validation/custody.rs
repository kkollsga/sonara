use ssh_key::{HashAlg, LineEnding, PrivateKey, PublicKey, SshSig};
use thiserror::Error;

use super::{
    from_canonical_bytes, to_canonical_bytes, ClaimToken, CustodyProof, DigestRef, EnvelopeKind,
    EvaluationReceipt, LedgerEnvelope, ProtectedCheckpoint, SignedCheckpoint, SignedEnvelope,
    TrustRoot, ValidationError, CHECKPOINT_NAMESPACE, CLAIM_NAMESPACE, COMPLETION_NAMESPACE,
};

#[derive(Debug, Error)]
pub enum CustodyError {
    #[error(transparent)]
    Validation(#[from] ValidationError),
    #[error(transparent)]
    Ssh(#[from] ssh_key::Error),
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
    #[error("evaluation authority has already been consumed")]
    AlreadyClaimed,
    #[error("custody ledger is already complete")]
    AlreadyCompleted,
    #[error("claim token does not name the protected ledger head")]
    StaleClaim,
    #[error("custody ledger head compare-and-swap failed")]
    HeadChanged,
    #[error("invalid custody proof: {0}")]
    InvalidProof(&'static str),
    #[error("custody ledger identity mismatch")]
    LedgerIdentity,
}

#[derive(Clone)]
pub struct SshSigner {
    private_key: PrivateKey,
    principal: String,
}

impl SshSigner {
    pub fn new(
        private_key: PrivateKey,
        principal: impl Into<String>,
    ) -> Result<Self, CustodyError> {
        let principal = principal.into();
        if principal.is_empty() {
            return Err(CustodyError::InvalidProof("empty signer principal"));
        }
        Ok(Self {
            private_key,
            principal,
        })
    }

    pub fn from_openssh(
        private_key: impl AsRef<[u8]>,
        principal: impl Into<String>,
    ) -> Result<Self, CustodyError> {
        Self::new(PrivateKey::from_openssh(private_key)?, principal)
    }

    pub fn trust_root(&self) -> Result<TrustRoot, CustodyError> {
        Ok(TrustRoot {
            principal: self.principal.clone(),
            public_key_openssh: self.private_key.public_key().to_openssh()?,
        })
    }

    pub fn sign(&self, envelope: LedgerEnvelope) -> Result<SignedEnvelope, CustodyError> {
        let namespace = match envelope.kind {
            EnvelopeKind::Claim => CLAIM_NAMESPACE,
            EnvelopeKind::Completion => COMPLETION_NAMESPACE,
        };
        let bytes = to_canonical_bytes(&envelope)?;
        let signature = self
            .private_key
            .sign(namespace, HashAlg::Sha512, &bytes)?
            .to_pem(LineEnding::LF)?;
        Ok(SignedEnvelope {
            envelope,
            namespace: namespace.into(),
            principal: self.principal.clone(),
            signature,
        })
    }

    pub fn sign_checkpoint(
        &self,
        checkpoint: ProtectedCheckpoint,
    ) -> Result<SignedCheckpoint, CustodyError> {
        let bytes = to_canonical_bytes(&checkpoint)?;
        let signature = self
            .private_key
            .sign(CHECKPOINT_NAMESPACE, HashAlg::Sha512, &bytes)?
            .to_pem(LineEnding::LF)?;
        Ok(SignedCheckpoint {
            checkpoint,
            namespace: CHECKPOINT_NAMESPACE.into(),
            principal: self.principal.clone(),
            signature,
        })
    }
}

pub trait CustodyProvider {
    fn claim(&mut self, evaluation_digest: &DigestRef) -> Result<ClaimToken, CustodyError>;
    fn complete(
        &mut self,
        token: &ClaimToken,
        receipt: &EvaluationReceipt,
    ) -> Result<CustodyProof, CustodyError>;
}

fn verify_signature(signed: &SignedEnvelope, root: &TrustRoot) -> Result<(), CustodyError> {
    if signed.principal != root.principal {
        return Err(CustodyError::InvalidProof("signer principal mismatch"));
    }
    let expected_namespace = match signed.envelope.kind {
        EnvelopeKind::Claim => CLAIM_NAMESPACE,
        EnvelopeKind::Completion => COMPLETION_NAMESPACE,
    };
    if signed.namespace != expected_namespace {
        return Err(CustodyError::InvalidProof("signature namespace mismatch"));
    }
    // Requiring canonical round-trip prevents alternate JSON encodings from
    // becoming a second identity for the same signed envelope.
    let bytes = to_canonical_bytes(&signed.envelope)?;
    let _: LedgerEnvelope = from_canonical_bytes(&bytes)?;
    let public_key = PublicKey::from_openssh(&root.public_key_openssh)?;
    let signature = SshSig::from_pem(&signed.signature)?;
    public_key.verify(expected_namespace, &bytes, &signature)?;
    Ok(())
}

fn verify_checkpoint(signed: &SignedCheckpoint, root: &TrustRoot) -> Result<(), CustodyError> {
    if signed.principal != root.principal || signed.namespace != CHECKPOINT_NAMESPACE {
        return Err(CustodyError::InvalidProof(
            "checkpoint signer/namespace mismatch",
        ));
    }
    let bytes = to_canonical_bytes(&signed.checkpoint)?;
    let _: ProtectedCheckpoint = from_canonical_bytes(&bytes)?;
    let public_key = PublicKey::from_openssh(&root.public_key_openssh)?;
    let signature = SshSig::from_pem(&signed.signature)?;
    public_key.verify(CHECKPOINT_NAMESPACE, &bytes, &signature)?;
    Ok(())
}

pub fn verify_custody_proof(
    proof: &CustodyProof,
    receipt: &EvaluationReceipt,
    expected_root: &TrustRoot,
) -> Result<(), CustodyError> {
    proof.validate_shape()?;
    receipt.validate()?;
    if &proof.trust_root != expected_root {
        return Err(CustodyError::InvalidProof("unpinned trust root"));
    }
    if proof.ledger_id != proof.claim.envelope.ledger_id
        || proof.ledger_id != proof.completion.envelope.ledger_id
    {
        return Err(CustodyError::InvalidProof("ledger mismatch"));
    }
    verify_signature(&proof.claim, &proof.trust_root)?;
    verify_signature(&proof.completion, &proof.trust_root)?;
    verify_checkpoint(&proof.checkpoint, &proof.trust_root)?;
    let claim = &proof.claim.envelope;
    let completion = &proof.completion.envelope;
    if claim.kind != EnvelopeKind::Claim
        || claim.sequence != 1
        || claim.previous_digest.is_some()
        || claim.completion.is_some()
    {
        return Err(CustodyError::InvalidProof("claim is not genesis"));
    }
    let claim_digest = claim.envelope_digest()?;
    if completion.kind != EnvelopeKind::Completion
        || completion.sequence != claim.sequence + 1
        || completion.previous_digest.as_ref() != Some(&claim_digest)
    {
        return Err(CustodyError::InvalidProof("completion is not exact next"));
    }
    let completion_payload = completion
        .completion
        .as_ref()
        .ok_or(CustodyError::InvalidProof("completion payload missing"))?;
    let receipt_digest = receipt.receipt_digest()?;
    if claim.evaluation_digest != receipt.evaluation_digest
        || completion.evaluation_digest != receipt.evaluation_digest
        || completion_payload.evaluation_digest != receipt.evaluation_digest
        || completion_payload.claim_digest != claim_digest
        || completion_payload.receipt_digest != receipt_digest
        || receipt.claim_digest != claim_digest
    {
        return Err(CustodyError::InvalidProof("receipt chain mismatch"));
    }
    if proof.checkpoint.checkpoint.format != "sonara.protected-checkpoint.v1"
        || proof.checkpoint.checkpoint.ledger_id != proof.ledger_id
        || proof.checkpoint.checkpoint.sequence != completion.sequence
        || proof.checkpoint.checkpoint.head_digest != completion.envelope_digest()?
    {
        return Err(CustodyError::InvalidProof("protected head mismatch"));
    }
    Ok(())
}
