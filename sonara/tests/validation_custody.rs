#![cfg(feature = "validation-ledger")]

use std::sync::{Arc, Barrier};

use rand_core::OsRng;
use sonara::validation::*;
use ssh_key::{Algorithm, PrivateKey};

fn digest(byte: char) -> DigestRef {
    DigestRef::sha256(byte.to_string().repeat(64))
}

fn signer() -> SshSigner {
    SshSigner::new(
        PrivateKey::random(&mut OsRng, Algorithm::Ed25519).unwrap(),
        "independent-custodian",
    )
    .unwrap()
}

fn receipt(token: &ClaimToken) -> EvaluationReceipt {
    EvaluationReceipt {
        format: "sonara.evaluation-receipt.v1".into(),
        evaluation_digest: token.evaluation_digest.clone(),
        claim_digest: token.claim_digest.clone(),
        runtime_digest: digest('b'),
        runner_digest: digest('c'),
        outcome: Outcome::Pass,
        metrics: vec![NamedDecimal {
            name: "accuracy".into(),
            value: "0.9".into(),
        }],
        evidence: Vec::new(),
        disclosure: DisclosureClass::AggregateOnly,
    }
}

#[test]
fn signed_claim_completion_roundtrip_verifies_offline() {
    let root = tempfile::tempdir().unwrap();
    let mut ledger =
        SqliteCustody::open(root.path().join("ledger.db"), "audit-1", signer()).unwrap();
    let token = ledger.claim(&digest('a')).unwrap();
    let receipt = receipt(&token);
    let proof = ledger.complete(&token, &receipt).unwrap();
    verify_custody_proof(&proof, &receipt, &proof.trust_root).unwrap();
}

#[test]
fn dropped_process_state_does_not_restore_consumed_authority() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("ledger.db");
    let signing_key = signer();
    {
        let mut ledger = SqliteCustody::open(&path, "audit-crash", signing_key.clone()).unwrap();
        ledger.claim(&digest('a')).unwrap();
        // Dropping immediately after the committed claim models process loss
        // before any result/output is written.
    }
    let mut reopened = SqliteCustody::open(&path, "audit-crash", signing_key).unwrap();
    assert!(matches!(
        reopened.claim(&digest('a')),
        Err(CustodyError::AlreadyClaimed)
    ));
    assert!(matches!(
        reopened.claim(&digest('d')),
        Err(CustodyError::AlreadyClaimed)
    ));
}

#[test]
fn concurrent_claim_has_exactly_one_winner() {
    let root = tempfile::tempdir().unwrap();
    let path = Arc::new(root.path().join("ledger.db"));
    let barrier = Arc::new(Barrier::new(2));
    let signing_key = signer();
    // Initialize schema before racing independent connections.
    drop(SqliteCustody::open(&*path, "audit-race", signing_key.clone()).unwrap());
    let mut threads = Vec::new();
    for _ in 0..2 {
        let path = Arc::clone(&path);
        let barrier = Arc::clone(&barrier);
        let signing_key = signing_key.clone();
        threads.push(std::thread::spawn(move || {
            let mut ledger = SqliteCustody::open(&*path, "audit-race", signing_key).unwrap();
            barrier.wait();
            ledger.claim(&digest('a')).is_ok()
        }));
    }
    let winners = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .filter(|won| *won)
        .count();
    assert_eq!(winners, 1);
}

#[test]
fn proof_rejects_gap_namespace_signature_and_result_substitution() {
    let root = tempfile::tempdir().unwrap();
    let mut ledger =
        SqliteCustody::open(root.path().join("ledger.db"), "audit-2", signer()).unwrap();
    let token = ledger.claim(&digest('a')).unwrap();
    let receipt = receipt(&token);
    let proof = ledger.complete(&token, &receipt).unwrap();

    let mut gap = proof.clone();
    gap.completion.envelope.sequence = 3;
    assert!(verify_custody_proof(&gap, &receipt, &proof.trust_root).is_err());

    let mut namespace = proof.clone();
    namespace.completion.namespace = CLAIM_NAMESPACE.into();
    assert!(verify_custody_proof(&namespace, &receipt, &proof.trust_root).is_err());

    let mut unsigned = proof.clone();
    unsigned.claim.signature.clear();
    assert!(verify_custody_proof(&unsigned, &receipt, &proof.trust_root).is_err());

    let mut stale_head = proof.clone();
    stale_head.checkpoint.checkpoint.head_digest = digest('8');
    assert!(verify_custody_proof(&stale_head, &receipt, &proof.trust_root).is_err());

    let mut substituted = receipt.clone();
    substituted.metrics[0].value = "1.0".into();
    assert!(verify_custody_proof(&proof, &substituted, &proof.trust_root).is_err());

    let wrong_root = signer().trust_root().unwrap();
    assert!(verify_custody_proof(&proof, &receipt, &wrong_root).is_err());
}

#[test]
fn stale_token_and_double_completion_fail() {
    let root = tempfile::tempdir().unwrap();
    let mut ledger =
        SqliteCustody::open(root.path().join("ledger.db"), "audit-3", signer()).unwrap();
    let token = ledger.claim(&digest('a')).unwrap();
    let mut stale = token.clone();
    stale.claim_digest = digest('9');
    assert!(matches!(
        ledger.complete(&stale, &receipt(&stale)),
        Err(CustodyError::StaleClaim)
    ));
    let receipt = receipt(&token);
    ledger.complete(&token, &receipt).unwrap();
    assert!(matches!(
        ledger.complete(&token, &receipt),
        Err(CustodyError::AlreadyCompleted)
    ));
}
