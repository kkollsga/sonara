#![cfg(feature = "validation-ledger")]

use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};

use rand_core::OsRng;
use sha2::{Digest, Sha256};
use sonara::validation::*;
use ssh_key::{Algorithm, PrivateKey};

fn hash(bytes: &[u8]) -> DigestRef {
    DigestRef::sha256(format!("{:x}", Sha256::digest(bytes)))
}

fn artifact(id: &str, role: &str, bytes: &[u8]) -> ArtifactRef {
    ArtifactRef {
        id: id.into(),
        role: role.into(),
        digest: hash(bytes),
        size_bytes: bytes.len() as u64,
    }
}

fn signer() -> SshSigner {
    SshSigner::new(
        PrivateKey::random(&mut OsRng, Algorithm::Ed25519).unwrap(),
        "validation-custodian",
    )
    .unwrap()
}

fn fixture(runner: &[u8], input: &[u8]) -> ValidationCapsule {
    ValidationCapsule {
        format: "sonara.validation-capsule.v1".into(),
        feature: "generic-validation".into(),
        model_id: "candidate-model".into(),
        command_digest: hash(b"fixture-command"),
        candidate: CandidateIdentity {
            commit: "a".repeat(40),
            tree: "b".repeat(40),
            binary_diff: hash(b"binary-diff"),
        },
        runtime: RuntimeIdentity {
            model_id: "candidate-model".into(),
            schema_version: 1,
            artifacts: vec![artifact("runner.sh", "runtime-executable", runner)],
        },
        suite: ValidationSuite {
            kind: "generic-suite".into(),
            schema_version: 1,
            thresholds: vec![NamedDecimal {
                name: "score-min".into(),
                value: "1".into(),
            }],
            evaluator: EvaluatorEvidence {
                id: "independent-evaluator".into(),
                kind: EvaluatorKind::Executable,
                artifacts: vec![artifact("input.bin", "evaluator-input", input)],
                output_digest: hash(b"expected-output"),
                attestation_digest: hash(b"evaluator-attestation"),
            },
            controls: Vec::new(),
            tooling: Vec::new(),
        },
        artifacts: Vec::new(),
    }
}

fn bindings(runner_path: &std::path::Path, input_path: &std::path::Path) -> BindingManifest {
    BindingManifest {
        format: "sonara.validation-bindings.v1".into(),
        resources: vec![
            ResourceBinding {
                id: "runner.sh".into(),
                path: runner_path.to_string_lossy().into_owned(),
            },
            ResourceBinding {
                id: "input.bin".into(),
                path: input_path.to_string_lossy().into_owned(),
            },
        ],
    }
}

fn passing_output() -> RunnerOutput {
    RunnerOutput {
        outcome: Outcome::Pass,
        metrics: vec![NamedDecimal {
            name: "score".into(),
            value: "1".into(),
        }],
        evidence: Vec::new(),
        disclosure: DisclosureClass::AggregateOnly,
    }
}

#[test]
fn prepare_never_opens_private_bindings() {
    let capsule = fixture(b"runner", b"input");
    let missing = std::path::Path::new("/definitely/missing/private-resource");
    let prepared = prepare(&capsule, &bindings(missing, missing)).unwrap();
    assert_eq!(prepared.resource_count, 2);
    assert_eq!(
        prepared.evaluation_digest,
        capsule.evaluation_digest().unwrap()
    );
}

#[test]
fn verified_snapshot_survives_binding_mutation_and_proof_verifies_offline() {
    let root = tempfile::tempdir().unwrap();
    let runner_path = root.path().join("runner");
    let input_path = root.path().join("input");
    fs::write(&runner_path, b"runner").unwrap();
    fs::write(&input_path, b"original").unwrap();
    let capsule = fixture(b"runner", b"original");
    let manifest = bindings(&runner_path, &input_path);
    let signing_key = signer();
    let trust_root = signing_key.trust_root().unwrap();
    let mut ledger = SqliteCustody::open(
        root.path().join("ledger.db"),
        "snapshot-ledger",
        signing_key,
    )
    .unwrap();

    let result = run_with(&mut ledger, &capsule, &manifest, "runner.sh", |context| {
        fs::write(&input_path, b"mutated-after-snapshot").unwrap();
        assert_eq!(
            context.resources.bytes("input.bin"),
            Some(b"original".as_slice())
        );
        Ok(passing_output())
    })
    .unwrap();

    verify_custody_proof(&result.proof, &result.receipt, &trust_root).unwrap();
    drop(root);
    verify_custody_proof(&result.proof, &result.receipt, &trust_root).unwrap();
}

#[test]
fn resource_mismatch_consumes_authority_before_runner_entry() {
    let root = tempfile::tempdir().unwrap();
    let runner_path = root.path().join("runner");
    let input_path = root.path().join("input");
    fs::write(&runner_path, b"runner").unwrap();
    fs::write(&input_path, b"tampered").unwrap();
    let capsule = fixture(b"runner", b"expected");
    let manifest = bindings(&runner_path, &input_path);
    let signing_key = signer();
    let ledger_path = root.path().join("ledger.db");
    let mut ledger =
        SqliteCustody::open(&ledger_path, "failed-load-ledger", signing_key.clone()).unwrap();
    let mut entered = false;

    assert!(matches!(
        run_with(&mut ledger, &capsule, &manifest, "runner.sh", |_| {
            entered = true;
            Ok(passing_output())
        }),
        Err(RunnerError::ResourceMismatch(_))
    ));
    assert!(!entered);
    drop(ledger);

    let mut reopened =
        SqliteCustody::open(&ledger_path, "failed-load-ledger", signing_key).unwrap();
    assert!(matches!(
        reopened.claim(&capsule.evaluation_digest().unwrap()),
        Err(CustodyError::AlreadyClaimed)
    ));
}

#[test]
fn runner_crash_consumes_authority_and_alternate_output_cannot_replay() {
    let root = tempfile::tempdir().unwrap();
    let runner_path = root.path().join("runner");
    let input_path = root.path().join("input");
    fs::write(&runner_path, b"runner").unwrap();
    fs::write(&input_path, b"input").unwrap();
    let capsule = fixture(b"runner", b"input");
    let manifest = bindings(&runner_path, &input_path);
    let signing_key = signer();
    let ledger_path = root.path().join("ledger.db");
    let mut ledger =
        SqliteCustody::open(&ledger_path, "panic-ledger", signing_key.clone()).unwrap();

    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _ = run_with(&mut ledger, &capsule, &manifest, "runner.sh", |_| {
            panic!("simulated evaluator crash")
        });
    }));
    assert!(panicked.is_err());
    drop(ledger);

    let mut reopened = SqliteCustody::open(&ledger_path, "panic-ledger", signing_key).unwrap();
    assert!(matches!(
        run_with(&mut reopened, &capsule, &manifest, "runner.sh", |_| {
            Ok(passing_output())
        }),
        Err(RunnerError::Custody(CustodyError::AlreadyClaimed))
    ));
}

#[test]
fn bytecode_and_duplicate_or_missing_bindings_are_rejected() {
    let capsule = fixture(b"runner", b"input");
    let mut manifest = bindings(
        std::path::Path::new("/tmp/__pycache__/runner"),
        std::path::Path::new("/tmp/input.pyc"),
    );
    assert!(matches!(
        prepare(&capsule, &manifest),
        Err(RunnerError::BytecodeForbidden)
    ));

    manifest.resources[0].id = "input.bin".into();
    assert!(matches!(
        prepare(&capsule, &manifest),
        Err(RunnerError::InvalidBindings("resource coverage"))
    ));
}

#[cfg(unix)]
#[test]
fn command_runner_uses_only_verified_materialized_inputs() {
    let root = tempfile::tempdir().unwrap();
    let runner = br##"#!/bin/sh
printf '%s' '{"disclosure":"aggregate_only","evidence":[],"metrics":[{"name":"score","value":"1"}],"outcome":"pass"}'
"##;
    let input = b"sealed-input";
    let runner_path = root.path().join("runner.sh");
    let input_path = root.path().join("input.bin");
    fs::write(&runner_path, runner).unwrap();
    fs::write(&input_path, input).unwrap();
    let command = CommandSpec {
        executable_id: "runner.sh".into(),
        arguments: vec![CommandArgument::Resource("input.bin".into())],
    };
    let mut capsule = fixture(runner, input);
    capsule.command_digest = command.digest().unwrap();
    let manifest = bindings(&runner_path, &input_path);
    let mut ledger =
        SqliteCustody::open(root.path().join("ledger.db"), "command-ledger", signer()).unwrap();

    let result = run_command(&mut ledger, &capsule, &manifest, &command).unwrap();
    assert_eq!(result.receipt.outcome, Outcome::Pass);
}

#[cfg(unix)]
#[test]
fn command_substitution_is_rejected_before_execution() {
    let runner = b"runner";
    let input = b"input";
    let capsule = fixture(runner, input);
    let root = tempfile::tempdir().unwrap();
    let mut ledger =
        SqliteCustody::open(root.path().join("ledger.db"), "command-binding", signer()).unwrap();
    let error = match run_command(
        &mut ledger,
        &capsule,
        &bindings(root.path(), root.path()),
        &CommandSpec {
            executable_id: "runner.sh".into(),
            arguments: Vec::new(),
        },
    ) {
        Err(error) => error,
        Ok(_) => panic!("substituted command was accepted"),
    };
    assert!(matches!(
        error,
        RunnerError::InvalidBindings("command digest")
    ));
}
