#![cfg(all(feature = "validation-cli", unix))]

use std::fs;
use std::process::Command;

use rand_core::OsRng;
use sha2::{Digest, Sha256};
use sonara::validation::*;
use ssh_key::{Algorithm, LineEnding, PrivateKey};

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

fn capsule(runner: &[u8], input: &[u8]) -> ValidationCapsule {
    let command = CommandSpec {
        executable_id: "runner.sh".into(),
        arguments: vec![CommandArgument::Resource("input.bin".into())],
    };
    ValidationCapsule {
        format: "sonara.validation-capsule.v1".into(),
        feature: "cli-contract".into(),
        model_id: "cli-model".into(),
        command_digest: command.digest().unwrap(),
        candidate: CandidateIdentity {
            commit: "a".repeat(40),
            tree: "b".repeat(40),
            binary_diff: hash(b"diff"),
        },
        runtime: RuntimeIdentity {
            model_id: "cli-model".into(),
            schema_version: 1,
            artifacts: vec![artifact("runner.sh", "runtime-executable", runner)],
        },
        suite: ValidationSuite {
            kind: "cli-suite".into(),
            schema_version: 1,
            thresholds: Vec::new(),
            evaluator: EvaluatorEvidence {
                id: "cli-evaluator".into(),
                kind: EvaluatorKind::Executable,
                artifacts: vec![artifact("input.bin", "evaluator-input", input)],
                output_digest: hash(b"output"),
                attestation_digest: hash(b"attestation"),
            },
            controls: Vec::new(),
            tooling: Vec::new(),
        },
        artifacts: Vec::new(),
    }
}

#[test]
fn prepare_run_verify_share_the_rust_contract() {
    let root = tempfile::tempdir().unwrap();
    let runner = br##"#!/bin/sh
printf '%s' '{"disclosure":"aggregate_only","evidence":[],"metrics":[{"name":"cli-score","value":"1"}],"outcome":"pass"}'
"##;
    let input = b"private-evaluation-input";
    let runner_path = root.path().join("runner.sh");
    let input_path = root.path().join("input.bin");
    fs::write(&runner_path, runner).unwrap();
    fs::write(&input_path, input).unwrap();

    let capsule_path = root.path().join("capsule.json");
    fs::write(
        &capsule_path,
        to_canonical_bytes(&capsule(runner, input)).unwrap(),
    )
    .unwrap();
    let bindings = BindingManifest {
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
    };
    let bindings_path = root.path().join("bindings.json");
    fs::write(&bindings_path, serde_json::to_vec(&bindings).unwrap()).unwrap();
    let command_path = root.path().join("command.json");
    fs::write(
        &command_path,
        serde_json::to_vec(&CommandSpec {
            executable_id: "runner.sh".into(),
            arguments: vec![CommandArgument::Resource("input.bin".into())],
        })
        .unwrap(),
    )
    .unwrap();

    let binary = env!("CARGO_BIN_EXE_sonara");
    let prepared = Command::new(binary)
        .args([
            "validate",
            "prepare",
            "--capsule",
            capsule_path.to_str().unwrap(),
            "--bindings",
            bindings_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        prepared.status.success(),
        "{}",
        String::from_utf8_lossy(&prepared.stderr)
    );
    let prepared: PreparedValidation = from_canonical_bytes(&prepared.stdout).unwrap();
    assert_eq!(prepared.resource_count, 2);

    let private_key = PrivateKey::random(&mut OsRng, Algorithm::Ed25519).unwrap();
    let signer = SshSigner::new(private_key.clone(), "cli-custodian").unwrap();
    let key_path = root.path().join("custodian-key");
    private_key
        .write_openssh_file(&key_path, LineEnding::LF)
        .unwrap();
    let trust_path = root.path().join("trust-root.json");
    fs::write(
        &trust_path,
        to_canonical_bytes(&signer.trust_root().unwrap()).unwrap(),
    )
    .unwrap();
    let receipt_path = root.path().join("receipt.json");
    let proof_path = root.path().join("proof.json");
    let ledger_path = root.path().join("ledger.db");

    let run = Command::new(binary)
        .args([
            "validate",
            "run",
            "--capsule",
            capsule_path.to_str().unwrap(),
            "--bindings",
            bindings_path.to_str().unwrap(),
            "--command",
            command_path.to_str().unwrap(),
            "--ledger",
            ledger_path.to_str().unwrap(),
            "--ledger-id",
            "cli-ledger",
            "--private-key",
            key_path.to_str().unwrap(),
            "--principal",
            "cli-custodian",
            "--receipt",
            receipt_path.to_str().unwrap(),
            "--proof",
            proof_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let verified = Command::new(binary)
        .args([
            "validate",
            "verify",
            "--receipt",
            receipt_path.to_str().unwrap(),
            "--proof",
            proof_path.to_str().unwrap(),
            "--trust-root",
            trust_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );

    // Neither a new output filename nor a different result can restore the
    // already consumed one-evaluation ledger.
    let replay_receipt = root.path().join("alternate-receipt.json");
    let replay_proof = root.path().join("alternate-proof.json");
    let replay = Command::new(binary)
        .args([
            "validate",
            "run",
            "--capsule",
            capsule_path.to_str().unwrap(),
            "--bindings",
            bindings_path.to_str().unwrap(),
            "--command",
            command_path.to_str().unwrap(),
            "--ledger",
            ledger_path.to_str().unwrap(),
            "--ledger-id",
            "cli-ledger",
            "--private-key",
            key_path.to_str().unwrap(),
            "--principal",
            "cli-custodian",
            "--receipt",
            replay_receipt.to_str().unwrap(),
            "--proof",
            replay_proof.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!replay.status.success());
    assert!(!replay_receipt.exists());
    assert!(!replay_proof.exists());
}
