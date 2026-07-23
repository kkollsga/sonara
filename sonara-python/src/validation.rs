//! Python bridge for the opt-in validation protocol.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use sonara::validation::{
    from_canonical_bytes, prepare, run_command, to_canonical_bytes, verify_custody_proof,
    BindingManifest, CommandSpec, CustodyProof, EvaluationReceipt, SqliteCustody, SshSigner,
    TrustRoot, ValidationCapsule,
};

fn invalid(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn runtime(error: impl std::fmt::Display) -> PyErr {
    PyIOError::new_err(error.to_string())
}

fn text<T: serde::Serialize>(value: &T) -> PyResult<String> {
    String::from_utf8(to_canonical_bytes(value).map_err(invalid)?)
        .map_err(|error| invalid(error.to_string()))
}

/// Return the domain-separated identity of a canonical validation capsule.
#[pyfunction]
#[pyo3(name = "_validation_capsule_digest")]
fn capsule_digest(capsule_json: &str) -> PyResult<String> {
    let capsule: ValidationCapsule =
        from_canonical_bytes(capsule_json.as_bytes()).map_err(invalid)?;
    Ok(capsule.evaluation_digest().map_err(invalid)?.value)
}

/// Validate public metadata without opening any bound resource path.
#[pyfunction]
#[pyo3(name = "_validation_prepare")]
fn validation_prepare(capsule_json: &str, bindings_json: &str) -> PyResult<String> {
    let capsule: ValidationCapsule =
        from_canonical_bytes(capsule_json.as_bytes()).map_err(invalid)?;
    let bindings: BindingManifest =
        from_canonical_bytes(bindings_json.as_bytes()).map_err(invalid)?;
    text(&prepare(&capsule, &bindings).map_err(invalid)?)
}

/// Consume a SQLite ledger, verify bound bytes, and execute an attested runner.
#[pyfunction]
#[pyo3(name = "_validation_run")]
#[allow(clippy::too_many_arguments)]
fn validation_run(
    capsule_json: &str,
    bindings_json: &str,
    command_json: &str,
    ledger_path: &str,
    ledger_id: &str,
    private_key_path: &str,
    principal: &str,
) -> PyResult<(String, String)> {
    let capsule: ValidationCapsule =
        from_canonical_bytes(capsule_json.as_bytes()).map_err(invalid)?;
    let bindings: BindingManifest =
        from_canonical_bytes(bindings_json.as_bytes()).map_err(invalid)?;
    let command: CommandSpec = from_canonical_bytes(command_json.as_bytes()).map_err(invalid)?;
    let private_key = std::fs::read(private_key_path).map_err(runtime)?;
    let signer = SshSigner::from_openssh(&private_key, principal).map_err(runtime)?;
    let mut custody = SqliteCustody::open(ledger_path, ledger_id, signer).map_err(runtime)?;
    let result = run_command(&mut custody, &capsule, &bindings, &command).map_err(runtime)?;
    Ok((text(&result.receipt)?, text(&result.proof)?))
}

/// Verify a receipt/proof against a separately pinned public trust root.
#[pyfunction]
#[pyo3(name = "_validation_verify")]
fn validation_verify(receipt_json: &str, proof_json: &str, trust_root_json: &str) -> PyResult<()> {
    let receipt: EvaluationReceipt =
        from_canonical_bytes(receipt_json.as_bytes()).map_err(invalid)?;
    let proof: CustodyProof = from_canonical_bytes(proof_json.as_bytes()).map_err(invalid)?;
    let trust_root: TrustRoot =
        from_canonical_bytes(trust_root_json.as_bytes()).map_err(invalid)?;
    verify_custody_proof(&proof, &receipt, &trust_root).map_err(invalid)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(capsule_digest, m)?)?;
    m.add_function(wrap_pyfunction!(validation_prepare, m)?)?;
    m.add_function(wrap_pyfunction!(validation_run, m)?)?;
    m.add_function(wrap_pyfunction!(validation_verify, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonara::validation::{
        ArtifactRef, CandidateIdentity, DigestRef, EvaluatorEvidence, EvaluatorKind,
        RuntimeIdentity, ValidationSuite,
    };

    fn digest(byte: char) -> DigestRef {
        DigestRef::sha256(byte.to_string().repeat(64))
    }

    #[test]
    fn capsule_digest_binding_matches_core_contract() {
        let capsule = ValidationCapsule {
            format: "sonara.validation-capsule.v1".into(),
            feature: "python-contract".into(),
            model_id: "model".into(),
            candidate: CandidateIdentity {
                commit: "a".repeat(40),
                tree: "b".repeat(40),
                binary_diff: digest('c'),
            },
            runtime: RuntimeIdentity {
                model_id: "model".into(),
                schema_version: 1,
                artifacts: Vec::new(),
            },
            suite: ValidationSuite {
                kind: "generic".into(),
                schema_version: 1,
                thresholds: Vec::new(),
                evaluator: EvaluatorEvidence {
                    id: "evaluator".into(),
                    kind: EvaluatorKind::Executable,
                    artifacts: Vec::new(),
                    output_digest: digest('d'),
                    attestation_digest: digest('e'),
                },
                controls: Vec::new(),
                tooling: Vec::new(),
            },
            artifacts: vec![ArtifactRef {
                id: "freeze".into(),
                role: "suite-freeze".into(),
                digest: digest('f'),
                size_bytes: 1,
            }],
        };
        let encoded = text(&capsule).unwrap();
        assert_eq!(
            capsule_digest(&encoded).unwrap(),
            capsule.evaluation_digest().unwrap().value
        );
    }
}
