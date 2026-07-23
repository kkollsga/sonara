#![cfg(feature = "validation")]

use sonara::validation::*;

fn digest(byte: char) -> DigestRef {
    DigestRef::sha256(byte.to_string().repeat(64))
}

fn artifact(id: &str, role: &str, byte: char) -> ArtifactRef {
    ArtifactRef {
        id: id.into(),
        role: role.into(),
        digest: digest(byte),
        size_bytes: 42,
    }
}

fn capsule() -> ValidationCapsule {
    ValidationCapsule {
        format: "sonara.validation-capsule.v1".into(),
        feature: "aggression".into(),
        model_id: "aggression-rank-v2".into(),
        candidate: CandidateIdentity {
            commit: "a".repeat(40),
            tree: "b".repeat(40),
            binary_diff: digest('c'),
        },
        runtime: RuntimeIdentity {
            model_id: "aggression-rank-v2".into(),
            schema_version: 5,
            artifacts: vec![artifact("native-module", "runtime-native", 'd')],
        },
        suite: ValidationSuite {
            kind: "pairwise-rank".into(),
            schema_version: 1,
            thresholds: vec![NamedDecimal {
                name: "spearman-min".into(),
                value: "0.65".into(),
            }],
            evaluator: EvaluatorEvidence {
                id: "independent-rater".into(),
                kind: EvaluatorKind::Executable,
                artifacts: vec![artifact("rater", "evaluator-executable", 'e')],
                output_digest: digest('f'),
                attestation_digest: digest('1'),
            },
            controls: vec![artifact("gain-controls", "control-manifest", '2')],
            tooling: vec![artifact("runner", "suite-tooling", '3')],
        },
        artifacts: vec![artifact("freeze", "suite-freeze", '4')],
    }
}

#[test]
fn capsule_digest_binds_actual_freeze_bytes() {
    let original = capsule();
    let mut changed = original.clone();
    changed.artifacts[0].digest = digest('5');
    assert_ne!(
        original.evaluation_digest().unwrap(),
        changed.evaluation_digest().unwrap()
    );
}

#[test]
fn canonical_roundtrip_is_byte_exact() {
    let value = capsule();
    let bytes = to_canonical_bytes(&value).unwrap();
    let decoded: ValidationCapsule = from_canonical_bytes(&bytes).unwrap();
    assert_eq!(decoded, value);

    let mut spaced = bytes.clone();
    spaced.push(b'\n');
    assert!(from_canonical_bytes::<ValidationCapsule>(&spaced).is_err());
}

#[test]
fn duplicate_and_unknown_fields_are_rejected() {
    let bytes = to_canonical_bytes(&capsule()).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    let duplicate = text.replacen(
        "\"feature\":\"aggression\"",
        "\"feature\":\"aggression\",\"feature\":\"other\"",
        1,
    );
    assert!(serde_json::from_str::<ValidationCapsule>(&duplicate).is_err());

    let unknown = text.replacen('{', "{\"unknown\":1,", 1);
    assert!(serde_json::from_str::<ValidationCapsule>(&unknown).is_err());
}

#[test]
fn private_paths_and_duplicate_artifacts_are_rejected() {
    let mut value = capsule();
    value.artifacts[0].id = "/private/audio.wav".into();
    assert!(matches!(
        value.validate(),
        Err(ValidationError::PrivatePath)
    ));

    let mut value = capsule();
    value.artifacts.push(value.artifacts[0].clone());
    assert!(matches!(
        value.validate(),
        Err(ValidationError::DuplicateIdentity(_))
    ));
}

#[test]
fn decimal_identity_rejects_exponents_and_non_finite_values() {
    for invalid in ["1e-3", "NaN", "inf", "+1", "01", "1."] {
        let value = NamedDecimal {
            name: "threshold".into(),
            value: invalid.into(),
        };
        assert!(value.validate().is_err(), "accepted {invalid}");
    }
}

#[test]
fn signature_roles_are_distinct() {
    assert_ne!(CLAIM_NAMESPACE, COMPLETION_NAMESPACE);
}
