use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Component, Path};
use std::process::Command;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::{
    canonical_digest, from_canonical_bytes, ArtifactRef, ClaimToken, CustodyError, CustodyProof,
    CustodyProvider, DigestRef, DisclosureClass, EvaluationReceipt, NamedDecimal, Outcome,
    ValidationCapsule, ValidationError,
};

const BINDING_FORMAT: &str = "sonara.validation-bindings.v1";
const PREPARED_FORMAT: &str = "sonara.prepared-validation.v1";
const COMMAND_DOMAIN: &str = "sonara-validation-command-v1";

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error(transparent)]
    Validation(#[from] ValidationError),
    #[error(transparent)]
    Custody(#[from] CustodyError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("binding manifest is invalid: {0}")]
    InvalidBindings(&'static str),
    #[error("resource content does not match capsule: {0}")]
    ResourceMismatch(String),
    #[error("Python bytecode is forbidden in a validation runtime")]
    BytecodeForbidden,
    #[error("runner process failed with status {0}")]
    ProcessFailed(String),
    #[error("runner emitted invalid canonical output")]
    InvalidRunnerOutput,
    #[error("runner output violates the validation receipt contract")]
    InvalidReceiptOutput,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceBinding {
    pub id: String,
    pub path: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BindingManifest {
    pub format: String,
    pub resources: Vec<ResourceBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedValidation {
    pub format: String,
    pub evaluation_digest: DigestRef,
    pub resource_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerOutput {
    pub outcome: Outcome,
    pub metrics: Vec<NamedDecimal>,
    pub evidence: Vec<ArtifactRef>,
    pub disclosure: DisclosureClass,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum CommandArgument {
    Literal(String),
    Resource(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandSpec {
    pub executable_id: String,
    pub arguments: Vec<CommandArgument>,
}

impl CommandSpec {
    pub fn digest(&self) -> Result<DigestRef, ValidationError> {
        Ok(DigestRef::sha256(canonical_digest(COMMAND_DOMAIN, self)?))
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedResources {
    identities: BTreeMap<String, ArtifactRef>,
    bytes: BTreeMap<String, Vec<u8>>,
}

impl VerifiedResources {
    pub fn bytes(&self, id: &str) -> Option<&[u8]> {
        self.bytes.get(id).map(Vec::as_slice)
    }

    pub fn identity(&self, id: &str) -> Option<&ArtifactRef> {
        self.identities.get(id)
    }

    pub fn runtime_digest(&self) -> Result<DigestRef, ValidationError> {
        let identities = self.identities.values().collect::<Vec<_>>();
        Ok(DigestRef::sha256(canonical_digest(
            "sonara-verified-runtime-v1",
            &identities,
        )?))
    }
}

pub struct RunContext<'a> {
    pub token: &'a ClaimToken,
    pub resources: &'a VerifiedResources,
}

pub struct RunArtifacts {
    pub receipt: EvaluationReceipt,
    pub proof: CustodyProof,
}

fn all_artifacts(capsule: &ValidationCapsule) -> Vec<&ArtifactRef> {
    capsule
        .runtime
        .artifacts
        .iter()
        .chain(capsule.suite.evaluator.artifacts.iter())
        .chain(capsule.suite.controls.iter())
        .chain(capsule.suite.tooling.iter())
        .chain(capsule.artifacts.iter())
        .collect()
}

fn safe_binding_path(path: &Path) -> bool {
    path.components().all(|component| {
        !matches!(component, Component::ParentDir)
            && component.as_os_str().to_str() != Some("__pycache__")
    }) && path.extension().and_then(|value| value.to_str()) != Some("pyc")
}

pub fn prepare(
    capsule: &ValidationCapsule,
    bindings: &BindingManifest,
) -> Result<PreparedValidation, RunnerError> {
    capsule.validate()?;
    if bindings.format != BINDING_FORMAT {
        return Err(RunnerError::InvalidBindings("format"));
    }
    let required = all_artifacts(capsule)
        .into_iter()
        .map(|artifact| artifact.id.as_str())
        .collect::<HashSet<_>>();
    let supplied = bindings
        .resources
        .iter()
        .map(|binding| binding.id.as_str())
        .collect::<HashSet<_>>();
    if required != supplied || supplied.len() != bindings.resources.len() {
        return Err(RunnerError::InvalidBindings("resource coverage"));
    }
    if bindings
        .resources
        .iter()
        .any(|binding| binding.path.is_empty() || !safe_binding_path(Path::new(&binding.path)))
    {
        return Err(RunnerError::BytecodeForbidden);
    }
    Ok(PreparedValidation {
        format: PREPARED_FORMAT.into(),
        evaluation_digest: capsule.evaluation_digest()?,
        resource_count: required.len() as u64,
    })
}

fn verify_and_load(
    capsule: &ValidationCapsule,
    bindings: &BindingManifest,
) -> Result<VerifiedResources, RunnerError> {
    prepare(capsule, bindings)?;
    let expected = all_artifacts(capsule)
        .into_iter()
        .map(|artifact| (artifact.id.as_str(), artifact))
        .collect::<HashMap<_, _>>();
    let mut identities = BTreeMap::new();
    let mut bytes = BTreeMap::new();
    for binding in &bindings.resources {
        let path = Path::new(&binding.path);
        if !safe_binding_path(path) {
            return Err(RunnerError::BytecodeForbidden);
        }
        let content = std::fs::read(path)?;
        let artifact = expected[&binding.id.as_str()];
        let digest = format!("{:x}", Sha256::digest(&content));
        if artifact.digest.algorithm != "sha256"
            || artifact.digest.value != digest
            || artifact.size_bytes != content.len() as u64
        {
            return Err(RunnerError::ResourceMismatch(binding.id.clone()));
        }
        identities.insert(binding.id.clone(), artifact.clone());
        bytes.insert(binding.id.clone(), content);
    }
    Ok(VerifiedResources { identities, bytes })
}

pub fn run_with<P, F>(
    provider: &mut P,
    capsule: &ValidationCapsule,
    bindings: &BindingManifest,
    runner_id: &str,
    runner: F,
) -> Result<RunArtifacts, RunnerError>
where
    P: CustodyProvider,
    F: FnOnce(RunContext<'_>) -> Result<RunnerOutput, RunnerError>,
{
    let evaluation_digest = capsule.evaluation_digest()?;
    // This commit is deliberately before resource loading and runner entry.
    let token = provider.claim(&evaluation_digest)?;
    let resources = verify_and_load(capsule, bindings)?;
    let runner_digest = resources
        .identity(runner_id)
        .ok_or(RunnerError::InvalidBindings("runner id"))?
        .digest
        .clone();
    let runtime_digest = resources.runtime_digest()?;
    let output = runner(RunContext {
        token: &token,
        resources: &resources,
    })?;
    let receipt = EvaluationReceipt {
        format: "sonara.evaluation-receipt.v1".into(),
        evaluation_digest,
        claim_digest: token.claim_digest.clone(),
        runtime_digest,
        runner_digest,
        outcome: output.outcome,
        metrics: output.metrics,
        evidence: output.evidence,
        disclosure: output.disclosure,
    };
    receipt
        .validate()
        .map_err(|_| RunnerError::InvalidReceiptOutput)?;
    let proof = provider.complete(&token, &receipt)?;
    Ok(RunArtifacts { receipt, proof })
}

fn materialize(resources: &VerifiedResources, root: &Path) -> Result<(), RunnerError> {
    for (id, content) in &resources.bytes {
        let relative = Path::new(id);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| matches!(component, Component::RootDir | Component::Prefix(_)))
            || !safe_binding_path(relative)
        {
            return Err(RunnerError::InvalidBindings("unsafe logical resource id"));
        }
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(content)?;
        file.sync_all()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o500))?;
        }
    }
    Ok(())
}

fn contains_bytecode(root: &Path) -> Result<bool, std::io::Error> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().and_then(|value| value.to_str()) == Some("__pycache__") {
                    return Ok(true);
                }
                pending.push(path);
            } else if path.extension().and_then(|value| value.to_str()) == Some("pyc") {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub fn run_command<P: CustodyProvider>(
    provider: &mut P,
    capsule: &ValidationCapsule,
    bindings: &BindingManifest,
    command: &CommandSpec,
) -> Result<RunArtifacts, RunnerError> {
    if command.digest()? != capsule.command_digest {
        return Err(RunnerError::InvalidBindings("command digest"));
    }
    run_with(
        provider,
        capsule,
        bindings,
        &command.executable_id,
        |context| {
            let directory = tempfile::Builder::new()
                .prefix("sonara-validation-")
                .tempdir()?;
            materialize(context.resources, directory.path())?;
            if contains_bytecode(directory.path())? {
                return Err(RunnerError::BytecodeForbidden);
            }
            let executable = directory.path().join(&command.executable_id);
            let mut process = Command::new(executable);
            for argument in &command.arguments {
                match argument {
                    CommandArgument::Literal(value) => {
                        process.arg(value);
                    }
                    CommandArgument::Resource(id) => {
                        if context.resources.bytes(id).is_none() {
                            return Err(RunnerError::InvalidBindings("command resource"));
                        }
                        process.arg(directory.path().join(id));
                    }
                }
            }
            let output = process
                .current_dir(directory.path())
                .env("PYTHONDONTWRITEBYTECODE", "1")
                .env("PYTHONNOUSERSITE", "1")
                .env(
                    "SONARA_VALIDATION_EVALUATION_DIGEST",
                    &context.token.evaluation_digest.value,
                )
                .env(
                    "SONARA_VALIDATION_CLAIM_DIGEST",
                    &context.token.claim_digest.value,
                )
                .output()?;
            if contains_bytecode(directory.path())? {
                return Err(RunnerError::BytecodeForbidden);
            }
            if !output.status.success() {
                return Err(RunnerError::ProcessFailed(output.status.to_string()));
            }
            from_canonical_bytes(&output.stdout).map_err(|_| RunnerError::InvalidRunnerOutput)
        },
    )
}
