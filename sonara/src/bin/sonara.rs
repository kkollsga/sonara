use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use sonara::validation::{
    from_canonical_bytes, prepare, run_command, to_canonical_bytes, verify_custody_proof,
    BindingManifest, CommandSpec, CustodyProof, EvaluationReceipt, SqliteCustody, SshSigner,
    TrustRoot, ValidationCapsule,
};
use ssh_key::PrivateKey;

#[derive(Parser)]
#[command(name = "sonara", about = "Sonara audio analysis and validation tools")]
struct Cli {
    #[command(subcommand)]
    command: TopLevel,
}

#[derive(Subcommand)]
enum TopLevel {
    /// Prepare, execute, or verify a content-addressed validation.
    Validate {
        #[command(subcommand)]
        command: ValidateCommand,
    },
}

#[derive(Subcommand)]
enum ValidateCommand {
    /// Validate public metadata without opening any bound resource.
    Prepare {
        #[arg(long)]
        capsule: PathBuf,
        #[arg(long)]
        bindings: PathBuf,
    },
    /// Consume authority, verify bound bytes, then execute the runner.
    Run {
        #[arg(long)]
        capsule: PathBuf,
        #[arg(long)]
        bindings: PathBuf,
        #[arg(long)]
        command: PathBuf,
        #[arg(long)]
        ledger: PathBuf,
        #[arg(long)]
        ledger_id: String,
        #[arg(long)]
        private_key: PathBuf,
        #[arg(long)]
        principal: String,
        #[arg(long)]
        receipt: PathBuf,
        #[arg(long)]
        proof: PathBuf,
    },
    /// Verify a completed evaluation using a caller-pinned trust root.
    Verify {
        #[arg(long)]
        receipt: PathBuf,
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        trust_root: PathBuf,
    },
}

fn read_canonical<T: serde::de::DeserializeOwned + serde::Serialize>(
    path: &Path,
) -> Result<T, Box<dyn std::error::Error>> {
    Ok(from_canonical_bytes(&std::fs::read(path)?)?)
}

fn write_exclusive<T: serde::Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = to_canonical_bytes(value)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn execute(command: ValidateCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ValidateCommand::Prepare { capsule, bindings } => {
            let capsule = read_canonical::<ValidationCapsule>(&capsule)?;
            let bindings: BindingManifest = serde_json::from_slice(&std::fs::read(bindings)?)?;
            let prepared = prepare(&capsule, &bindings)?;
            std::io::stdout().write_all(&to_canonical_bytes(&prepared)?)?;
        }
        ValidateCommand::Run {
            capsule,
            bindings,
            command,
            ledger,
            ledger_id,
            private_key,
            principal,
            receipt,
            proof,
        } => {
            // Reject occupied output names before consuming authority. A successful
            // run always uses create_new below, so old results cannot be replaced.
            if receipt.exists() || proof.exists() {
                return Err("receipt or proof output already exists".into());
            }
            let capsule = read_canonical::<ValidationCapsule>(&capsule)?;
            let bindings: BindingManifest = serde_json::from_slice(&std::fs::read(bindings)?)?;
            let command: CommandSpec = serde_json::from_slice(&std::fs::read(command)?)?;
            let key = PrivateKey::read_openssh_file(&private_key)?;
            let signer = SshSigner::new(key, principal)?;
            let mut custody = SqliteCustody::open(ledger, ledger_id, signer)?;
            let artifacts = run_command(&mut custody, &capsule, &bindings, &command)?;
            write_exclusive(&receipt, &artifacts.receipt)?;
            write_exclusive(&proof, &artifacts.proof)?;
        }
        ValidateCommand::Verify {
            receipt,
            proof,
            trust_root,
        } => {
            let receipt = read_canonical::<EvaluationReceipt>(&receipt)?;
            let proof = read_canonical::<CustodyProof>(&proof)?;
            let trust_root = read_canonical::<TrustRoot>(&trust_root)?;
            verify_custody_proof(&proof, &receipt, &trust_root)?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Cli { command } = Cli::parse();
    match command {
        TopLevel::Validate { command } => execute(command),
    }
}
