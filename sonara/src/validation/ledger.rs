use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

use super::{
    to_canonical_bytes, ClaimToken, CompletionPayload, CustodyError, CustodyProof, CustodyProvider,
    DigestRef, EnvelopeKind, EvaluationReceipt, LedgerEnvelope, ProtectedCheckpoint,
    SignedCheckpoint, SignedEnvelope, SshSigner,
};

const ENVELOPE_FORMAT: &str = "sonara.custody-envelope.v1";
const PROOF_FORMAT: &str = "sonara.custody-proof.v1";

pub struct SqliteCustody {
    connection: Connection,
    ledger_id: String,
    signer: SshSigner,
}

impl SqliteCustody {
    pub fn open(
        path: impl AsRef<Path>,
        ledger_id: impl Into<String>,
        signer: SshSigner,
    ) -> Result<Self, CustodyError> {
        let ledger_id = ledger_id.into();
        let connection = Connection::open(path)?;
        connection.pragma_update(None, "journal_mode", "DELETE")?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS envelopes (
                sequence INTEGER PRIMARY KEY,
                digest TEXT NOT NULL UNIQUE,
                previous_digest TEXT UNIQUE,
                kind TEXT NOT NULL,
                signed_json BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claims (
                evaluation_digest TEXT PRIMARY KEY,
                claim_digest TEXT NOT NULL UNIQUE,
                claim_sequence INTEGER NOT NULL UNIQUE,
                completion_digest TEXT UNIQUE
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS head (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                sequence INTEGER NOT NULL,
                digest TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                signed_json BLOB NOT NULL
            );
            ",
        )?;
        let existing: Option<String> = connection
            .query_row(
                "SELECT value FROM metadata WHERE key = 'ledger_id'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        match existing {
            Some(existing) if existing != ledger_id => return Err(CustodyError::LedgerIdentity),
            Some(_) => {}
            None => {
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('ledger_id', ?1)",
                    [&ledger_id],
                )?;
            }
        }
        Ok(Self {
            connection,
            ledger_id,
            signer,
        })
    }

    fn signed_at(&self, sequence: u64) -> Result<SignedEnvelope, CustodyError> {
        let bytes: Vec<u8> = self.connection.query_row(
            "SELECT signed_json FROM envelopes WHERE sequence = ?1",
            [sequence],
            |row| row.get(0),
        )?;
        Ok(super::from_canonical_bytes(&bytes)?)
    }

    fn signed_checkpoint(&self) -> Result<SignedCheckpoint, CustodyError> {
        let bytes: Vec<u8> = self.connection.query_row(
            "SELECT signed_json FROM checkpoints WHERE singleton = 1",
            [],
            |row| row.get(0),
        )?;
        Ok(super::from_canonical_bytes(&bytes)?)
    }
}

impl CustodyProvider for SqliteCustody {
    fn claim(&mut self, evaluation_digest: &DigestRef) -> Result<ClaimToken, CustodyError> {
        evaluation_digest.validate()?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let head: Option<(u64, String, String)> = transaction
            .query_row(
                "SELECT sequence, digest, kind FROM head WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if head.is_some() {
            return Err(CustodyError::AlreadyClaimed);
        }
        let envelope = LedgerEnvelope {
            format: ENVELOPE_FORMAT.into(),
            ledger_id: self.ledger_id.clone(),
            sequence: 1,
            previous_digest: None,
            kind: EnvelopeKind::Claim,
            evaluation_digest: evaluation_digest.clone(),
            completion: None,
        };
        let digest = envelope.envelope_digest()?;
        let signed = self.signer.sign(envelope)?;
        let bytes = to_canonical_bytes(&signed)?;
        transaction.execute(
            "INSERT INTO envelopes(sequence, digest, previous_digest, kind, signed_json)
             VALUES (1, ?1, NULL, 'claim', ?2)",
            params![digest.value, bytes],
        )?;
        transaction.execute(
            "INSERT INTO claims(evaluation_digest, claim_digest, claim_sequence)
             VALUES (?1, ?2, 1)",
            params![evaluation_digest.value, digest.value],
        )?;
        transaction.execute(
            "INSERT INTO head(singleton, sequence, digest, kind) VALUES (1, 1, ?1, 'claim')",
            [&digest.value],
        )?;
        transaction.commit()?;
        Ok(ClaimToken {
            ledger_id: self.ledger_id.clone(),
            evaluation_digest: evaluation_digest.clone(),
            claim_digest: digest,
            sequence: 1,
        })
    }

    fn complete(
        &mut self,
        token: &ClaimToken,
        receipt: &EvaluationReceipt,
    ) -> Result<CustodyProof, CustodyError> {
        receipt.validate()?;
        if token.ledger_id != self.ledger_id
            || token.evaluation_digest != receipt.evaluation_digest
            || token.claim_digest != receipt.claim_digest
            || token.sequence != 1
        {
            return Err(CustodyError::StaleClaim);
        }
        let receipt_digest = receipt.receipt_digest()?;
        let trust_root = self.signer.trust_root()?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let claim: Option<(String, u64, Option<String>)> = transaction
            .query_row(
                "SELECT claim_digest, claim_sequence, completion_digest
                 FROM claims WHERE evaluation_digest = ?1",
                [&token.evaluation_digest.value],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let Some((claim_digest, claim_sequence, completion_digest)) = claim else {
            return Err(CustodyError::StaleClaim);
        };
        if completion_digest.is_some() {
            return Err(CustodyError::AlreadyCompleted);
        }
        if claim_digest != token.claim_digest.value || claim_sequence != token.sequence {
            return Err(CustodyError::StaleClaim);
        }
        let head: (u64, String, String) = transaction.query_row(
            "SELECT sequence, digest, kind FROM head WHERE singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        if head
            != (
                token.sequence,
                token.claim_digest.value.clone(),
                "claim".into(),
            )
        {
            return Err(CustodyError::HeadChanged);
        }
        let envelope = LedgerEnvelope {
            format: ENVELOPE_FORMAT.into(),
            ledger_id: self.ledger_id.clone(),
            sequence: token.sequence + 1,
            previous_digest: Some(token.claim_digest.clone()),
            kind: EnvelopeKind::Completion,
            evaluation_digest: token.evaluation_digest.clone(),
            completion: Some(CompletionPayload {
                evaluation_digest: token.evaluation_digest.clone(),
                claim_digest: token.claim_digest.clone(),
                receipt_digest,
            }),
        };
        let completion_digest = envelope.envelope_digest()?;
        let signed_completion = self.signer.sign(envelope)?;
        let completion_bytes = to_canonical_bytes(&signed_completion)?;
        transaction.execute(
            "INSERT INTO envelopes(sequence, digest, previous_digest, kind, signed_json)
             VALUES (?1, ?2, ?3, 'completion', ?4)",
            params![
                token.sequence + 1,
                completion_digest.value,
                token.claim_digest.value,
                completion_bytes
            ],
        )?;
        transaction.execute(
            "UPDATE claims SET completion_digest = ?1
             WHERE evaluation_digest = ?2 AND completion_digest IS NULL",
            params![completion_digest.value, token.evaluation_digest.value],
        )?;
        let changed = transaction.execute(
            "UPDATE head SET sequence = ?1, digest = ?2, kind = 'completion'
             WHERE singleton = 1 AND sequence = ?3 AND digest = ?4 AND kind = 'claim'",
            params![
                token.sequence + 1,
                completion_digest.value,
                token.sequence,
                token.claim_digest.value
            ],
        )?;
        if changed != 1 {
            return Err(CustodyError::HeadChanged);
        }
        let checkpoint = self.signer.sign_checkpoint(ProtectedCheckpoint {
            format: "sonara.protected-checkpoint.v1".into(),
            ledger_id: self.ledger_id.clone(),
            sequence: token.sequence + 1,
            head_digest: completion_digest.clone(),
        })?;
        transaction.execute(
            "INSERT INTO checkpoints(singleton, signed_json) VALUES (1, ?1)",
            [to_canonical_bytes(&checkpoint)?],
        )?;
        transaction.commit()?;
        let claim = self.signed_at(1)?;
        let completion = self.signed_at(2)?;
        let checkpoint = self.signed_checkpoint()?;
        Ok(CustodyProof {
            format: PROOF_FORMAT.into(),
            ledger_id: self.ledger_id.clone(),
            trust_root,
            checkpoint,
            claim,
            completion,
        })
    }
}
