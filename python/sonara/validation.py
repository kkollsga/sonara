"""Content-addressed, single-use validation custody.

Validation is separate from ordinary audio analysis. Public capsules, receipts,
and proofs never contain private resource paths; paths live only in the local
binding manifest. Capsule, receipt, proof, and trust-root JSON files must use
Sonara's canonical compact representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Union

from sonara._sonara import (
    _validation_capsule_digest,
    _validation_prepare,
    _validation_run,
    _validation_verify,
)


JsonObject = Mapping[str, Any]
JsonSource = Union[JsonObject, str, Path]


def canonical_json(value: JsonObject) -> str:
    """Encode a JSON object in the validation protocol's canonical form."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _read(source: JsonSource, *, canonicalize_mapping: bool = True) -> str:
    if isinstance(source, Mapping):
        if not canonicalize_mapping:
            raise TypeError("this validation input must preserve its original canonical bytes")
        return canonical_json(source)
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if not isinstance(source, str):
        raise TypeError("expected a mapping, JSON string, or pathlib.Path")
    if source.lstrip().startswith("{"):
        return source
    return Path(source).read_text(encoding="utf-8")


def _normalized(source: JsonSource) -> str:
    """Normalize local, non-identity manifests before crossing the Rust API."""
    return canonical_json(json.loads(_read(source)))


@dataclass(frozen=True)
class PreparedValidation:
    evaluation_digest: str
    resource_count: int


@dataclass(frozen=True)
class ValidationRun:
    receipt: dict[str, Any]
    proof: dict[str, Any]
    receipt_json: str
    proof_json: str


def capsule_digest(capsule: JsonSource) -> str:
    """Return the domain-separated identity of a canonical capsule."""
    return _validation_capsule_digest(_read(capsule))


def prepare(capsule: JsonSource, bindings: JsonSource) -> PreparedValidation:
    """Check public metadata and binding coverage without opening resources."""
    value = json.loads(_validation_prepare(_read(capsule), _normalized(bindings)))
    return PreparedValidation(
        evaluation_digest=value["evaluation_digest"]["value"],
        resource_count=value["resource_count"],
    )


def run(
    capsule: JsonSource,
    bindings: JsonSource,
    command: JsonSource,
    *,
    ledger: Union[str, Path],
    ledger_id: str,
    private_key: Union[str, Path],
    principal: str,
) -> ValidationRun:
    """Consume authority, verify resources, and execute the attested runner.

    A failed resource check or runner crash still consumes the ledger. A new
    output path cannot replay the evaluation.
    """
    receipt_json, proof_json = _validation_run(
        _read(capsule),
        _normalized(bindings),
        _normalized(command),
        str(ledger),
        ledger_id,
        str(private_key),
        principal,
    )
    return ValidationRun(
        receipt=json.loads(receipt_json),
        proof=json.loads(proof_json),
        receipt_json=receipt_json,
        proof_json=proof_json,
    )


def verify(receipt: JsonSource, proof: JsonSource, trust_root: JsonSource) -> None:
    """Verify a completed run offline against a separately pinned trust root."""
    _validation_verify(_read(receipt), _read(proof), _read(trust_root))


__all__ = [
    "PreparedValidation",
    "ValidationRun",
    "canonical_json",
    "capsule_digest",
    "prepare",
    "run",
    "verify",
]
