from pathlib import Path
from typing import Any, Mapping, Union

JsonObject = Mapping[str, Any]
JsonSource = Union[JsonObject, str, Path]

class PreparedValidation:
    evaluation_digest: str
    resource_count: int

class ValidationRun:
    receipt: dict[str, Any]
    proof: dict[str, Any]
    receipt_json: str
    proof_json: str

def canonical_json(value: JsonObject) -> str: ...
def capsule_digest(capsule: JsonSource) -> str: ...
def prepare(capsule: JsonSource, bindings: JsonSource) -> PreparedValidation: ...
def run(capsule: JsonSource, bindings: JsonSource, command: JsonSource, *, ledger: Union[str, Path], ledger_id: str, private_key: Union[str, Path], principal: str) -> ValidationRun: ...
def verify(receipt: JsonSource, proof: JsonSource, trust_root: JsonSource) -> None: ...
