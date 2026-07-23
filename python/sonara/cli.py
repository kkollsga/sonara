"""Installed ``sonara`` console command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from sonara import validation


def _write_exclusive(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sonara")
    top = parser.add_subparsers(dest="group", required=True)
    validate = top.add_parser("validate", help="content-addressed validation custody")
    commands = validate.add_subparsers(dest="operation", required=True)

    prepare = commands.add_parser("prepare", help="validate public metadata")
    prepare.add_argument("--capsule", type=Path, required=True)
    prepare.add_argument("--bindings", type=Path, required=True)

    run = commands.add_parser("run", help="consume authority and execute a runner")
    run.add_argument("--capsule", type=Path, required=True)
    run.add_argument("--bindings", type=Path, required=True)
    run.add_argument("--command", type=Path, required=True)
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--ledger-id", required=True)
    run.add_argument("--private-key", type=Path, required=True)
    run.add_argument("--principal", required=True)
    run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--proof", type=Path, required=True)

    verify = commands.add_parser("verify", help="verify against a pinned trust root")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--proof", type=Path, required=True)
    verify.add_argument("--trust-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "prepare":
        prepared = validation.prepare(args.capsule, args.bindings)
        sys.stdout.write(validation.canonical_json({
            "evaluation_digest": {"algorithm": "sha256", "value": prepared.evaluation_digest},
            "format": "sonara.prepared-validation.v1",
            "resource_count": prepared.resource_count,
        }))
        return 0
    if args.operation == "run":
        if args.receipt.exists() or args.proof.exists():
            raise FileExistsError("receipt or proof output already exists")
        result = validation.run(
            args.capsule,
            args.bindings,
            args.command,
            ledger=args.ledger,
            ledger_id=args.ledger_id,
            private_key=args.private_key,
            principal=args.principal,
        )
        _write_exclusive(args.receipt, result.receipt_json)
        _write_exclusive(args.proof, result.proof_json)
        return 0
    validation.verify(args.receipt, args.proof, args.trust_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
