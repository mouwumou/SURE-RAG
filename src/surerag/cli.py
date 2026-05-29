"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from surerag import SureRAGVerifier
from surerag.artifact import save_artifact
from surerag.io import read_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="surerag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="Verify JSONL requests")
    verify_parser.add_argument("--input", required=True, help="Input JSONL request file")
    verify_parser.add_argument("--output", required=True, help="Output JSONL result file")
    verify_parser.add_argument("--backend", choices=["toy"], default=None)
    verify_parser.add_argument("--artifact", default=None, help="Verifier artifact directory")

    export_parser = subparsers.add_parser("export", help="Export a verifier artifact")
    export_parser.add_argument("--backend", choices=["toy"], required=True)
    export_parser.add_argument("--output", required=True, help="Output artifact directory")

    export_toy_parser = subparsers.add_parser(
        "export-toy",
        help="Export a toy verifier artifact",
    )
    export_toy_parser.add_argument("--output", required=True, help="Output artifact directory")

    args = parser.parse_args(argv)

    if args.command == "verify":
        verifier = _load_verifier(args.backend, args.artifact)
        rows = read_jsonl(args.input)
        results = [verifier.verify(row) for row in rows]
        write_jsonl(args.output, results)
        return

    if args.command == "export":
        save_artifact(_load_verifier(args.backend, artifact=None), Path(args.output))
        return

    if args.command == "export-toy":
        save_artifact(_load_verifier("toy", artifact=None), Path(args.output))
        return

    parser.error(f"unknown command: {args.command}")


def _load_verifier(backend: str | None, artifact: str | None) -> SureRAGVerifier:
    if artifact is not None:
        return SureRAGVerifier.from_artifact(artifact)
    if backend == "toy":
        return SureRAGVerifier.toy()
    raise SystemExit("Either --backend toy or --artifact must be supplied")


if __name__ == "__main__":
    main()
