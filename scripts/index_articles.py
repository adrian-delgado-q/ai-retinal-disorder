from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.agent_tools import MedicalQueryInput, answer_with_citations, retrieve_medical_chunks
from rag.config import RAGConfig
from rag.index_builder import build_index, inspect_index


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--chroma-dir", type=Path, default=None)
    parser.add_argument("--chroma-collection", default=None)
    parser.add_argument("--embed-model-name", default=None)


def add_query_args(parser: argparse.ArgumentParser) -> None:
    add_common_config_args(parser)
    parser.add_argument("question")
    parser.add_argument("--disease-tag", default=None)
    parser.add_argument("--trust-level", action="append", dest="trust_levels", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-low-trust", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index and query ophthalmology PDFs with Chroma and LlamaIndex.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index", help="Build or update the Chroma index.")
    add_common_config_args(build_parser)

    raw_parser = subparsers.add_parser("query-raw", help="Retrieve raw chunk results.")
    add_query_args(raw_parser)

    answer_parser = subparsers.add_parser("query-answer", help="Retrieve a synthesized answer with citations.")
    add_query_args(answer_parser)

    inspect_parser = subparsers.add_parser("inspect-index", help="Inspect dataset and index health.")
    add_common_config_args(inspect_parser)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> RAGConfig:
    return RAGConfig.from_env(
        dataset_path=args.dataset_path,
        chroma_dir=args.chroma_dir,
        chroma_collection=args.chroma_collection,
        embed_model_name=args.embed_model_name,
    )


def build_query_input(args: argparse.Namespace) -> MedicalQueryInput:
    return MedicalQueryInput(
        question=args.question,
        disease_tag=args.disease_tag,
        trust_levels=args.trust_levels,
        source=args.source,
        year_min=args.year_min,
        year_max=args.year_max,
        top_k=args.top_k,
        allow_low_trust=args.allow_low_trust,
    )


def print_json(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.command == "build-index":
        print_json(build_index(config).to_dict())
        return
    if args.command == "query-raw":
        print_json(retrieve_medical_chunks(build_query_input(args), config=config))
        return
    if args.command == "query-answer":
        print_json(answer_with_citations(build_query_input(args), config=config))
        return
    if args.command == "inspect-index":
        print_json(inspect_index(config))
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
