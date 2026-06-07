#!/usr/bin/env python3
"""AST-based code chunker for Python repositories.

This script walks Python files and creates semantic chunks for:
- top-level functions
- classes
- methods inside classes

Output format is JSONL where each line is a chunk object ready for RAG indexing.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Chunk:
    chunk_id: str
    path: str
    symbol: str
    chunk_type: str
    start_line: int
    end_line: int
    code: str
    docstring: str | None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "symbol": self.symbol,
            "chunk_type": self.chunk_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
            "docstring": self.docstring,
        }


class ASTChunkCollector(ast.NodeVisitor):
    """Collect class/function/method chunks from a parsed module AST."""

    def __init__(self, module_path: str, source_lines: list[str]) -> None:
        self.module_path = module_path
        self.source_lines = source_lines
        self.class_stack: list[str] = []
        self.chunks: list[Chunk] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        
        self._append_chunk(node=node, symbol=node.name, chunk_type="class")

        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.class_stack:
            symbol = f"{self.class_stack[-1]}.{node.name}"
            chunk_type = "method"
        else:
            symbol = node.name
            chunk_type = "function"

        self._append_chunk(node=node, symbol=symbol, chunk_type=chunk_type)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self.class_stack:
            symbol = f"{self.class_stack[-1]}.{node.name}"
            chunk_type = "method"
        else:
            symbol = node.name
            chunk_type = "function"

        self._append_chunk(node=node, symbol=symbol, chunk_type=chunk_type)
        self.generic_visit(node)

    def _append_chunk(
        self,
        node: ast.AST,
        symbol: str,
        chunk_type: str,
    ) -> None:
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        code = "".join(self.source_lines[start_line - 1 : end_line]).rstrip("\n")
        docstring = ast.get_docstring(node) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else None

        chunk_id = f"{self.module_path}:{symbol}:{start_line}"
        self.chunks.append(
            Chunk(
                chunk_id=chunk_id,
                path=self.module_path,
                symbol=symbol,
                chunk_type=chunk_type,
                start_line=start_line,
                end_line=end_line,
                code=code,
                docstring=docstring,
            )
        )


def iter_python_files(source_root: Path) -> Iterable[Path]:
    for file_path in source_root.rglob("*.py"):
        if "__pycache__" in file_path.parts:
            continue
        yield file_path


def normalize_module_path(
    file_path: Path,
    source_root: Path,
    project_prefix: str,
) -> str:
    relative = file_path.relative_to(source_root)
    return f"{project_prefix}/{relative.as_posix()}"


def collect_chunks_for_file(
    file_path: Path,
    source_root: Path,
    project_prefix: str,
) -> list[Chunk]:
    source = file_path.read_text(encoding="utf-8")
    source_lines = source.splitlines(keepends=True)

    tree = ast.parse(source)
    module_path = normalize_module_path(file_path, source_root, project_prefix)

    collector = ASTChunkCollector(module_path=module_path, source_lines=source_lines)
    collector.visit(tree)

    return collector.chunks


def write_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AST chunker for Python code")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/gymhero/gymhero"),
        help="Directory with Python source files to chunk",
    )
    parser.add_argument(
        "--project-prefix",
        type=str,
        default="gymhero",
        help="Path prefix used in chunk_id path section",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("RAG/chunks_ast.jsonl"),
        help="Output JSONL file path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_root = args.source_root.resolve()
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist or is not a directory: {source_root}")

    all_chunks: list[Chunk] = []
    for file_path in sorted(iter_python_files(source_root)):
        try:
            file_chunks = collect_chunks_for_file(
                file_path=file_path,
                source_root=source_root,
                project_prefix=args.project_prefix,
            )
        except SyntaxError as exc:
            print(f"[WARN] Skipping {file_path}: syntax error: {exc}")
            continue

        all_chunks.extend(file_chunks)

    write_jsonl(all_chunks, args.output)
    print(f"Wrote {len(all_chunks)} chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
