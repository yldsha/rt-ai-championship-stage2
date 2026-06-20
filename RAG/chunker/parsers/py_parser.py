"""AST-based code chunker for Python repositories.

This script walks Python files and creates semantic chunks for:
- top-level functions
- classes
- methods inside classes

Output format is JSONL where each line is a chunk object ready for RAG indexing.
"""

from __future__ import annotations

import ast

from chunker.chunk import Chunk


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
        docstring = (
            ast.get_docstring(node)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            else None
        )

        chunk_id = f"{self.module_path}:{symbol}:{start_line}"
        self.chunks.append(
            Chunk(
                chunk_id=chunk_id,
                path=self.module_path,
                language="python",
                symbol=symbol,
                chunk_type=chunk_type,
                start_line=start_line,
                end_line=end_line,
                code=code,
                docstring=docstring,
            )
        )


def parse_python(file_path, module_path, source_text) -> list[Chunk]:
    source_lines = source_text.splitlines(keepends=True)
    tree = ast.parse(source_text)
    collector = ASTChunkCollector(module_path=module_path, source_lines=source_lines)
    collector.visit(tree)
    return collector.chunks
