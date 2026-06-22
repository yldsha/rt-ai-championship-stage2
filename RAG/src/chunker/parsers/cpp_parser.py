"""C++ code chunker based on Tree-sitter."""

from __future__ import annotations

from pathlib import Path

from tree_sitter_languages import get_language, get_parser

from RAG.src.chunker.chunk import Chunk

CPP_QUERY = """
(class_specifier) @class
(struct_specifier) @class
(function_definition) @function
"""


def parse_cpp(
    file_path: Path,
    module_path: str,
    source_text: str,
) -> list[Chunk]:
    """Parses C++ files and extracts semantic chunks."""
    lang_name = "cpp"
    lang = get_language(lang_name)
    parser = get_parser(lang_name)

    source_bytes = bytes(source_text, "utf-8")
    tree = parser.parse(source_bytes)

    query = lang.query(CPP_QUERY)
    captures = query.captures(tree.root_node)

    chunks: list[Chunk] = []

    for node, capture_name in captures:
        if capture_name == "class":
            chunk_type = "class"
            name_node = node.child_by_field_name("name")
        else:
            chunk_type = "function"
            decl_node = node.child_by_field_name("declarator")
            name_node = decl_node
            if decl_node:
                while name_node.child_by_field_name("declarator"):
                    name_node = name_node.child_by_field_name("declarator")

        if name_node:
            symbol = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
            if "(" in symbol:
                symbol = symbol.split("(")[0].strip()
        else:
            symbol = f"anonymous_{chunk_type}"

        if chunk_type == "function":
            parent = node.parent
            while parent and parent.type not in ("class_specifier", "struct_specifier"):
                parent = parent.parent
            if parent:
                chunk_type = "method"
                parent_name_node = parent.child_by_field_name("name")
                if parent_name_node:
                    parent_name = source_bytes[
                        parent_name_node.start_byte : parent_name_node.end_byte
                    ].decode("utf-8")
                    symbol = f"{parent_name}::{symbol}"

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        code = source_bytes[node.start_byte : node.end_byte].decode("utf-8")

        docstring = None
        prev_sibling = node.prev_sibling
        if prev_sibling and prev_sibling.type == "comment":
            docstring = source_bytes[prev_sibling.start_byte : prev_sibling.end_byte].decode("utf-8")

        chunk_id = f"{module_path}:{symbol}:{start_line}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                path=module_path,
                language=lang_name,
                symbol=symbol,
                chunk_type=chunk_type,
                start_line=start_line,
                end_line=end_line,
                code=code,
                docstring=docstring,
            )
        )

    return chunks