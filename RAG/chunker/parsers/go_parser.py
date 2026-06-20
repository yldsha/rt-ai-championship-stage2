"""Go code chunker based on Tree-sitter."""

from __future__ import annotations

from pathlib import Path

from tree_sitter_languages import get_language, get_parser

from chunker.chunk import Chunk

GO_QUERY = """
(type_declaration (type_spec type: (struct_type))) @class
(function_declaration) @function
(method_declaration) @method
"""


def parse_go(
    file_path: Path,
    module_path: str,
    source_text: str,
) -> list[Chunk]:
    """Parses Go files and extracts semantic chunks."""
    lang_name = "go"
    lang = get_language(lang_name)
    parser = get_parser(lang_name)

    source_bytes = bytes(source_text, "utf-8")
    tree = parser.parse(source_bytes)

    query = lang.query(GO_QUERY)
    captures = query.captures(tree.root_node)

    chunks: list[Chunk] = []

    for node, capture_name in captures:
        if capture_name == "class":
            chunk_type = "class"
        elif capture_name == "method":
            chunk_type = "method"
        else:
            chunk_type = "function"

        symbol = f"anonymous_{chunk_type}"

        if chunk_type == "class":
            type_spec = node.child_by_field_name("name") or node.named_child(0)
            if type_spec:
                name_node = type_spec.child_by_field_name("name")
                if name_node:
                    symbol = source_bytes[
                        name_node.start_byte : name_node.end_byte
                    ].decode("utf-8")
        else:
            name_node = node.child_by_field_name("name")
            if name_node:
                symbol = source_bytes[name_node.start_byte : name_node.end_byte].decode(
                    "utf-8"
                )

        # Если это метод Go, вытаскиваем тип ресивера (получателя) из параметров: func (r *MyStruct) Method()
        if chunk_type == "method":
            receiver_node = node.child_by_field_name("receiver")
            if receiver_node:
                struct_name = None
                for child in receiver_node.children:
                    if child.type == "parameter_declaration":
                        type_node = child.child_by_field_name("type")
                        if type_node:
                            if type_node.type == "pointer_type":
                                type_node = type_node.named_child(0)
                            if type_node:
                                struct_name = source_bytes[
                                    type_node.start_byte : type_node.end_byte
                                ].decode("utf-8")
                if struct_name:
                    symbol = f"{struct_name}.{symbol}"

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        code = source_bytes[node.start_byte : node.end_byte].decode("utf-8")

        docstring = None
        prev_sibling = node.prev_sibling
        if prev_sibling and prev_sibling.type in ("comment", "line_comment"):
            docstring = source_bytes[
                prev_sibling.start_byte : prev_sibling.end_byte
            ].decode("utf-8")

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
