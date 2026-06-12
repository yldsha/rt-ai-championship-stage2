"""JavaScript and TypeScript code chunker based on Tree-sitter.

Extracts classes, top-level functions, and class methods from JS/TS source code
using tree-sitter grammars. It automatically maps node coordinates and 
attempts to extract leading JSDoc comments as docstrings.
"""

from __future__ import annotations
from pathlib import Path

from tree_sitter_languages import get_language, get_parser

from chunk import Chunk

# Универсальный синтаксический запрос для JS и TS
# Ищет объявления классов, функций и методов
TS_JS_QUERY = """
(class_declaration) @class
(function_declaration) @function
(method_definition) @method
"""


def parse_javascript(
    file_path: Path,
    module_path: str,
    source_text: str,
    lang_name: str = "javascript",
) -> list[Chunk]:
    """Parses JS/TS files and extracts semantic chunks.

    Args:
        file_path: Path object pointing to the source file.
        module_path: Normalized project-relative path used in chunk_id.
        source_text: Raw string content of the file.
        lang_name: Target language identifier ('javascript' or 'typescript').

    Returns:
        A list of initialized Chunk objects.
    """
    # инициализируем языковой движок tree-sitter
    lang = get_language(lang_name)
    parser = get_parser(lang_name)

    # Tree-sitter строго работает с байтами, а не со строками
    source_bytes = bytes(source_text, "utf-8")
    tree = parser.parse(source_bytes)

    # Компилируем запрос и ищем совпадения в дереве
    query = lang.query(TS_JS_QUERY)
    captures = query.captures(tree.root_node)

    chunks: list[Chunk] = []

    for node, capture_name in captures:
        # Определяем базовый тип чанка
        if capture_name == "class":
            chunk_type = "class"
        elif capture_name == "method":
            chunk_type = "method"
        else:
            chunk_type = "function"

        # Безопасно извлекаем имя сущности (символ) через встроенные поля tree-sitter
        name_node = node.child_by_field_name("name")
        if name_node:
            symbol = source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
        else:
            symbol = f"anonymous_{chunk_type}"

        # Если это метод, поднимаемся по дереву выше, чтобы найти имя его класса
        if chunk_type == "method":
            parent = node.parent
            while parent and parent.type != "class_declaration":
                parent = parent.parent
            if parent:
                parent_name_node = parent.child_by_field_name("name")
                if parent_name_node:
                    parent_name = source_bytes[
                        parent_name_node.start_byte : parent_name_node.end_byte
                    ].decode("utf-8")
                    symbol = f"{parent_name}.{symbol}"

        # Считаем координаты строк (у tree-sitter они начинаются с 0, прибавляем 1)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Извлекаем чистый код по байтовым смещениям
        code = source_bytes[node.start_byte : node.end_byte].decode("utf-8")

        # Бонусная логика: извлекаем комментарий (JSDoc), если он идет прямо перед узлом
        docstring = None
        prev_sibling = node.prev_sibling
        if prev_sibling and prev_sibling.type == "comment":
            docstring = source_bytes[
                prev_sibling.start_byte : prev_sibling.end_byte
            ].decode("utf-8")

        # Формируем уникальный chunk_id и упаковываем данные в наш контракт
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