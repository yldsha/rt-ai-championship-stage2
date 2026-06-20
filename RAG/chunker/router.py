"""
Router for mapping file extensions to their corresponding syntax parsers.
"""

from pathlib import Path

import chunker.utils
from chunker.chunk import Chunk
from chunker.parsers.cpp_parser import parse_cpp
from chunker.parsers.go_parser import parse_go
from chunker.parsers.java_parser import parse_java
from chunker.parsers.jsts_parser import parse_javascript
from chunker.parsers.py_parser import parse_python

PARSERS_MAP = {
    ".py": parse_python,
    ".js": lambda f, m, s: parse_javascript(f, m, s, "javascript"),
    ".jsx": lambda f, m, s: parse_javascript(f, m, s, "javascript"),
    ".ts": lambda f, m, s: parse_javascript(f, m, s, "typescript"),
    ".tsx": lambda f, m, s: parse_javascript(f, m, s, "typescript"),
    ".cpp": parse_cpp,
    ".hpp": parse_cpp,
    ".h": parse_cpp,
    ".go": parse_go,
    ".java": parse_java,
}


def get_supported_extensions():
    """Returns a view of all supported file extensions."""
    return PARSERS_MAP.keys()


def collect_chunks_for_file(
    file_path: Path, source_root: Path, project_prefix: str
) -> list[Chunk]:
    """Reads a file and routes it to the correct parser based on its extension."""
    ext = file_path.suffix.lower()
    if ext not in PARSERS_MAP:
        return []

    source_text = file_path.read_text(encoding="utf-8")
    module_path = chunker.utils.normalize_module_path(
        file_path, source_root, project_prefix
    )

    parse_function = PARSERS_MAP[ext]
    return parse_function(file_path, module_path, source_text)
