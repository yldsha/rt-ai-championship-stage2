#!/usr/bin/env python3
import argparse
from pathlib import Path

from chunk import Chunk
from parsers.js_ts import parse_javascript
from parsers.py_ast import parse_python
import utils

# Карта парсеров
PARSERS_MAP = {
    ".py": parse_python,
    ".js": lambda f, m, s: parse_javascript(f, m, s, "javascript"),
    ".ts": lambda f, m, s: parse_javascript(f, m, s, "typescript"),
}


def collect_chunks_for_file(file_path: Path, source_root: Path, project_prefix: str) -> list[Chunk]:
    ext = file_path.suffix.lower()
    if ext not in PARSERS_MAP:
        return []

    source_text = file_path.read_text(encoding="utf-8")
    module_path = utils.normalize_module_path(file_path, source_root, project_prefix)

    parse_function = PARSERS_MAP[ext]
    return parse_function(file_path, module_path, source_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-language code chunker for RAG")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/gymhero/gymhero"),
        help="Directory with source files to chunk",
    )
    parser.add_argument(
        "--project-prefix",
        type=str,
        default="gymhero",
        help="Path prefix used in chunk_id",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("RAG/chunks.jsonl"),
        help="Output JSONL file path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()

    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist or is not a directory: {source_root}")

    all_chunks: list[Chunk] = []
    
    # Передаем список поддерживаемых расширений напрямую из ключей словаря
    for file_path in sorted(utils.iter_files(source_root, supported_extensions=PARSERS_MAP.keys())):
        try:
            file_chunks = collect_chunks_for_file(
                file_path=file_path,
                source_root=source_root,
                project_prefix=args.project_prefix,
            )
            all_chunks.extend(file_chunks)
        except Exception as exc:
            print(f"[WARN] Skipping {file_path} due to error: {exc}")
            continue

    utils.write_jsonl(all_chunks, args.output)
    print(f"Wrote {len(all_chunks)} chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())