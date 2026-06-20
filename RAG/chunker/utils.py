import json
from chunk import Chunk
from pathlib import Path
from typing import Iterable

from chunker.chunk import Chunk

def normalize_module_path(
    file_path: Path, source_root: Path, project_prefix: str
) -> str:
    relative = file_path.relative_to(source_root)
    if project_prefix == source_root.name or not project_prefix:
        return relative.as_posix()
    return f"{project_prefix}/{relative.as_posix()}"


def write_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def iter_files(
    source_root: Path, supported_extensions: Iterable[str]
) -> Iterable[Path]:
    for file_path in source_root.rglob("*"):
        if file_path.is_file() and "__pycache__" not in file_path.parts:
            if file_path.suffix.lower() in supported_extensions:
                yield file_path
