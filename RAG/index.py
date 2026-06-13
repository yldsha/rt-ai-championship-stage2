#!/usr/bin/env python3
"""
End-to-end indexing pipeline for RAG.
Extracts semantic chunks from source code, generates embeddings, and saves them to ChromaDB.

Usage:
    python index.py <path_to_directory>
    Example: python index.py data/gymhero/gymhero
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
import chromadb
import dataclasses

from chunker.chunk import Chunk
from chunker.parsers.jsts_parser import parse_javascript
from chunker.parsers.py_parser import parse_python
from chunker.parsers.cpp_parser import parse_cpp
from chunker.parsers.go_parser import parse_go
from chunker.parsers.java_parser import parse_java
import chunker.utils

# configurations
MODEL_NAME = "BAAI/bge-m3"
CHROMA_DB_DIR = "RAG/chroma_db"
COLLECTION_NAME = "gymhero_code"

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


def collect_chunks_for_file(file_path: Path, source_root: Path, project_prefix: str) -> list[Chunk]:
    """Reads a file and routes it to the correct parser based on its extension."""
    ext = file_path.suffix.lower()
    if ext not in PARSERS_MAP:
        return []

    source_text = file_path.read_text(encoding="utf-8")
    module_path = chunker.utils.normalize_module_path(file_path, source_root, project_prefix)

    parse_function = PARSERS_MAP[ext]
    return parse_function(file_path, module_path, source_text)


def get_embedding(text: str, model: SentenceTransformer) -> list:
    """Generates a normalized embedding vector for a single text string."""
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

def get_embeddings(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    """Generates embedding vectors for a list of text strings (batch processing)."""
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(description="Multi-language code chunker and indexer for RAG")
    # Делаем папку позиционным обязательным аргументом
    parser.add_argument(
        "source_root",
        type=Path,
        help="Directory with source files to chunk and index",
    )
    parser.add_argument(
        "--project-prefix",
        type=str,
        default=None,
        help="Path prefix used in chunk_id (по умолчанию берется имя папки)",
    )
    parser.add_argument(
        "--output-backup",
        type=Path,
        default=Path("RAG/data/chunks.jsonl"),
        help="Output JSONL file path for backup",
    )
    return parser.parse_args()


def chunk_to_dict(chunk: Chunk) -> dict:
    """Safely converts a Chunk instance or dataclass into a standard dictionary."""
    if dataclasses.is_dataclass(chunk):
        return dataclasses.asdict(chunk)
    elif hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    return vars(chunk)


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()

    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist or is not a directory: {source_root}")

    project_prefix = args.project_prefix or source_root.name

    # cбор чанков
    print(f"Extracting semantic chunks from repository {source_root}...")
    all_chunks: list[Chunk] = []
    
    for file_path in sorted(chunker.utils.iter_files(source_root, supported_extensions=PARSERS_MAP.keys())):
        try:
            file_chunks = collect_chunks_for_file(
                file_path=file_path,
                source_root=source_root,
                project_prefix=project_prefix,
            )
            all_chunks.extend(file_chunks)
        except Exception as exc:
            print(f"[WARN] Skipping {file_path} due to error: {exc}")
            continue

    print(f"Successfully extracted {len(all_chunks)} semantic chunks")
    
    if not all_chunks:
        print("Chunks not found. Indexing cancelled.")
        return 0

    # сохраняем бэкап
    chunker.utils.write_jsonl(all_chunks, args.output_backup)
    print(f"Backup saved to {args.output_backup}")

    print(f"\nLoading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Connecting to ChromaDB (folder {CHROMA_DB_DIR})...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    print(f"Purging existing collection '{COLLECTION_NAME}' for a clean re-indexing...")
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("\nGenerating embeddings and saving to ChromaDB in batches...")
    
    batch_size = 8 
    
    for i in tqdm(range(0, len(all_chunks), batch_size)):

        batch_chunks = all_chunks[i : i + batch_size]
        
        batch_ids = []
        batch_texts = []
        batch_metadatas = []
        
        for j, chunk_obj in enumerate(batch_chunks):
            chunk = chunk_to_dict(chunk_obj)
            
            # Context Injection
            text_parts = []
            if chunk.get("path"): text_parts.append(f"File: {chunk['path']}")
            if chunk.get("language"): text_parts.append(f"Language: {chunk['language']}")
            if chunk.get("symbol"): text_parts.append(f"{chunk['chunk_type'].capitalize()}: {chunk['symbol']}")
            
            docstring = f"\nDescription: {chunk['docstring']}" if chunk.get("docstring") else ""
            code_body = f"\nCode:\n{chunk.get('code', '')}"
            
            enriched_text = f"{", ".join(text_parts)}{docstring}{code_body}".strip()
            
            # если код пустой (бывает при сбоях парсера), делаем фоллбэк
            if not enriched_text:
                enriched_text = chunk.get("code", "empty chunk")
                
            # собираем чистую метадату
            meta = {}
            for k, v in chunk.items():
                if k not in ["code", "docstring"] and isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            
            # уникальный ID чанка
            global_idx = i + j
            chunk_id = chunk.get("chunk_id", str(global_idx))
            
            batch_ids.append(chunk_id)
            batch_texts.append(enriched_text)
            batch_metadatas.append(meta)
            
        batch_embeddings = get_embeddings(batch_texts, model)
        
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    print(f"\nDone! Successfully processed and saved {len(all_chunks)} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())