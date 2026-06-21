#!/usr/bin/env python3
"""
End-to-end indexing pipeline for RAG.
Extracts semantic chunks from source code, generates embeddings, and saves them to ChromaDB.

Usage:
    python index.py <path_to_directory>
    Example: python index.py data/gymhero/gymhero
"""

import argparse
import dataclasses
import logging
import time
from pathlib import Path

import chromadb
import chunker.utils
import torch
from chunker.chunk import Chunk
from chunker.router import collect_chunks_for_file, get_supported_extensions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# настройка логирования
logging.basicConfig(
    level=logging.WARNING, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-language code chunker and indexer for RAG")
    parser.add_argument("source_root", type=Path, help="Directory with source files to chunk and index")
    parser.add_argument("--output-backup", type=Path, default=Path("RAG/data/chunks.jsonl"))
    parser.add_argument("--force", action="store_true", help="Force re-indexing: purge existing collection")
    
    # конфигурация в аргументах
    parser.add_argument("--model-name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--db-path", type=Path, default=Path("RAG/chroma_db"))
    parser.add_argument("--collection", type=str, default="gymhero_code")
    parser.add_argument("--batch-size", type=int, default=16)
    
    return parser.parse_args()


def get_embeddings(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    """Generates embedding vectors for a list of text strings (batch processing)."""
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def chunk_to_dict(chunk: Chunk) -> dict:
    """Safely converts a Chunk instance or dataclass into a standard dictionary."""
    if dataclasses.is_dataclass(chunk):
        return dataclasses.asdict(chunk)
    elif hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    return vars(chunk)


def build_enriched_text(chunk: dict) -> str:
    """Context Injection"""
    text_parts = []
    if chunk.get("path"):
        text_parts.append(f"File: {chunk['path']}")
    if chunk.get("language"):
        text_parts.append(f"Language: {chunk['language']}")
    if chunk.get("symbol"):
        chunk_type = chunk.get('chunk_type', 'symbol')
        text_parts.append(f"{chunk_type.capitalize()}: {chunk['symbol']}")

    docstring = f"\nDescription: {chunk['docstring']}" if chunk.get("docstring") else ""
    code_body = f"\nCode:\n{chunk.get('code', '')}"

    enriched_text = f"{', '.join(text_parts)}{docstring}{code_body}".strip()
    
    return enriched_text if enriched_text else chunk.get("code", "empty chunk")


def extract_metadata(chunk: dict) -> dict:
    """Extracts clean metadata, filtering out heavy text fields and unsupported types."""
    return {
        k: v for k, v in chunk.items()
        if k not in ["code", "docstring"] and isinstance(v, (str, int, float, bool))
    }


def main() -> int:
    start_time = time.time()

    args = parse_args()
    source_root = args.source_root.resolve()

    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(
            f"Source root does not exist or is not a directory: {source_root}"
        )

    project_prefix = args.source_root.name

    # cбор чанков
    logger.info(f"Extracting semantic chunks from repository {source_root}...")
    all_chunks: list[Chunk] = []

    supported_exts = get_supported_extensions()

    for file_path in sorted(chunker.utils.iter_files(source_root, supported_exts)):
        try:
            file_chunks = collect_chunks_for_file(
                file_path=file_path,
                source_root=source_root,
                project_prefix=project_prefix,
            )
            all_chunks.extend(file_chunks)
        except Exception as exc:
            logger.warning(f"Skipping {file_path} due to error: {exc}")
            continue

    logger.info(f"Successfully extracted {len(all_chunks)} semantic chunks")

    if not all_chunks:
        logger.error("Chunks not found. Indexing cancelled.")
        return 0

    # сохраняем бэкап
    chunker.utils.write_jsonl(all_chunks, args.output_backup)
    logger.info(f"Backup saved to {args.output_backup}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")
    logger.info(f"\nLoading model {args.model_name}...")
    model = SentenceTransformer(args.model_name, device=device)

    logger.info(f"Connecting to ChromaDB (folder {args.db_path})...")
    chroma_client = chromadb.PersistentClient(path=args.db_path)

    if getattr(args, "force", False):
        logger.info(f"FORCE MODE: Purging existing collection '{args.collection}' for a clean re-indexing...")
        try:
            chroma_client.delete_collection(name=args.collection)
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=args.collection, metadata={"hnsw:space": "cosine"}
    )

    existing_ids = set(collection.get()["ids"])
    logger.info(f"Found {len(existing_ids)} chunks already in the database.")

    logger.info("\nGenerating embeddings and saving to ChromaDB in batches...")

    for i in tqdm(range(0, len(all_chunks), args.batch_size)):
        batch_chunks = all_chunks[i : i + args.batch_size]

        batch_ids = []
        batch_texts = []
        batch_metadatas = []

        for j, chunk_obj in enumerate(batch_chunks):
            chunk = chunk_to_dict(chunk_obj)
            
            chunk_id = chunk["chunk_id"]

            if chunk_id in existing_ids:
                continue

            enriched_text = build_enriched_text(chunk)
            meta = extract_metadata(chunk)

            batch_ids.append(chunk_id)
            batch_texts.append(enriched_text)
            batch_metadatas.append(meta)

        if batch_texts:
            batch_embeddings = get_embeddings(batch_texts, model)

            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    end_time = time.time()

    logger.info(f"\nDone! Successfully processed and saved {len(all_chunks)} chunks.")
    logger.info(f"Time: {end_time - start_time:.3f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
