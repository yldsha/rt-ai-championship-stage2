#!/usr/bin/env python3
"""
End-to-end indexing pipeline for RAG.
Extracts semantic chunks from source code, generates embeddings, and saves them to ChromaDB.
"""

import time
import hashlib
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import dataclasses

import torch
from sentence_transformers import SentenceTransformer
import chromadb

from chunker.router import collect_chunks_for_file, get_supported_extensions
from chunker.chunk import Chunk
import chunker.utils

# логирование вместо print
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-language code chunker and indexer for RAG")
    parser.add_argument("source_root", type=Path, help="Directory with source files to chunk and index")
    parser.add_argument("--project-prefix", type=str, default=None, help="Path prefix used in chunk_id")
    parser.add_argument("--output-backup", type=Path, default=Path("RAG/data/chunks.jsonl"))
    parser.add_argument("--force", action="store_true", help="Force re-indexing: purge existing collection")
    
    # выносим конфигурацию в аргументы
    parser.add_argument("--model-name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--db-path", type=Path, default=Path("RAG/chroma_db"))
    parser.add_argument("--collection", type=str, default="gymhero_code")
    parser.add_argument("--batch-size", type=int, default=16)
    
    return parser.parse_args()


def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def chunk_to_dict(chunk: Chunk) -> dict:
    """Safely converts a Chunk instance or dataclass into a standard dictionary."""
    if dataclasses.is_dataclass(chunk):
        return dataclasses.asdict(chunk)
    elif hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    return vars(chunk)


def build_enriched_text(chunk: dict) -> str:
    """ Context Injection """
    text_parts = []
    if chunk.get("path"): text_parts.append(f"File: {chunk['path']}")
    if chunk.get("language"): text_parts.append(f"Language: {chunk['language']}")
    if chunk.get("symbol"): text_parts.append(f"{chunk.get('chunk_type', 'symbol').capitalize()}: {chunk['symbol']}")
    
    docstring = f"\nDescription: {chunk['docstring']}" if chunk.get("docstring") else ""
    code_body = f"\nCode:\n{chunk.get('code', '')}"
    
    enriched_text = f"{', '.join(text_parts)}{docstring}{code_body}".strip()
    return enriched_text if enriched_text else chunk.get("code", "empty chunk")


def extract_all_chunks(source_root: Path, project_prefix: str) -> list[Chunk]:
    """ Chunk extraction with error handling and logging """
    logger.info(f"Extracting semantic chunks from repository {source_root}...")
    all_chunks = []
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
            
    logger.info(f"Successfully extracted {len(all_chunks)} semantic chunks")
    return all_chunks


def setup_chromadb(db_path: Path, collection_name: str, force: bool):
    """Initialize ChromaDB client and collection, with optional force purge"""
    logger.info(f"Connecting to ChromaDB at {db_path}...")
    client = chromadb.PersistentClient(path=str(db_path))

    if force:
        logger.warning(f"FORCE MODE: Purging existing collection '{collection_name}'...")
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def main() -> int:
    start_time = time.time()

    args = parse_args()
    source_root = args.source_root.resolve()

    if not source_root.exists() or not source_root.is_dir():
        logger.error(f"Source root does not exist or is not a directory: {source_root}")
        return 1

    project_prefix = args.project_prefix or source_root.name

    # извлечение данных
    all_chunks = extract_all_chunks(source_root, project_prefix)
    if not all_chunks:
        logger.info("Chunks not found. Indexing cancelled.")
        return 0

    # бэкап
    chunker.utils.write_jsonl(all_chunks, args.output_backup)
    logger.info(f"Backup saved to {args.output_backup}")

    # загрузка модели
    logger.info(f"Loading model {args.model_name}...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)
    
    # бд
    collection = setup_chromadb(args.db_path, args.collection, args.force)
    existing_ids = set(collection.get()["ids"])
    logger.info(f"Found {len(existing_ids)} chunks already in the database.")

    # индексирование с батчингом и проверкой на дубликаты
    logger.info("Generating embeddings for new chunks...")
    processed_count = 0
    
    for i in tqdm(range(0, len(all_chunks), args.batch_size)):
        batch_chunks = all_chunks[i : i + args.batch_size]
        
        batch_ids, batch_texts, batch_metadatas = [], [], []
        
        for chunk_obj in batch_chunks:
            chunk = chunk_to_dict(chunk_obj)
            enriched_text = build_enriched_text(chunk)
            content_hash = get_text_hash(enriched_text)
            
            if content_hash in existing_ids:
                continue
                
            meta = {k: v for k, v in chunk.items() 
                    if k not in ["code", "docstring", "chunk_id"] 
                    and isinstance(v, (str, int, float, bool)) and v is not None}
            
            batch_ids.append(content_hash)
            batch_texts.append(enriched_text)
            batch_metadatas.append(meta)
            
        if batch_ids:
            batch_embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            processed_count += len(batch_ids)

        # очистка памяти для видеокарты
        if device == "mps":
            torch.mps.empty_cache()

    end_time = time.time()

    logger.info(f"Indexing complete! Added/Updated: {processed_count} chunks.")
    logger.info(f"Time: {end_time - start_time:2f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())