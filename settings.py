"""Shared settings for ingestion and retrieval."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore_v2"
COLLECTION_NAME = "course_information"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 50
TOP_K = 3
