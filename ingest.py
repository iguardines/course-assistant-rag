"""Load course documents, split them into tokens, and persist their embeddings."""

import argparse
import hashlib
import json
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import (
    BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DATA_DIR,
    EMBEDDING_MODEL, VECTORSTORE_DIR,
)


def load_documents() -> list[Document]:
    """Read UTF-8 Markdown and text files and preserve their relative paths."""
    documents = []
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                documents.append(Document(
                    page_content=content,
                    metadata={"source": path.relative_to(BASE_DIR).as_posix()},
                ))
    if not documents:
        raise ValueError("No non-empty .md or .txt documents found in data/.")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Use a token-based size limit; short documents remain short."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def ingest_documents(dry_run: bool = False) -> int:
    """Index once; reuse unchanged data and reject incompatible existing indexes."""
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} documents; created {len(chunks)} chunks.")
    for chunk in chunks:
        print(f"  [{chunk.metadata['chunk_index']}] {chunk.metadata['source']}")
    if dry_run:
        print("Preview only: no OpenAI calls or database writes.")
        return len(chunks)

    # Include both content and settings so an existing index cannot be silently
    # reused with different documents or a different embedding model.
    payload = {
        "model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "chunks": [(c.page_content, c.metadata) for c in chunks],
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    import chromadb

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    metadata = collection.metadata or {}
    if collection.count():
        if metadata.get("signature") != signature:
            raise ValueError(
                "Documents or settings changed. Use a new VECTORSTORE_DIR in "
                "settings.py to build a fresh index; the existing one was preserved."
            )
        if collection.count() == len(chunks):
            print("Existing index is complete and unchanged. No embedding calls needed.")
            return len(chunks)
    elif metadata.get("signature") not in (None, signature):
        raise ValueError("Index settings differ. Choose a new VECTORSTORE_DIR.")

    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("Set OPENAI_API_KEY in .env before indexing.")

    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    collection.modify(metadata={"signature": signature, "embedding_model": EMBEDDING_MODEL})
    store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    # Stable IDs allow an interrupted indexing run to resume without duplicates.
    existing_ids = set(collection.get(include=[])["ids"])
    ids = [f"{signature}:{i}" for i in range(len(chunks))]
    pending = [(chunk, chunk_id) for chunk, chunk_id in zip(chunks, ids)
               if chunk_id not in existing_ids]
    if pending:
        store.add_documents(
            documents=[chunk for chunk, _ in pending],
            ids=[chunk_id for _, chunk_id in pending],
        )
    print(f"Saved {collection.count()} chunks in {VECTORSTORE_DIR}.")
    return collection.count()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls.")
    args = parser.parse_args()
    try:
        ingest_documents(dry_run=args.dry_run)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
