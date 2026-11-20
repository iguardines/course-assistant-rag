"""Retrieve course passages by meaning, without generating an answer."""

import argparse
import asyncio
import os

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from settings import (
    BASE_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K, VECTORSTORE_DIR,
)


def build_retriever() -> VectorStoreRetriever:
    """Open the existing collection using the ingestion embedding model."""
    if not (VECTORSTORE_DIR / "chroma.sqlite3").exists():
        raise ValueError("Vector database not found. Run python ingest.py first.")

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError as exc:
        raise ValueError("Course collection not found. Run python ingest.py first.") from exc
    if collection.count() == 0:
        raise ValueError("Course collection is empty. Run python ingest.py first.")
    if (collection.metadata or {}).get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("Embedding model differs from the model used during ingestion.")
    if not 3 <= TOP_K <= 5:
        raise ValueError("TOP_K must be between 3 and 5.")

    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("Set OPENAI_API_KEY in .env before querying.")

    store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        create_collection_if_not_exists=False,
    )
    return store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})


async def retrieve_documents(query: str) -> list[Document]:
    """Embed a question and return the nearest passages asynchronously."""
    if not query.strip():
        raise ValueError("The question must not be empty.")
    retriever = build_retriever()
    return await retriever.ainvoke(query.strip())


async def main(query: str) -> None:
    documents = await retrieve_documents(query)
    print(f"Question: {query}\nRetrieved {len(documents)} passages:\n")
    for rank, document in enumerate(documents, start=1):
        print(f"--- Result {rank}: {document.metadata['source']} "
              f"(chunk {document.metadata['chunk_index']}) ---")
        print(document.page_content)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="The question to search for.")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.query))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
