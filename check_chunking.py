"""Exercise the configured splitter with synthetic text, never indexed as course data."""

import tiktoken
from langchain_core.documents import Document

from ingest import split_documents
from settings import CHUNK_OVERLAP, CHUNK_SIZE


def main():
    encoder = tiktoken.get_encoding("cl100k_base")
    # Repeated, individually numbered words make boundary overlap observable.
    text = " ".join(f"item{i:04d}" for i in range(1500))
    document = Document(page_content=text, metadata={"source": "synthetic-fixture"})
    chunks = split_documents([document])
    sizes = [len(encoder.encode(chunk.page_content)) for chunk in chunks]
    assert len(chunks) > 1
    assert max(sizes) <= CHUNK_SIZE
    assert all(c.metadata["source"] == "synthetic-fixture" for c in chunks)
    # Check the shared word boundary, rather than assuming recursive chunking
    # always yields exactly the configured number of overlapping tokens.
    left, right = chunks[0].page_content.split(), chunks[1].page_content.split()
    overlap = max((n for n in range(1, min(len(left), len(right)) + 1)
                   if left[-n:] == right[:n]), default=0)
    assert overlap > 0
    print(f"PASS: {len(chunks)} chunks, max {max(sizes)} tokens <= {CHUNK_SIZE}.")
    print(f"PASS: {overlap} shared words at first boundary; overlap target {CHUNK_OVERLAP} tokens.")
    short = split_documents([Document(page_content="Short document.", metadata={"source": "short"})])
    assert len(short) == 1 and short[0].page_content == "Short document."
    print("PASS: short documents are preserved without padding.")


if __name__ == "__main__":
    main()
