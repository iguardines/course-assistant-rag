# course-assistant-rag

A command-line assistant for prospective students of **First Steps in Python for Finance + AI**. It retrieves information from local course documents and generates a cited answer using an asynchronous LangChain Expression Language (LCEL) pipeline.

ChromaDB runs locally. OpenAI supplies embeddings and answer generation: documents are sent to the embeddings API, and the retrieved passages and question are sent to the chat API. API usage is billed by the provider.

## Requirements

- Python 3.12 (tested).
- An OpenAI API key with access to the configured models and available API quota.
- Internet access for installation, the first tokenizer download, and API calls.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` locally:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

Do not commit `.env`. Only `.env.example`, which contains no credentials, belongs in Git. `OPENAI_CHAT_MODEL` is optional; the default is `gpt-4.1-mini`. The default is the model tested with this project, not a claim that it is the newest model. Choose a compatible model available to your account if necessary. Chat model changes do not require reindexing; embedding model changes do.

## 1. Ingest documents

```bash
python ingest.py --dry-run
python ingest.py
```

The script reads non-empty UTF-8 `.txt` and `.md` files recursively under `data/`. It uses `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with a **600-token size limit** and **50-token overlap target**, counted with `cl100k_base`. It stores source paths and chunk indexes with each passage.

The current dataset has four short documents, so it produces four chunks. A size limit does not pad short documents to 600 tokens. Recursive splitting respects separators, so actual chunk sizes and overlap can vary. The assignment's “minimum 500 tokens” is interpreted here as configuring a chunk size of at least 500, not forcing every fragment to contain at least 500 tokens. The separate synthetic check below demonstrates splitting longer text without inventing course facts.

Chroma persists automatically in `vectorstore_v2/`. An unchanged rerun skips embedding calls. A signature includes document content, source metadata, embedding model, and splitter settings. If these change, ingestion refuses to silently reuse an incompatible index. To rebuild, choose a fresh directory in `VECTORSTORE_DIR` in `settings.py`, then rerun ingestion. Both ingestion and retrieval import the same settings.

## 2. Inspect retrieval

```bash
python retriever.py "Do I need programming experience?"
```

Retrieval embeds the question with the same `text-embedding-3-small` model used for ingestion, searches the local index, and returns the three nearest passages. It does not generate an answer. Similarity is not proof that a passage answers the question; even an unrelated question can retrieve results.

## 3. Ask the assistant

```bash
python rag.py "Do I need programming experience?"
python rag.py "Which university accredits the course certificate?"
```

Example supported answer (wording can vary):

```json
{
  "text": "No prior programming experience is required. The course is suitable for beginners.",
  "references": ["data/prerequisites.md"]
}
```

Expected unsupported answer:

```json
{
  "text": "I don't know",
  "references": []
}
```

The English phrase “I don't know” implements the assignment's “No lo sé” behavior. The code, dataset, documentation, and responses use English.

## Pipeline and asynchronous entry point

```text
Question → Retriever → Formatted context → Prompt → Chat model → Pydantic parser
```

`get_rag_response(query: str)` in `rag.py` returns a validated `RAGResponse`. The LCEL pipeline uses `RunnablePassthrough.assign` to retain the question and retrieved documents, and `prompt | model | parser` for generation. `await chain.ainvoke(...)` executes retrieval and generation asynchronously; local initialization still performs ordinary synchronous setup.

```python
import asyncio
from rag import get_rag_response

response = asyncio.run(get_rag_response("How much does the course cost?"))
print(response.text)
print(response.references)
```

In a notebook with an active event loop, use `await get_rag_response(...)` directly.

The system prompt requires explicit contextual evidence, forbids outside knowledge and assumptions, and requests abstention when the context cannot fully answer the question. `PydanticOutputParser` validates the response structure. Additional checks reject duplicate citations, citations for an unknown answer, unsupported source paths, and factual answers with no citations. These checks do not establish that each claim is true; grounding still needs evaluation.

## Verification

```bash
python check_chunking.py
python check_rag.py
python ingest.py
```

- `check_chunking.py`: uses synthetic text in memory, checks that long text splits into multiple chunks of at most 600 tokens, and checks overlap on a controlled word-boundary fixture. No OpenAI API calls or database writes. The tokenizer may download its encoding on its first use.
- `check_rag.py`: makes two live RAG calls, checks the supported response and expected citation, and requires exact abstention with no references for the unsupported university accreditation question. Results are written to `rag_check_results.json`. Review the supported answer against its source as well.
- The second ingestion run should report an unchanged index and make no embedding calls.

The committed result files are example observed runs, not a guarantee of future output. Models and responses may change.

## Dataset and retrieval experiment

| File | Information |
| --- | --- |
| `data/prerequisites.md` | Audience and prior programming experience |
| `data/access_and_format.md` | Recorded format and three-month access |
| `data/syllabus.md` | Numbered classes and recorded API/AI material |
| `data/enrollment.md` | ARS 120,000 price, transfer payment, registration and contact |

The data is translated and reorganized from information supplied by the course owner. Registration fields describe what the form asks for; the dataset contains no actual student submissions. The owner's contact information is part of the supplied course description.

It remains unclear whether the three recorded API/AI classes are additional to the five numbered classes. No total class count, university accreditation, refund policy, or other missing condition is invented.

Reorganizing the documents by question intent improved the expected source's top-1 rank from 3/7 to 6/7 on a small exploratory set; top-3 remained 7/7. These are development examples, not an independent benchmark. Details are in `retrieval_evaluation.json`.

`evaluate_retrieval.py` reproduces the historical comparison only if both local indexes (`vectorstore/` and `vectorstore_v2/`) exist. Those indexes are intentionally not distributed, so this optional script is not part of the clean-clone setup. A fresh clone only needs `ingest.py`, `retriever.py`, `rag.py`, and the checks.

## Files

- `settings.py`: shared paths, embedding model, splitter settings, and top-k.
- `ingest.py`: loading, chunking, persistence, and reuse detection.
- `retriever.py`: asynchronous similarity search.
- `rag.py`: LCEL chain, prompt, output schema, and asynchronous public function.
- `check_rag.py`, `check_chunking.py`: verification scripts.
- `requirements.txt`: dependency ranges; `requirements-lock.txt`: installed versions from the tested Python 3.12 environment.

For the exact tested versions, install with `python -m pip install -r requirements-lock.txt`. Some transitive packages may need different versions on other platforms.

## Troubleshooting

- Missing key: complete `OPENAI_API_KEY` in `.env`.
- Authentication/quota/model-access errors: check your API account, billing, and chosen model. Do not paste credentials into logs or issues.
- Missing index: run `python ingest.py` first.
- Changed documents/settings: choose a new `VECTORSTORE_DIR` and reindex.
- Parser/validation errors: the model did not satisfy the output contract; the script raises an error rather than presenting an unvalidated answer.

## GitHub delivery

Commit source code, the four documents, `.env.example`, documentation, dependency files, and example test results. Exclude credentials, virtual environments, generated vector databases, and local backups. A web interface is not needed to run this project.
