"""Compare two indexes using identical question embeddings; no answer generation."""

import json

import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from settings import BASE_DIR, COLLECTION_NAME, EMBEDDING_MODEL

CASES = [
    ('Do I need programming experience?', 'course_overview.md', 'prerequisites.md'),
    ('Can someone who has never written code take this course?', 'course_overview.md', 'prerequisites.md'),
    ('How much does the course cost and how can I pay?', 'enrollment_and_payment.md', 'enrollment.md'),
    ('How long can I access the videos and WhatsApp group?', 'course_overview.md', 'access_and_format.md'),
    ('Are the classes live or recorded?', 'course_overview.md', 'access_and_format.md'),
    ('Will I learn Pandas and how to move data between Excel and Python?', 'course_syllabus.md', 'syllabus.md'),
    ('How can I contact the course owner?', 'contact.md', 'enrollment.md'),
]


def main():
    load_dotenv(BASE_DIR / '.env')
    collections = [
        chromadb.PersistentClient(path=str(BASE_DIR / name)).get_collection(COLLECTION_NAME)
        for name in ('vectorstore', 'vectorstore_v2')
    ]
    for collection in collections:
        if (collection.metadata or {}).get('embedding_model') != EMBEDDING_MODEL:
            raise ValueError('Embedding model mismatch.')
    vectors = OpenAIEmbeddings(model=EMBEDDING_MODEL).embed_documents([c[0] for c in CASES])
    rows = []
    for (query, old_expected, new_expected), vector in zip(CASES, vectors):
        row = {'query': query}
        for label, collection, expected in zip(('before', 'after'), collections, (old_expected, new_expected)):
            result = collection.query(query_embeddings=[vector], n_results=4, include=['metadatas'])
            sources = [m['source'] for m in result['metadatas'][0]]
            rank = sources.index('data/' + expected) + 1
            row[label] = {'expected_source': 'data/' + expected, 'rank': rank, 'sources': sources}
        rows.append(row)
        print(f"{query}\n  Expected source rank: {row['before']['rank']} -> {row['after']['rank']}")
    for label in ('before', 'after'):
        top1 = sum(r[label]['rank'] == 1 for r in rows)
        top3 = sum(r[label]['rank'] <= 3 for r in rows)
        print(f'{label}: top-1 {top1}/{len(rows)}, top-3 {top3}/{len(rows)}')
    (BASE_DIR / 'retrieval_evaluation.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
