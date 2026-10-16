"""
Upsert PDF text to Pinecone. Like Employee Handbook, Policies etc.
Local PDF → Extract text → Split into 500-token chunks with 75-token overlap → Upsert chunks to Pinecone → Pinecone generates embeddings.
Creating chunks, embedding etc. is managed by Pinecone.
Using integrated embedding model- llama-text-embed-v2, which can be selected while creating index.
The PDF file path is expected as input in CLI.
Oct 2026
"""

import os
from pathlib import Path

import pymupdf
import tiktoken
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Configuration
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
API_KEY = os.getenv("PINECONE_API_KEY")
NAMESPACE = os.getenv("PINECONE_NAMESPACE_PDF")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
BATCH_SIZE = 50

encoding = tiktoken.get_encoding("cl100k_base")


def extract_and_chunk(pdf_path: Path) -> list[dict]:
    """Extract PDF text and create token-based chunks."""
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            tokens = encoding.encode(text)
            chunk_number = 1

            for start in range(0, len(tokens), step):
                chunk_tokens = tokens[start:start + CHUNK_SIZE]

                if not chunk_tokens:
                    break

                chunk_id = (
                    f"{pdf_path.stem}"
                    f"#page-{page_number}"
                    f"#chunk-{chunk_number}"
                )

                chunks.append({
                    "_id": chunk_id,
                    "text": encoding.decode(chunk_tokens),
                    "source": pdf_path.name,
                    "page": page_number,
                })

                chunk_number += 1

                if start + CHUNK_SIZE >= len(tokens):
                    break

    return chunks


def main():
    if not API_KEY or not INDEX_HOST or not NAMESPACE:
        raise ValueError(
            "Check PINECONE_API_KEY, PINECONE_INDEX_HOST, "
            "and PINECONE_NAMESPACE_PDF in .env"
        )

    pdf_input = input("Enter PDF path: ").strip().strip('"')
    pdf_path = Path(pdf_input)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Extract and chunk
    chunks = extract_and_chunk(pdf_path)

    if not chunks:
        raise ValueError("No text extracted from the PDF.")

    # Connect to Pinecone
    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(host=INDEX_HOST)

    # Upsert in batches; Pinecone generates embeddings internally
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        index.upsert_records(
            namespace=NAMESPACE,
            records=batch,
        )

        # Print IDs only after the batch succeeds
        for record in batch:
            print(record["_id"])

    print(f"\nTotal chunks upserted: {len(chunks)}")


if __name__ == "__main__":
    main()
