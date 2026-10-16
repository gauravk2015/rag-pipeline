# RAG Integrations

## Pinecone RAG

A compact Retrieval-Augmented Generation (RAG) project that uses Pinecone for vector search and Anthropic Claude for grounded responses. The [`rag-pinecone`](rag-pinecone) directory contains the complete Python implementation for static records and PDF documents.

### Features

- Upsert and search structured records, such as product data.
- Extract text from PDFs, split it into overlapping token-based chunks, and store the chunks in Pinecone.
- Use Pinecone's integrated embedding model for both indexing and querying.
- Generate concise, context-aware answers with Anthropic Claude.

### Project structure

```text
rag-pinecone/
├── upsert_to_pinecone.py          # Upsert static records
├── search_pinecone.py             # Search records and generate answers
├── upsert_to_pinecone_pdf.py      # Extract, chunk, and upsert PDF content
├── search_pinecone_pdf.py         # Search PDF content and generate answers
└── .env.example                   # Environment variable template
```

### Setup

1. Create a Python environment and install the dependencies:

   ```bash
   pip install pinecone anthropic python-dotenv pymupdf tiktoken
   ```

2. Copy the environment template and configure the required values:

   ```bash
   cp rag-pinecone/.env.example rag-pinecone/.env
   ```

   Set the Pinecone credentials, index host, namespaces, and Anthropic model details in `.env`.

### Usage

Run commands from the project root:

```bash
python3 rag-pinecone/upsert_to_pinecone.py
python3 rag-pinecone/search_pinecone.py
```

For PDF-based retrieval:

```bash
python3 rag-pinecone/upsert_to_pinecone_pdf.py
python3 rag-pinecone/search_pinecone_pdf.py
```

The PDF workflow extracts text page by page and creates 500-token chunks with a 75-token overlap. PDF records retain source-file and page metadata so responses can reference the relevant document location.

### Configuration

Use separate Pinecone namespaces for structured records and PDF content:

- `PINECONE_NAMESPACE` — static records
- `PINECONE_NAMESPACE_PDF` — PDF chunks

Keep `.env` private and use `.env.example` only as a configuration template.
