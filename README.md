# RAG PDF Answering System

`rag_generation.py` ianalyzes multiple PDF documents, chunks their content into smaller context units, retrieves the most relevant chunks for a user question, and generates grounded answers strictly from the source text. The system is designed to reduce hallucination by relying only on retrieved PDF context and returning an answer only when the information is supported by the documents, builds a FAISS vector index, and answers questions with Ollama using only the retrieved context.

## What it does

- Scans the repository for PDF files.
- Extracts text from each PDF page.
- Splits page text into overlapping sentence chunks.
- Embeds the chunks with `sentence-transformers`.
- Stores embeddings in a FAISS index.
- Retrieves the most relevant chunks for a user question.
- Sends the retrieved context to Ollama for the final answer.

## PDF sources

The script currently reads PDFs from these folders if they exist:

- `data/`
- `pdfs/`

It also searches recursively under the selected root, so any additional PDFs placed in those folders will be included automatically.

## Encrypted PDFs

Some PDFs may be encrypted. When available, the script uses Poppler `pdftotext` to extract text from those files so they can still be chunked and indexed.

If `pdftotext` is not available, the script falls back to `pypdf`.

## Requirements

- Python 3.10+
- `faiss`
- `numpy`
- `pypdf`
- `sentence-transformers`
- `ollama`
- Poppler `pdftotext` recommended for encrypted PDFs

## Setup

Install the Python packages in your environment:

```bash
pip install faiss-cpu numpy pypdf sentence-transformers ollama
```

Make sure Ollama is installed and running locally, and pull the model used by the script:

```bash
ollama pull gemma3:12b
```

## Run

From the project root:

```bash
python3 rag_generation.py
```

The script will:

1. Find all PDFs in the configured folders.
2. Build the vector index.
3. Start an interactive question loop.

Type a question and press Enter. Type `exit` to quit.

## How chunking works

The current chunking strategy is sentence-based:

- Each page is cleaned by removing extra whitespace.
- Sentences are grouped into chunks of 3 sentences.
- The chunks overlap by 1 sentence to preserve context between chunks.

This improves retrieval quality compared with fixed-size, non-overlapping chunks.

## Output

For each question, the script prints:

- the top matching chunks
- the source PDF name
- the page number
- the chunk number
- the similarity score
- the final Ollama-generated answer

## Notes

- If no readable PDFs are found, the script raises an error instead of running with an empty index.
- If `ollama` is not installed in the active Python environment, the script will stop when it reaches the answer-generation step.
- The script is intended for local document QA, not for general web search.
