# Grounded PDF RAG System

This repository contains a local PDF question-answering workflow and two standalone evaluation scripts for comparing semantic chunk sizes.

The main RAG pipeline reads PDFs, chunks the extracted text, embeds the chunks, stores them in FAISS, retrieves the most relevant context with cosine similarity, and generates a grounded answer from the retrieved text.

## Repository Contents

- `rag_generation.py` - main interactive RAG script
- `semantic_chunking_512.py` - standalone evaluation script for semantic chunking with 512-character chunks
- `semantic_chunking_1024.py` - standalone evaluation script for semantic chunking with 1024-character chunks
- `data/` and `pdfs/` - PDF inputs used by the scripts

## Main RAG Workflow

1. Load PDF files from the repository folders.
2. Extract page text from each PDF.
3. Chunk the extracted text.
4. Convert chunks into embeddings.
5. Store embeddings in a FAISS vector index.
6. Embed the user question.
7. Retrieve the top-k most similar chunks.
8. Send the retrieved context to the LLM for a grounded answer.

## Indexing and Retrieval Methods

- Indexing method: `FAISS IndexFlatIP`
- Retrieval method: top-5 cosine similarity search
- Embeddings are L2-normalized before indexing, so inner product acts like cosine similarity.

## Main Script Details

`rag_generation.py` is the interactive RAG application. It:

- scans `data/` and `pdfs/` for PDF files
- extracts text from each page
- chunks text into overlapping sentence windows
- builds embeddings using `all-MiniLM-L6-v2`
- stores vectors in `FAISS IndexFlatIP`
- retrieves the top 5 relevant chunks
- sends the retrieved context to Ollama for the final answer

### PDF Extraction

The script uses `pdftotext` when available to support encrypted PDFs. If `pdftotext` is not available, it falls back to `pypdf`.

### Answering Behavior

The prompt is constrained to the retrieved context so the answer stays grounded in the source PDFs and reduces hallucination risk.

## Semantic Chunking Evaluation Scripts

These scripts are independent from the main RAG application and are used only to compare chunk sizes.

### `semantic_chunking_512.py`

- Method: Semantic Chunking
- Chunk size: 512 characters
- Chunk overlap: 100 characters
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector database: FAISS `IndexFlatIP`
- Retrieval: top-5 cosine similarity
- Metrics: Precision and Recall

### `semantic_chunking_1024.py`

- Method: Semantic Chunking
- Chunk size: 1024 characters
- Chunk overlap: 100 characters
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector database: FAISS `IndexFlatIP`
- Retrieval: top-5 cosine similarity
- Metrics: Precision and Recall

### Evaluation Flow

Both evaluation scripts follow the same workflow:

1. Load PDFs.
2. Extract text from each page.
3. Chunk the text using `RecursiveCharacterTextSplitter`.
4. Create embeddings.
5. Build a FAISS vector index.
6. Retrieve the top 5 chunks for each evaluation question.
7. Compute Precision and Recall using the evaluation dataset.
8. Print per-question and summary statistics.

## Evaluation Dataset

The evaluation scripts include a small source-labeled dataset inside the script. Each question is mapped to one or more relevant PDF sources, and the retrieved top-5 chunks are checked against those sources to compute:

- Precision@5
- Recall@5

## Requirements

- Python 3.10+
- `faiss`
- `numpy`
- `pypdf`
- `sentence-transformers`
- `ollama` for the main RAG script
- `langchain_text_splitters` recommended for the semantic chunking evaluation scripts
- Poppler `pdftotext` recommended for encrypted PDFs

## Setup

Install the core Python packages:

```bash
pip install faiss-cpu numpy pypdf sentence-transformers ollama
```

For the semantic chunking scripts, install the text splitter package:

```bash
pip install langchain-text-splitters
```

If you want to use the main RAG script with Ollama, make sure the model is available locally:

```bash
ollama pull gemma3:12b
```

## Run

Main RAG application:

```bash
python3 rag_generation.py
```

Semantic chunking evaluation with 512-character chunks:

```bash
python3 semantic_chunking_512.py
```

Semantic chunking evaluation with 1024-character chunks:

```bash
python3 semantic_chunking_1024.py
```

## Notes

- The main RAG script is interactive and continues accepting questions until you type `exit`.
- The semantic chunking scripts are evaluation-only and print retrieval statistics instead of generating LLM answers.
- If no readable PDFs are found, the scripts stop with an error instead of building an empty index.
- The evaluation scripts are intentionally separate from the main RAG implementation so the chunk-size comparison stays isolated and reproducible.
