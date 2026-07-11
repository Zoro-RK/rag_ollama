from pathlib import Path
import shutil
import subprocess
import re

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_ROOT = Path(".")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemma3:12b"
CHUNK_SIZE = 3
CHUNK_OVERLAP = 1
TOP_K = 5
PDFTOTEXT_BIN = shutil.which("pdftotext")
if PDFTOTEXT_BIN is None:
    bundled_pdftotext = Path(
        "/Users/ruuban/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdftotext"
    )
    PDFTOTEXT_BIN = str(bundled_pdftotext) if bundled_pdftotext.exists() else None


def find_pdf_files(root: Path) -> list[Path]:
    pdf_files = sorted(
        path for path in root.rglob("*.pdf") if path.is_file()
    )
    return pdf_files


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_page_texts(pdf_path: Path) -> list[str]:
    if PDFTOTEXT_BIN:
        result = subprocess.run(
            [PDFTOTEXT_BIN, str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return [page.strip() for page in result.stdout.split("\f")]

    reader = PdfReader(str(pdf_path))
    page_texts = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    return page_texts


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    sentences = re.split(r"(?<=[.!?]) +", text)
    if not sentences:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []

    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def extract_document_chunks(pdf_path: Path) -> list[dict]:
    print(f"\nLoading PDF: {pdf_path}")

    records = []

    for page_number, extracted_text in enumerate(extract_page_texts(pdf_path), start=1):
        cleaned_text = clean_text(extracted_text)
        if not cleaned_text:
            continue

        page_chunks = chunk_text(cleaned_text)
        for chunk_number, chunk in enumerate(page_chunks, start=1):
            records.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk,
                }
            )

    print(f"Loaded {len(records)} chunks from {pdf_path.name}")
    return records


def build_corpus(pdf_files: list[Path]) -> list[dict]:
    all_chunks = []

    for pdf_path in pdf_files:
        try:
            all_chunks.extend(extract_document_chunks(pdf_path))
        except Exception as exc:
            print(f"Skipping {pdf_path}: {exc}")

    return all_chunks


def build_index(records: list[dict]):
    if not records:
        raise FileNotFoundError(
            "No readable PDF text was found. Add at least one text-based PDF to the repository."
        )

    print("\nLoading embedding model...\n")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded")

    print("\nCreating embeddings...\n")
    texts = [record["text"] for record in records]
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    print("Embeddings created")

    print("\nCreating FAISS vector database...\n")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Total vectors stored: {index.ntotal}")

    return model, index


def resolve_pdf_root() -> Path:
    if (PDF_ROOT / "data").exists() and (PDF_ROOT / "pdfs").exists():
        return PDF_ROOT

    candidate_dirs = [Path("data"), Path("pdfs")]
    for directory in candidate_dirs:
        if directory.exists():
            return directory

    return PDF_ROOT


def search(query: str, model, index, records: list[dict], top_k: int = TOP_K):
    try:
        import ollama
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ollama is not installed in this Python environment. Run the script in the project venv or install the ollama package."
        ) from exc

    print("\n" + "=" * 80)
    print("USER QUESTION")
    print("=" * 80)
    print(query)

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, min(top_k, len(records)))

    print("\n" + "=" * 80)
    print("TOP MATCHING CHUNKS")
    print("=" * 80)

    context_parts = []

    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        score = float(scores[0][rank])
        record = records[idx]

        if score > 0.85:
            similarity = "Very Strong Match"
        elif score > 0.70:
            similarity = "Strong Match"
        elif score > 0.50:
            similarity = "Moderate Match"
        else:
            similarity = "Weak Match"

        print(f"\nRANK #{rank + 1}")
        print(f"Source: {record['source']} (page {record['page']}, chunk {record['chunk']})")
        print(f"Chunk ID: {idx}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Similarity Level: {similarity}")
        print("\nRetrieved Chunk:\n")
        print(record["text"])
        print("\n" + "-" * 80)

        context_parts.append(
            f"Source: {record['source']} | Page: {record['page']} | Chunk: {record['chunk']}\n{record['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the information provided in the context below.
If the answer requires combining multiple documents, synthesize the shared guidance.
If the answer cannot be found in the context, reply exactly:
"I couldn't find that information in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    print("\nGenerating answer using Ollama...\n")

    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response["message"]["content"]

        print("=" * 80)
        print("FINAL GENERATED ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)
    except Exception as exc:
        print("\nError communicating with Ollama.")
        print(exc)


def main():
    root = resolve_pdf_root()
    pdf_files = find_pdf_files(root)

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files were found in the repository. Add the PDFs you want indexed and rerun the script."
        )

    print(f"Found {len(pdf_files)} PDF file(s).")

    records = build_corpus(pdf_files)
    model, index = build_index(records)

    print("\nRAG system ready!")

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("\nExiting RAG system...\n")
            break

        search(question, model, index, records)


if __name__ == "__main__":
    main()
