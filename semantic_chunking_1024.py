
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:

    class RecursiveCharacterTextSplitter:
        """Lightweight local fallback with the same public name."""

        def __init__(
            self,
            chunk_size: int = 1024,
            chunk_overlap: int = 100,
            separators: list[str] | None = None,
        ) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", " ", ""]

        def split_text(self, text: str) -> list[str]:
            text = text.strip()
            if not text:
                return []

            segments = self._recursive_split(text, self.separators)
            chunks = self._merge_segments(segments)
            return [chunk for chunk in chunks if chunk.strip()]

        def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
            if len(text) <= self.chunk_size:
                return [text.strip()]

            separator = next((sep for sep in separators if sep and sep in text), None)
            if separator is None or len(separators) == 1:
                return self._hard_split(text)

            parts = [part for part in text.split(separator) if part.strip()]
            if len(parts) <= 1:
                return self._hard_split(text)

            segments: list[str] = []
            next_separators = separators[1:]
            for idx, part in enumerate(parts):
                piece = part.strip()
                if idx < len(parts) - 1:
                    piece = f"{piece}{separator}"

                if len(piece) > self.chunk_size and next_separators:
                    segments.extend(self._recursive_split(piece, next_separators))
                else:
                    segments.append(piece)

            return segments

        def _hard_split(self, text: str) -> list[str]:
            if self.chunk_size <= 0:
                return [text.strip()]

            step = max(1, self.chunk_size - max(self.chunk_overlap, 0))
            return [
                text[i : i + self.chunk_size].strip()
                for i in range(0, len(text), step)
                if text[i : i + self.chunk_size].strip()
            ]

        def _merge_segments(self, segments: list[str]) -> list[str]:
            if not segments:
                return []

            merged: list[str] = []
            current = ""

            for segment in segments:
                if not segment.strip():
                    continue

                candidate = f"{current} {segment}".strip() if current else segment.strip()
                if len(candidate) <= self.chunk_size:
                    current = candidate
                    continue

                if current.strip():
                    merged.append(current.strip())

                current = segment.strip()
                while len(current) > self.chunk_size:
                    merged.append(current[: self.chunk_size].strip())
                    current = current[self.chunk_size - self.chunk_overlap :].strip()

            if current.strip():
                merged.append(current.strip())

            if self.chunk_overlap <= 0 or len(merged) <= 1:
                return merged

            overlapped: list[str] = [merged[0]]
            for chunk in merged[1:]:
                prefix = overlapped[-1][-self.chunk_overlap :].strip()
                if prefix:
                    combined = f"{prefix} {chunk}".strip()
                    overlapped.append(combined[: self.chunk_size].strip())
                else:
                    overlapped.append(chunk.strip())

            return overlapped


SCRIPT_DIR = Path(__file__).resolve().parent
PDF_SEARCH_ROOTS = [SCRIPT_DIR / "data", SCRIPT_DIR / "pdfs"]
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
TOP_K = 5

PDFTOTEXT_BIN = shutil.which("pdftotext")
if PDFTOTEXT_BIN is None:
    bundled = Path(
        "/Users/ruuban/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdftotext"
    )
    PDFTOTEXT_BIN = str(bundled) if bundled.exists() else None


@dataclass(frozen=True)
class ChunkRecord:
    source: str
    page: int
    chunk_id: int
    text: str


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    relevant_sources: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    precision_at_k: float
    recall_at_k: float
    retrieved: list[ChunkRecord]
    relevant_hits: list[ChunkRecord]
    relevant_sources_found: tuple[str, ...]


EVALUATION_DATASET: list[EvaluationCase] = [
    EvaluationCase(
        question="How can artificial intelligence help detect insurance fraud?",
        relevant_sources=("artificial-intelligence-fraud-detection.pdf",),
    ),
    EvaluationCase(
        question="What technologies help organizations fight fraud and improve fraud management?",
        relevant_sources=("fraud-detection-prevention.pdf",),
    ),
    EvaluationCase(
        question="What makes a strong fraud solution and how do predictive fraud models help?",
        relevant_sources=("fighting-fraud-in-financial-services.pdf",),
    ),
    EvaluationCase(
        question="How do behavioral analysis and digital forensics support fraud detection?",
        relevant_sources=("artificial-intelligence-fraud-detection.pdf",),
    ),
    EvaluationCase(
        question="How should organizations combine manual review with automated fraud controls?",
        relevant_sources=(
            "fraud-detection-prevention.pdf",
            "fighting-fraud-in-financial-services.pdf",
        ),
    ),
    EvaluationCase(
        question="What are the main benefits of selecting the right fraud solution?",
        relevant_sources=("fighting-fraud-in-financial-services.pdf",),
    ),
]


def find_pdf_files(search_roots: Iterable[Path]) -> list[Path]:
    pdf_files: list[Path] = []
    for root in search_roots:
        if root.exists():
            pdf_files.extend(path for path in root.rglob("*.pdf") if path.is_file())
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for pdf_path in sorted(pdf_files):
        if pdf_path not in seen:
            seen.add(pdf_path)
            unique_files.append(pdf_path)
    return unique_files


def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def extract_page_texts(pdf_path: Path) -> list[str]:
    if PDFTOTEXT_BIN:
        try:
            result = subprocess.run(
                [PDFTOTEXT_BIN, str(pdf_path), "-"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            return [page.strip() for page in result.stdout.split("\f")]
        except Exception:
            pass

    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def extract_chunks_from_pdf(pdf_path: Path, splitter: RecursiveCharacterTextSplitter) -> list[ChunkRecord]:
    print(f"Loading PDF: {pdf_path.name}")
    records: list[ChunkRecord] = []
    page_texts = extract_page_texts(pdf_path)

    for page_number, page_text in enumerate(page_texts, start=1):
        cleaned = clean_text(page_text)
        if not cleaned:
            continue

        page_chunks = splitter.split_text(cleaned)
        for chunk_id, chunk_text in enumerate(page_chunks, start=1):
            if chunk_text.strip():
                records.append(
                    ChunkRecord(
                        source=pdf_path.name,
                        page=page_number,
                        chunk_id=chunk_id,
                        text=chunk_text.strip(),
                    )
                )

    print(f"  -> {len(records)} chunks")
    return records


def build_corpus(pdf_files: list[Path]) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    corpus: list[ChunkRecord] = []
    for pdf_path in pdf_files:
        try:
            corpus.extend(extract_chunks_from_pdf(pdf_path, splitter))
        except Exception as exc:
            print(f"Skipping {pdf_path.name}: {exc}")

    return corpus


def build_index(records: list[ChunkRecord]):
    if not records:
        raise RuntimeError("No PDF chunks were produced. Check the input PDFs.")

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [record.text for record in records]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"Embeddings indexed: {index.ntotal}")
    return model, index


def retrieve_top_k(
    question: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    records: list[ChunkRecord],
    top_k: int = TOP_K,
) -> list[tuple[ChunkRecord, float]]:
    query_embedding = model.encode([question], convert_to_numpy=True)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    k = min(top_k, len(records))
    scores, indices = index.search(query_embedding, k)

    results: list[tuple[ChunkRecord, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append((records[idx], float(score)))
    return results


def evaluate_case(
    case: EvaluationCase,
    retrieved: list[tuple[ChunkRecord, float]],
) -> EvaluationResult:
    relevant_sources = set(case.relevant_sources)
    relevant_hits = [record for record, _ in retrieved if record.source in relevant_sources]
    relevant_sources_found = tuple(sorted({record.source for record in relevant_hits}))

    denominator = len(retrieved) or 1
    precision = len(relevant_hits) / denominator
    recall = len(set(relevant_sources_found)) / len(relevant_sources)

    return EvaluationResult(
        question=case.question,
        precision_at_k=precision,
        recall_at_k=recall,
        retrieved=[record for record, _ in retrieved],
        relevant_hits=relevant_hits,
        relevant_sources_found=relevant_sources_found,
    )


def print_case_report(case: EvaluationCase, result: EvaluationResult, scores: list[tuple[ChunkRecord, float]]) -> None:
    print("\n" + "=" * 90)
    print(f"Question: {case.question}")
    print(f"Relevant sources: {', '.join(case.relevant_sources)}")
    print(f"Precision@{TOP_K}: {result.precision_at_k:.4f}")
    print(f"Recall@{TOP_K}: {result.recall_at_k:.4f}")
    print("-" * 90)
    for rank, (record, score) in enumerate(scores, start=1):
        is_relevant = "YES" if record.source in case.relevant_sources else "NO"
        print(
            f"{rank:>2}. {record.source} | page {record.page} | chunk {record.chunk_id} | "
            f"score {score:.4f} | relevant: {is_relevant}"
        )
        print(f"    {record.text[:220]}")
    print(f"Relevant sources found: {', '.join(result.relevant_sources_found) or 'None'}")


def main() -> None:
    pdf_files = find_pdf_files(PDF_SEARCH_ROOTS)
    if not pdf_files:
        raise FileNotFoundError("No PDF files were found under data/ or pdfs/.")

    print("Semantic Chunking Evaluation")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
    print(f"Retrieval: Top-{TOP_K} cosine similarity with FAISS IndexFlatIP")
    print(f"PDF files found: {len(pdf_files)}")

    records = build_corpus(pdf_files)
    print(f"Total chunks produced: {len(records)}")

    model, index = build_index(records)

    results: list[EvaluationResult] = []
    print("\nRunning evaluation set...")
    for case in EVALUATION_DATASET:
        retrieved_with_scores = retrieve_top_k(case.question, model, index, records)
        result = evaluate_case(case, retrieved_with_scores)
        results.append(result)
        print_case_report(case, result, retrieved_with_scores)

    mean_precision = mean(result.precision_at_k for result in results)
    mean_recall = mean(result.recall_at_k for result in results)

    print("\n" + "=" * 90)
    print("Final Evaluation Summary")
    print("=" * 90)
    print(f"Queries evaluated: {len(results)}")
    print(f"Average Precision@{TOP_K}: {mean_precision:.4f}")
    print(f"Average Recall@{TOP_K}: {mean_recall:.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
