from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import ollama

# =========================================================
# STEP 1: LOAD PDF
# =========================================================

PDF_FILE = "fighting-fraud-in-financial-services.pdf"

print("\nLoading PDF...\n")

reader = PdfReader(PDF_FILE)

text = ""

for page in reader.pages:
    extracted_text = page.extract_text()

    if extracted_text:
        text += extracted_text + "\n"

print("PDF Loaded Successfully")

# =========================================================
# STEP 2: CLEAN TEXT
# =========================================================

print("\nCleaning Text...\n")

text = text.replace("\n", " ")
text = re.sub(r"\s+", " ", text)

print("Text Cleaned")

# =========================================================
# STEP 3: SENTENCE-BASED CHUNKING
# =========================================================

print("\nCreating Chunks...\n")

sentences = re.split(r'(?<=[.!?]) +', text)

chunks = []

chunk_size = 2

for i in range(0, len(sentences), chunk_size):
    chunk = " ".join(sentences[i:i + chunk_size])

    if chunk.strip():
        chunks.append(chunk)

print(f"Total Chunks Created: {len(chunks)}")

# =========================================================
# STEP 4: LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading Embedding Model...\n")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding Model Loaded")

# =========================================================
# STEP 5: CREATE EMBEDDINGS
# =========================================================

print("\nCreating Embeddings...\n")

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

faiss.normalize_L2(embeddings)

print("Embeddings Created")

# =========================================================
# STEP 6: CREATE FAISS VECTOR DATABASE
# =========================================================

print("\nCreating FAISS Vector Database...\n")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"Total Vectors Stored: {index.ntotal}")

# =========================================================
# STEP 7: SEARCH + GENERATION
# =========================================================

def search(query, top_k=3):

    print("\n" + "=" * 80)
    print("USER QUESTION")
    print("=" * 80)

    print(query)

    # -----------------------------------------------------
    # Create Query Embedding
    # -----------------------------------------------------

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    # -----------------------------------------------------
    # Retrieve Similar Chunks
    # -----------------------------------------------------

    scores, indices = index.search(query_embedding, top_k)

    print("\n" + "=" * 80)
    print("TOP MATCHING CHUNKS")
    print("=" * 80)

    context = ""

    for rank, idx in enumerate(indices[0]):

        score = scores[0][rank]

        chunk = chunks[idx]

        if score > 0.85:
            similarity = "Very Strong Match"
        elif score > 0.70:
            similarity = "Strong Match"
        elif score > 0.50:
            similarity = "Moderate Match"
        else:
            similarity = "Weak Match"

        print(f"\nRANK #{rank+1}")
        print(f"Chunk ID: {idx}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Similarity Level: {similarity}")

        print("\nRetrieved Chunk:\n")
        print(chunk)

        print("\n" + "-" * 80)

        context += chunk + "\n\n"

    # -----------------------------------------------------
    # Prompt for Ollama
    # -----------------------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the information provided in the context below.

If the answer cannot be found in the context, reply:
"I couldn't find that information in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    # -----------------------------------------------------
    # Generate Answer using Ollama
    # -----------------------------------------------------

    print("\nGenerating Answer using Ollama...\n")

    try:

        response = ollama.chat(
            model="gemma3:12b",       # Change if using another model
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        print("=" * 80)
        print("FINAL GENERATED ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)

    except Exception as e:

        print("\nError communicating with Ollama.")
        print(e)

# =========================================================
# STEP 8: QUESTION LOOP
# =========================================================

print("\nRAG System Ready!")

while True:

    question = input("\nAsk a Question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("\nExiting RAG System...\n")
        break

    search(question)