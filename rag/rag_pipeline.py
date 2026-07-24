"""
EDIP - Retrieval-Augmented Generation (RAG) Pipeline
========================================================
Ingests unstructured company documents (annual reports, policies, board meeting
notes, market research) and builds a searchable vector index so the AI
consultant can answer questions with citations back to the source document.

Embedding model: TF-IDF (scikit-learn) --  a lightweight, dependency-free stand-in
for a production embedding model (e.g. OpenAI/Voyage/Sentence-Transformers
embeddings). Swap `TfidfEmbedder` below for a real embedding API in production;
the FAISS index and retrieval interface do not need to change.

Vector store: FAISS (flat L2 index over normalized vectors == cosine similarity)

Usage:
    from rag.rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    rag.build_index()                 # one-time (or whenever docs change)
    hits = rag.retrieve("Why did Q2 2026 revenue drop in Europe?", k=4)
    for h in hits:
        print(h["source"], h["score"], h["text"][:100])
"""
import json
import re
import numpy as np
import faiss
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "rag" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100


def chunk_text(text, source, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\n{2,}", "\n\n", text.strip())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append({"source": source, "text": chunk})
        if end == len(text):
            break
        start = end - overlap
    return chunks


class TfidfEmbedder:
    """Lightweight, offline embedder used as a swap-in placeholder for a real
    embedding API (OpenAI text-embedding-3, Voyage, Sentence-Transformers, etc).
    Interface (fit / transform -> dense numpy array) matches what a real
    embedding client would provide, so swapping it out is a one-file change."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2048, stop_words="english", ngram_range=(1, 2))

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        mat = self.vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms


class RAGPipeline:
    def __init__(self):
        self.embedder = TfidfEmbedder()
        self.index = None
        self.chunks = []

    def build_index(self):
        all_chunks = []
        for path in sorted(DOCS_DIR.glob("*.txt")):
            text = path.read_text()
            all_chunks.extend(chunk_text(text, source=path.name))

        texts = [c["text"] for c in all_chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.transform(texts)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs == cosine similarity
        index.add(vectors)

        self.index = index
        self.chunks = all_chunks

        faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
        joblib.dump(self.embedder, INDEX_DIR / "embedder.joblib")
        with open(INDEX_DIR / "chunks.json", "w") as f:
            json.dump(all_chunks, f, indent=2)
        print(f"RAG index built: {len(all_chunks)} chunks from "
              f"{len(list(DOCS_DIR.glob('*.txt')))} documents -> {INDEX_DIR}")
        return self

    def load(self):
        self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        self.embedder = joblib.load(INDEX_DIR / "embedder.joblib")
        with open(INDEX_DIR / "chunks.json") as f:
            self.chunks = json.load(f)
        return self

    def retrieve(self, query, k=4):
        if self.index is None:
            self.load()
        qvec = self.embedder.transform([query])
        scores, idxs = self.index.search(qvec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({"source": chunk["source"], "text": chunk["text"], "score": float(score)})
        return results


if __name__ == "__main__":
    print("NOTE: run `python3 rag/build_index.py` instead of this file directly -- building "
          "the index here would pickle TfidfEmbedder under the wrong module path ('__main__'), "
          "which then fails to load from the agent or report generator.")
    import sys
    sys.exit(1)
