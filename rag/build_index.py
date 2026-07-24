"""
Entry point for building the RAG vector index.

IMPORTANT: this file exists separately from rag_pipeline.py so that TfidfEmbedder
gets pickled with its real module path (`rag.rag_pipeline.TfidfEmbedder`) instead
of `__main__.TfidfEmbedder`. If you run `rag_pipeline.py` directly as a script
(or via `python3 -m rag.rag_pipeline`), Python sets that module's __name__ to
"__main__", and joblib pickles any class defined there under the "__main__"
module -- which then fails to unpickle from any other entry point (e.g. the
agent or the report generator). Always build the index via this script.

Run:
    python3 rag/build_index.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline().build_index()

    demo_queries = [
        "Why did revenue drop in Q2 2026?",
        "What is our pricing policy for discounts?",
        "What supplier risk do we have in Europe?",
    ]
    for q in demo_queries:
        print(f"\nQuery: {q}")
        for hit in rag.retrieve(q, k=2):
            print(f"  [{hit['source']}] score={hit['score']:.3f}  {hit['text'][:120].strip()}...")
