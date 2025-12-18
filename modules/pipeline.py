import os
import pickle
from typing import List

# NLTK safe import
import nltk
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from .loader import load_file, chunk_documents
from .embedder import EmbeddingManager
from .vectorstore import VectorStore
from .retriever import HybridRetriever
from .llm import LLMManager, get_llm
from .config import (
    HYBRID_WEIGHT_VECTOR,
    HYBRID_WEIGHT_BM25,
    BM25_CORPUS_FILE,
    BM25_INDEX_FILE,
    RETRIEVAL_MODE
)


class RAGPipeline:
    """
    Hybrid RAG Pipeline:
    - ChromaDB (vector search)
    - BM25 (keyword search)
    - Safe re-ingestion
    - Large PDF stable
    """

    def __init__(self, embedding_model: str = None, llm_model: str = "llama-3.1-8b-instant"):
        self.embedder = EmbeddingManager(model_name=embedding_model)
        self.vector_db = VectorStore()
        self.retriever = HybridRetriever(self.vector_db, self.embedder)
        self.llm = LLMManager(model_name=llm_model)

        self.corpus_tokens: List[List[str]] = []
        self.corpus_ids: List[str] = []
        self.bm25 = None

        self._load_bm25_index()

   
    # BM25 Restore

    def _load_bm25_index(self):
        try:
            if os.path.exists(BM25_CORPUS_FILE) and os.path.exists(BM25_INDEX_FILE):
                with open(BM25_CORPUS_FILE, "rb") as f:
                    self.corpus_tokens = pickle.load(f)
                with open(BM25_INDEX_FILE, "rb") as f:
                    self.corpus_ids = pickle.load(f)

                if self.corpus_tokens:
                    self.bm25 = BM25Okapi(self.corpus_tokens)

                print(f"[INFO] BM25 loaded ({len(self.corpus_ids)} docs)")
        except Exception as e:
            print(f"[WARN] BM25 load failed: {e}")

   
    # Reset

    def reset_all(self):
        self.vector_db.clear_all()
        self.corpus_tokens = []
        self.corpus_ids = []
        self.bm25 = None

        for f in [BM25_CORPUS_FILE, BM25_INDEX_FILE]:
            if os.path.exists(f):
                os.remove(f)

  
    # Ingest
 
    def ingest_file(self, file_path: str) -> int:
        filename = os.path.basename(file_path)
        print(f"[INFO] Ingesting {filename}")

        # IMPORTANT ORDER
        self._remove_from_bm25(filename)
        self.vector_db.delete_by_source_file(filename)

        docs = load_file(file_path)
        chunks = chunk_documents(docs)

        for c in chunks:
            c.metadata["source_file"] = filename

        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        ids = self.vector_db.add_documents(chunks, embeddings)

        tokens = [
            word_tokenize(c.page_content.lower()[:5000])
            for c in chunks
        ]

        self.corpus_tokens.extend(tokens)
        self.corpus_ids.extend(ids)
        self.bm25 = BM25Okapi(self.corpus_tokens)

        with open(BM25_CORPUS_FILE, "wb") as f:
            pickle.dump(self.corpus_tokens, f)
        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump(self.corpus_ids, f)

        print(f"[SUCCESS] {len(chunks)} chunks added")
        return len(chunks)

  
    # BM25 Cleanup (FIXED)
   
    def _remove_from_bm25(self, filename: str):
        if not self.corpus_ids:
            return

        kept_tokens, kept_ids = [], []

        try:
            data = self.vector_db.collection.get(include=["ids", "metadatas"])
            ids = data.get("ids", [])
            metas = data.get("metadatas", [])
        except Exception:
            return

        meta_map = {}
        for i, m in zip(ids, metas):
            if isinstance(m, dict):
                meta_map[i] = m
            elif isinstance(m, list) and m:
                meta_map[i] = m[0]
            else:
                meta_map[i] = {}

        for tokens, doc_id in zip(self.corpus_tokens, self.corpus_ids):
            if meta_map.get(doc_id, {}).get("source_file") != filename:
                kept_tokens.append(tokens)
                kept_ids.append(doc_id)

        self.corpus_tokens = kept_tokens
        self.corpus_ids = kept_ids
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

   
    # Hybrid Search
    
    def hybrid_search(self, query: str, top_k=5, score_threshold: float = 0.0):
        vector_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        if not self.bm25 or RETRIEVAL_MODE == "vector":
            for r in vector_results:
                r["bm25_score"] = 0.0
                r["combined_score"] = r["similarity_score"]
            return vector_results

        tokens = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokens)

        for r in vector_results:
            if r["id"] in self.corpus_ids:
                idx = self.corpus_ids.index(r["id"])
                r["bm25_score"] = float(bm25_scores[idx])
            else:
                r["bm25_score"] = 0.0

        sem = [r["similarity_score"] for r in vector_results]
        bm = [r["bm25_score"] for r in vector_results]

        def norm(vals):
            lo, hi = min(vals), max(vals)
            return [0 if hi == lo else (v - lo) / (hi - lo) for v in vals]

        n_sem = norm(sem)
        n_bm = norm(bm)

        for i, r in enumerate(vector_results):
            r["combined_score"] = (
                HYBRID_WEIGHT_VECTOR * n_sem[i] +
                HYBRID_WEIGHT_BM25 * n_bm[i]
            )

        return sorted(vector_results, key=lambda x: x["combined_score"], reverse=True)


   
    # Answer
   
    def answer(self, query: str, top_k=5, score_threshold: float = 0.0):
        results = self.hybrid_search(query, top_k, score_threshold)

        context = ""
        sources = []

        for r in results:
            doc = r["document"]
            chunk_id = doc.metadata.get("chunk_id", "-")

            context += f"[Chunk {chunk_id}] {doc.page_content}\n\n"
            sources.append({
                "source": doc.metadata.get("source_file", "unknown"),
                "chunk": chunk_id,
                "score": r.get("combined_score", 0.0),
                "preview": doc.page_content[:250] + "..."
            })

        answer_text = self.llm.generate_answer(query, context)

        return {
            "answer": answer_text,
            "context": context,
            "confidence": max([s["score"] for s in sources], default=0.0),
            "sources": sources
        }


    # Streamlit Wrapper 
   
    def query_advanced(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        return_context: bool = True
    ):
        return self.answer(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

    