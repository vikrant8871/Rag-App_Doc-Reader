
from typing import List, Dict, Any
import numpy as np
from langchain_core.documents import Document
from .vectorstore import VectorStore 
from .embedder import EmbeddingManager


class HybridRetriever:
    """
    Semantic retrieval using vectorstore + embedder.
    Returns items with id, document, similarity_score, distance, rank.
    """

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingManager):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        # 1. Embed query
        query_embedding = self.embedder.embed_query(query)

        # 2. Query vector database
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 3. Validate results structure
        if not results:
            return []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]   # Chroma still returns IDs automatically

        # If no docs retrieved
        if not docs:
            return []

        retrieved_items = []

        # 4. Build output
        for idx, (doc_id, text, meta, dist) in enumerate(zip(ids, docs, metas, distances)):

            # Convert distance → similarity
            try:
                similarity = 1.0 - float(dist)
            except:
                similarity = float(dist) if isinstance(dist, (int, float)) else 0.0

            # Skip low-scoring results
            if similarity < score_threshold:
                continue

            # Ensure metadata exists
            meta = meta or {}

            # Create LangChain-style Document
            document = Document(page_content=text, metadata=meta)

            retrieved_items.append({
                "id": doc_id,
                "document": document,
                "similarity_score": similarity,
                "distance": float(dist) if isinstance(dist, (int, float)) else 0.0,
                "rank": idx + 1,
                "bm25_score": 0.0,
                "combined_score": similarity
            })

        # 5. Sort by similarity
        retrieved_items.sort(key=lambda x: x["similarity_score"], reverse=True)

        return retrieved_items

