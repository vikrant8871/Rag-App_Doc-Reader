# modules/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np
import re
from typing import List, Optional
from .config import EMBEDDING_MODEL


class EmbeddingManager:
    """
    Universal Embedding Manager that supports any sentence-transformers model:
    - MiniLM models (384 dim)
    - MPNet models (768 dim)
    - BGE models (384/768/1024 dim)
    - E5 models (512/768 dim)
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = None
        self.embedding_dim = None
        self._load_model()

    def _load_model(self):
        """Load embedding model safely and detect embedding dimension."""
        try:
            print(f"[INFO] Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Auto detect embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            print(f"[INFO] Model loaded successfully.")
            print(f"[INFO] Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            raise

    def get_dimension(self) -> int:
        """Return embedding dimension to external modules."""
        return self.embedding_dim

    def preprocess_text(self, text: str) -> str:
        """Clean input text for embedding."""
        if not text:
            return ""
        text = text.lower()
        text = text.replace("\x00", "")
        text = re.sub(r"[^a-zA-Z0-9.,%:;?!()$-]+", " ", text)
        return text.strip()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        if not self.model:
            raise ValueError("Embedding model not loaded")

        cleaned = [self.preprocess_text(t) for t in texts]

        embeddings = self.model.encode(
            cleaned,
            show_progress_bar=False,
            normalize_embeddings=True,  # Important for cosine similarity
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        if not self.model:
            raise ValueError("Embedding model not loaded")

        cleaned = self.preprocess_text(query)
        embedding = self.model.encode(
            [cleaned],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embedding)[0]
