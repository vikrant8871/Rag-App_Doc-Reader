import os
from dotenv import load_dotenv

load_dotenv()


# API KEYS
# Standardized env var name: GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ2_API_KEY")


# Embedding Model

# Best 384-dim models
# "sentence-transformers/all-MiniLM-L6-v2"
# "sentence-transformers/all-MiniLM-L12-v2"
# "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Recommended stronger models:
# - "sentence-transformers/all-MiniLM-L12-v2"
# - "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
# - "BAAI/bge-small-en"
# - "BAAI/bge-base-en"
# - "intfloat/e5-base-v2"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


# Vector DB
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "vector_db/")

# Ensure directory exists at import time (safe)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 700))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))


# Hybrid Search Configuration
# Retrieval mode:
# "vector" → only embedding search
# "bm25"   → only BM25 keyword
# "hybrid" → ensemble (recommended) key word + embedding search.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

# Ensemble scoring weights
HYBRID_WEIGHT_VECTOR = float(os.getenv("HYBRID_WEIGHT_VECTOR", 0.6))
HYBRID_WEIGHT_BM25 = float(os.getenv("HYBRID_WEIGHT_BM25", 0.4))

# BM25 configuration
BM25_ENABLED = True
BM25_CORPUS_FILE = os.path.join(VECTOR_DB_DIR, "bm25_corpus.pkl")
BM25_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "bm25_index.pkl")

# Number of documents retrieved for each method
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", 5))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", 7))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 5))
