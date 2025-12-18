import os
import uuid
import numpy as np
from typing import List, Any, Dict
import chromadb
from chromadb.config import Settings
from .config import VECTOR_DB_DIR


class VectorStore:
    """
    ChromaDB wrapper that stores embeddings and metadata.
    Handles: add, query, delete by source file.
    """

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = VECTOR_DB_DIR
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

        self._initialize_store()

  
    # Initialize Store (Persistent)
    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)

        # Create Chroma persistent client 
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        except Exception:
            # Older Chroma versions
            self.client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.persist_directory
                )
            )

        # We will pass embeddings manually → embedding_function=None
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG vector storage"},
            embedding_function=None
        )

        print(f"[INFO] VectorStore ready → Collection: {self.collection_name}")
        try:
            print(f"[INFO] Total documents currently stored: {self.collection.count()}")
        except:
            pass

    # Add Documents

    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> List[str]:
        """
        Add vector embeddings + doc chunks.
        Ensures correct source_file metadata.
        """

        if len(documents) != len(embeddings):
            raise ValueError("Documents count must match embeddings count.")

        ids, metadatas, docs_text, embeddings_list = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"

            # Ensure metadata exists
            meta = dict(doc.metadata) if hasattr(doc, "metadata") else {}

            # Enrich metadata
            meta["chunk_id"] = meta.get("chunk_id", i)
            meta["source_file"] = meta.get("source_file", "unknown")
            meta["content_length"] = len(doc.page_content) if hasattr(doc, "page_content") else 0
            meta["cleaned_text"] = meta.get("cleaned_text", "").strip()

            ids.append(doc_id)
            metadatas.append(meta)
            docs_text.append(doc.page_content if hasattr(doc, "page_content") else "")
            embeddings_list.append(emb.tolist())

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=docs_text
        )

        print(f"[INFO] Inserted {len(ids)} embeddings.")
        try:
            print(f"[INFO] New total documents: {self.collection.count()}")
        except:
            pass

        return ids

   
    # Delete all chunks belonging to one PDF
    
    def delete_by_source_file(self, source_filename: str):
        """
        Deletes all document chunks and embeddings belonging to a file.
        Identified using metadata["source_file"].
        """

        try:
            deleted = self.collection.delete(
                where={"source_file": source_filename}
            )
            print(f"[INFO] Deleted embeddings for: {source_filename}")
            return deleted
        except Exception as e:
            print(f"[ERROR] delete_by_source_file failed → {e}")
            return None

    
    # Query Collection
    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Query ChromaDB using explicit query_embedding.
        """

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            return {
                "ids": results.get("ids", [[]]),
                "documents": results.get("documents", [[]]),
                "metadatas": results.get("metadatas", [[]]),
                "distances": results.get("distances", [[]]),
            }

        except Exception as e:
            print(f"[ERROR] VectorStore query failed: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    
    # Clear entire collection (optional helper)
  
    def clear_all(self):
        """
        WARNING: Deletes the entire vector store.
        """
        try:
            self.client.delete_collection(self.collection_name)
            print("[INFO] All embeddings cleared.")
        except:
            pass

        self._initialize_store()
