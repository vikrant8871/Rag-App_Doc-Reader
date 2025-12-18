import os
import re
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,?;:()\-\/ ]", "", text)
    return text.strip()


def convert_text_to_docs(text, metadata=None):
    if metadata is None:
        metadata = {}

    cleaned = clean_text(text)

    return [
        Document(
            page_content=text,
            metadata={
                **metadata,
                "cleaned_text": cleaned,
            }
        )
    ]



# PDF Loader
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    
    filename = os.path.basename(file_path)

    for d in docs:
        d.metadata["cleaned_text"] = clean_text(d.page_content)
        d.metadata["source_file"] = filename      
        d.metadata["type"] = "pdf"

    return docs


def load_txt(file_path):
    filename = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    return convert_text_to_docs(
        text,
        metadata={"source_file": filename, "type": "txt"}
    )


def load_csv(file_path):
    filename = os.path.basename(file_path)

    try:
        df = pd.read_csv(file_path, dtype=str).fillna("")
    except Exception:
        df = pd.read_csv(file_path, dtype=str, encoding="latin-1").fillna("")

    text = df.to_string(index=False)

    return convert_text_to_docs(
        text,
        metadata={"source_file": filename, "type": "csv"}
    )


def load_excel(file_path):
    filename = os.path.basename(file_path)

    df = pd.read_excel(file_path, dtype=str).fillna("")
    text = df.to_string(index=False)

    return convert_text_to_docs(
        text,
        metadata={"source_file": filename, "type": "excel"}
    )


def load_file(file_path):
    lower = file_path.lower()
    if lower.endswith(".pdf"):
        return load_pdf(file_path)
    elif lower.endswith(".txt"):
        return load_txt(file_path)
    elif lower.endswith(".csv"):
        return load_csv(file_path)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        return load_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")



# Chunking docs
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        # Retain filename
        chunk.metadata["chunk_id"] = i
        chunk.metadata["cleaned_text"] = clean_text(chunk.page_content)

        # 🟢 Ensure source_file metadata is passed to every chunk
        if "source_file" not in chunk.metadata:
            # take metadata from parent doc if exists
            if hasattr(chunk, "metadata") and "source_file" in chunk.metadata:
                pass
            else:
                # Fallback if something was missing
                chunk.metadata["source_file"] = "unknown"

    return chunks
