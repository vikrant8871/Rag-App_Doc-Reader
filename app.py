import streamlit as st
import os
from modules.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📘 RAG Chat Assistant (Groq + ChromaDB)")
st.write("Upload files → choose LLM → ask questions → get accurate RAG-powered responses.")

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.2-3b-preview",
    "mixtral-8x7b",
    "gemma2-9b-it"
]

st.sidebar.header("⚙️ Settings")

groq_api_key = st.sidebar.text_input("Groq API Key:", type="password")


selected_model = st.sidebar.selectbox("Choose LLM Model", GROQ_MODELS, index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key


@st.cache_resource
def load_pipeline():
    # Pipeline is instantiated once and shared across reruns
    return RAGPipeline()


rag = load_pipeline()


# UPDATE LLM
if st.sidebar.button("Update LLM"):
    success = rag.update_llm(model_name=selected_model, temperature=temperature)
    if success:
        st.sidebar.success("LLM updated.")
    else:
        st.sidebar.error("Failed to update LLM. Check logs.")


# OPTIONAL: HARD RESET BUTTON
if st.sidebar.button("Reset VectorDB & BM25"):
    rag.vector_db.clear_all()      # Clears chroma collection
    rag.corpus_tokens.clear()      # Clears BM25 corpus
    rag.corpus_ids.clear()
    rag.bm25 = None
    st.sidebar.success("Vector DB and BM25 Index Reset Successfully!")


# FILE UPLOAD + INGESTION
st.header("📂 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, CSV, Excel, or Text files",
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("uploaded_files", exist_ok=True)

    for file in uploaded_files:
        file_path = f"uploaded_files/{file.name}"

        # Save file
        with open(file_path, "wb") as f:
            f.write(file.read())

        with st.spinner(f"Processing {file.name}..."):
            chunks = rag.ingest_file(file_path)

        st.success(f"✓ Added {chunks} chunks from {file.name}")


# QUERY SECTION
st.header("💬 Chat with Your Documents")
query = st.text_input("Enter your question:")

col1, col2 = st.columns([1, 1])
top_k = col1.slider("Top K Retrieval", 1, 15, 5)
score_threshold = col2.slider("Minimum Similarity Score (0–1)", 0.0, 1.0, 0.2)

if st.button("Submit Query"):
    if not query.strip():
        st.warning("⚠ Please enter a question.")
    else:
        st.subheader("🧠 RAG Answer")

        with st.spinner("Generating answer..."):
            result = rag.query_advanced(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                return_context=True
            )

        st.success(result["answer"])
        st.write(f"**Confidence Score:** {round(result['confidence'], 3)}")

        with st.expander("📌 Retrieved Context"):
            st.write(result["context"])

        sources = result.get("sources", [])

        if sources:
            st.subheader("📄 Source Chunks Used")
            for src in sources:
                st.markdown(f"""
**File:** {src.get('source', 'unknown')}
**Chunk:** {src.get('chunk', '-') }
**Score:** {round(src.get('score', 0), 3)}

**Preview:**  
{src.get('preview', '')}
""")
                st.write("---")
        else:
            st.warning("No relevant sources found.")
