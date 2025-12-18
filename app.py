import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="University FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 University FAQ Chatbot")
st.write("Ask questions based on official university examination rules.")

# -------------------------------
# Initialize Chroma + Embeddings
# -------------------------------
@st.cache_resource
def load_resources():
    chroma_client = chromadb.Client(Settings(
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    ))

    collection = chroma_client.get_or_create_collection("university_faq")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return collection, embedder


collection, embedder = load_resources()

# -------------------------------
# Helper functions
# -------------------------------
def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_text_file(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        emb = embedder.encode(chunk).tolist()
        collection.add(
            ids=[f"{txt_path}_{i}"],
            documents=[chunk],
            embeddings=[emb],
            metadatas=[{"source": txt_path}]
        )


def answer_question(question):
    q_emb = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=2
    )

    context = "\n\n".join(results["documents"][0])

    if not context.strip():
        return "❌ I cannot find the answer in the official documents."

    return f"📌 **Answer (from university documents):**\n\n{context}"


# -------------------------------
# Ingest document (once)
# -------------------------------
if st.button("📥 Load Exam Rules"):
    ingest_text_file("exam_rules.txt")
    st.success("Exam rules loaded successfully!")

# -------------------------------
# Chat UI
# -------------------------------
st.subheader("💬 Ask a Question")

question = st.text_input(
    "Type your question here:",
    placeholder="e.g. What is the deadline for exam registration?"
)

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        answer = answer_question(question)
        st.markdown(answer)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("RAG-based University FAQ Chatbot | Streamlit Demo")
