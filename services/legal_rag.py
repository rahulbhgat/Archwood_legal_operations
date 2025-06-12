# services/legal_rag.py (Updated for new RAG with 100% online tools)

import os
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# Load free Hugging Face embedding model (MiniLM)
@st.cache_resource(show_spinner="🔧 Loading embedding model...")
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Chunk all documents and store vectors
@st.cache_resource(show_spinner="🧠 Indexing legal documents...")
def prepare_rag_index(folder="Data/legal_acts_cleaned_texts"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    chunks, titles = [], []

    if not os.path.exists(folder):
        st.error("❌ Folder not found: " + folder)
        return [], [], None

    files = [f for f in os.listdir(folder) if f.lower().endswith(".txt")]
    if not files:
        st.warning("⚠️ No .txt files found.")
        return [], [], None

    for file in files:
        file_path = os.path.join(folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                st.warning(f"⚠️ Skipping empty file: {file}")
                continue

            split_chunks = splitter.split_text(text)
            if not split_chunks:
                st.warning(f"⚠️ No chunks created from: {file}")
                continue

            chunks.extend(split_chunks)
            titles.extend([file.replace(".txt", "")] * len(split_chunks))

    if not chunks:
        st.error("❌ No valid chunks to embed.")
        return [], [], None

    embedder = get_embedder()
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True, show_progress_bar=True)

    return chunks, titles, chunk_embeddings


# Retrieve top-matching chunks by cosine similarity
def retrieve_top_chunks(query, chunks, titles, chunk_embeddings, top_k=3):
    embedder = get_embedder()
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]

    if similarities is None or similarities.shape[0] == 0:
        return []

    # Convert to numpy for safe slicing
    similarities = similarities.cpu().numpy().flatten()
    top_k = min(top_k, len(similarities))
    if top_k == 0:
        return []

    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(titles[i], chunks[i]) for i in top_indices]



# Free Together.ai LLM call (Mistral-7B)
def query_llm(prompt, model="mistral-7b-instruct"):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a legal expert on Indian laws."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }
    res = requests.post(url, headers=headers, json=body)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

# Main RAG entry point for Legal Act Explorer
def answer_query_with_rag(query):
    chunks, titles, chunk_embeddings = prepare_rag_index()

    if not chunks or chunk_embeddings is None:
        return "⚠️ No valid documents to search. Please check your .txt files."

    top_docs = retrieve_top_chunks(query, chunks, titles, chunk_embeddings)

    if not top_docs:
        return "⚠️ No relevant legal content found for this query."

    context = "\n\n".join([f"From [{title}]:\n{chunk[:1000]}..." for title, chunk in top_docs])
    prompt = f"Use the following Indian legal documents to answer the question clearly:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return query_llm(prompt)


