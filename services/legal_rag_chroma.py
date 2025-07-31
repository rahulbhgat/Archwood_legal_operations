# Filename: services/legal_rag_chroma.py

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import streamlit as st

    use_cache = True
except ImportError:
    use_cache = False

# Load correct Chroma client
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Detect if running on Streamlit Cloud
is_cloud = os.getenv("IS_STREAMLIT_CLOUD", "false").lower() == "true"

# Use local folder unless on cloud
CHROMA_DB_DIR = None if is_cloud else "./chroma_db"
COLLECTION_NAME = "legal_acts"

# Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Create Chroma client (new API)
def create_chroma_client():
    if is_cloud:
        # Use in-memory client (optional: EphemeralClient)
        raise RuntimeError("Cloud setup for Chroma not configured")
    return PersistentClient(path=CHROMA_DB_DIR)


if use_cache:
    chroma_client = st.cache_resource(create_chroma_client)()
else:
    chroma_client = create_chroma_client()


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def prepare_rag_index(folder_path="Data/actmetadata", batch_size=5000):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ Folder not found: {folder_path}")

    existing_collections = [col.name for col in chroma_client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    else:
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            ),
        )

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            if df.empty:
                continue

            required_columns = ["section_text", "title", "section_name"]
            if any(col not in df.columns for col in required_columns):
                continue

            text_column = "section_text"
            act_title_column = "title"
            section_name_column = "section_name"
            section_number_column = (
                "section_number" if "section_number" in df.columns else None
            )

            all_chunks, all_metadatas, all_ids = [], [], []

            for i, row in df.iterrows():
                text = str(row[text_column])
                act_title = str(row[act_title_column])
                section_name = str(row[section_name_column])
                section_number = (
                    str(row[section_number_column]) if section_number_column else "N/A"
                )

                chunks = chunk_text(text)
                for j, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadatas.append(
                        {
                            "source": file,
                            "act_title": act_title,
                            "section_name": section_name,
                            "section_number": section_number,
                        }
                    )
                    all_ids.append(f"{file}_{i}_{j}")

            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i : i + batch_size]
                batch_metadatas = all_metadatas[i : i + batch_size]
                batch_ids = all_ids[i : i + batch_size]

                collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )


def answer_query_with_rag(query, top_k=3):
    if COLLECTION_NAME not in [col.name for col in chroma_client.list_collections()]:
        raise ValueError("RAG index not prepared. Run `prepare_rag_index()` first.")

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        ),
    )

    results = collection.query(query_texts=[query], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "⚠️ No matching documents found."

    response = ""
    for doc, meta in zip(documents, metadatas):
        act_title = meta.get("act_title", "Unknown Act")
        section_number = meta.get("section_number", "N/A")
        section_name = meta.get("section_name", "Unnamed Section")

        response += f"""<div style="border: 1px solid #ccc; padding: 16px; border-radius: 8px; margin-bottom: 12px;">
            <h4 style="color:#003366;">📘 {act_title} — Section {section_number}: {section_name}</h4>
            <p style="color:#333; font-size: 15px; line-height: 1.6;">{doc}</p>
        </div>"""

    return response
