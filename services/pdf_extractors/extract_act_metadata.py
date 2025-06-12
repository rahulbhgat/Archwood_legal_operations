# services/legal_rag_chroma.py

import os
import chromadb
import pandas as pd
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Chroma client
chroma_client = chromadb.Client(Settings())
COLLECTION_NAME = "legal_acts"
ACT_METADATA_COLLECTION_NAME = "legal_act_metadata"
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def prepare_rag_index(folder_path="Data/actmetadata"):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ Folder not found: {folder_path}")

    collection = (
        chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        if COLLECTION_NAME not in [col.name for col in chroma_client.list_collections()]
        else chroma_client.get_collection(name=COLLECTION_NAME)
    )

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        required = ["section_text", "title", "section_name"]
        if any(col not in df.columns for col in required):
            print(f"⚠️ Skipping {file} - missing required columns.")
            continue

        chunks, metadatas, ids = [], [], []

        for i, row in df.iterrows():
            text = str(row["section_text"])
            title = str(row["title"]).strip()
            section_name = str(row["section_name"]).strip()
            section_number = (
                str(row["section_number"]) if "section_number" in row else "N/A"
            )

            small_chunks = chunk_text(text)
            for j, chunk in enumerate(small_chunks):
                chunks.append(chunk)
                metadatas.append(
                    {
                        "source": file,
                        "act_title": title,
                        "section_number": section_number,
                        "section_name": section_name,
                    }
                )
                ids.append(f"{file}_{i}_{j}")

        print(f"📦 {file}: {len(chunks)} chunks")
        embeddings = embedder.encode(chunks).tolist()
        collection.add(
            documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

    print("✅ Legal section index ready.")


def prepare_act_metadata_index(csv_path="Data/actmetadata/act_metadata.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Metadata CSV not found.")

    collection = (
        chroma_client.get_or_create_collection(name=ACT_METADATA_COLLECTION_NAME)
        if ACT_METADATA_COLLECTION_NAME
        not in [col.name for col in chroma_client.list_collections()]
        else chroma_client.get_collection(name=ACT_METADATA_COLLECTION_NAME)
    )

    df = pd.read_csv(csv_path)
    df["title"] = (
        df["title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df = df.drop_duplicates(subset="title").dropna(subset=["title"])

    documents, metadatas, ids = [], [], []

    for i, row in df.iterrows():
        title = row["title"]
        documents.append(title)
        metadatas.append({"title": title})
        ids.append(f"act_meta_{i}")

    embeddings = embedder.encode(documents).tolist()
    collection.add(
        documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
    )
    print("✅ Act metadata index ready.")


def answer_query_with_rag(query, top_k=3):
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "⚠️ No matching sections found."

    formatted = ""
    for doc, meta in zip(documents, metadatas):
        title = meta.get("act_title", "Unknown")
        sec = meta.get("section_number", "N/A")
        name = meta.get("section_name", "Unknown")
        formatted += f"""
        <div style="border:1px solid #ccc; padding:16px; margin-bottom:10px; border-radius:10px;">
            <h4>📘 {title} — Section {sec}: {name}</h4>
            <p>{doc}</p>
        </div>
        """
    return formatted


def search_acts(query, top_k=5):
    collection = chroma_client.get_collection(name=ACT_METADATA_COLLECTION_NAME)
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "⚠️ No matching acts found."

    return "".join(
        f"""
        <div style="border:1px solid #ccc; padding:16px; margin-bottom:10px; border-radius:10px;">
            <h4>📘 {meta.get('title')}</h4>
        </div>
        """
        for doc, meta in zip(documents, metadatas)
    )


if __name__ == "__main__":
    prepare_rag_index()
    prepare_act_metadata_index()
    print("✅ Indexing complete.")
