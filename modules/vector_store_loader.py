import ollama
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def get_embeddings(texts):
    def embed(chunk):
        try:
            response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
            return response['embedding']
        except Exception as e:
            print(f"[Embedding Error] {e}")
            return [0.0] * 768  # Return dummy vector on error

    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(embed, texts))
    return np.array(embeddings)

def smart_chunk_text(text, max_chunks=200):
    total_length = len(text)
    chunk_size = max(100, total_length // max_chunks)
    chunks = [text[i:i + chunk_size] for i in range(0, total_length, chunk_size)]
    return chunks[:max_chunks]

def load_vector_store(text: str):
    chunks = smart_chunk_text(text)

    if not chunks:
        raise ValueError("Text could not be chunked.")

    vectors = get_embeddings(chunks)

    if vectors.size == 0 or len(vectors[0]) == 0:
        raise ValueError("Failed to generate valid embeddings.")

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2
