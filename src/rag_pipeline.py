from src.embeddings import model
import numpy as np

def rag_answer(query, vector_store, texts):
    query_embedding = model.encode([query])
    indices = vector_store.search(query_embedding, k=5)

    retrieved = [texts[i] for i in indices]

    summary = "\n".join(retrieved)
    return f"Top relevant tweets:\n{summary}"