# query.py

import openai
import os
from pinecone import Pinecone

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("auditexpense2")  # T&E index

def query_pinecone(question: str, top_k: int = 5):
    embedding = openai.embeddings.create(
        input=question,
        model="text-embedding-3-large",
        dimensions=1024
    ).data[0].embedding

    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return results['matches']
