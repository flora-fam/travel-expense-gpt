# query.py

import openai
import os
from pinecone import Pinecone

# ✅ Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# ✅ Use your auditexpense2 index
index = pc.Index("auditexpense2")

def query_pinecone(question: str, top_k: int = 5):
    embedding = openai.Embedding.create(
        input=question,
        model="text-embedding-3-large",
        dimensions=1024
    ).data[0].embedding

    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return results['matches']
