# query.py

import openai
import os
from pinecone import Pinecone

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("auditexpense2")

def query_pinecone(question: str, top_k: int = 5):
    try:
        print("🔍 Embedding question:", question)
        response = openai.embeddings.create(
            input=question,
            model="text-embedding-3-large",
            dimensions=1024
        )
        embedding = response.data[0].embedding

        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        print(f"✅ Pinecone returned {len(results['matches'])} results")
        return results['matches']

    except Exception as e:
        print("❌ Error in query_pinecone:", str(e))
        raise
