import openai
import os
from pinecone import Pinecone

# ✅ Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY or PINECONE_API_KEY")

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("auditexpense2")  # ✅ Your index name

def query_pinecone(question: str, top_k: int = 5):
    try:
        print(f"🔍 Embedding question: {question}")
        
        # ✅ Embed using matching model
        response = openai.embeddings.create(
            input=question,
            model="text-embedding-3-large",
            dimensions=1024
        )
        embedding = response.data[0].embedding

        # ✅ Query Pinecone
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )

        matches = results.get("matches", [])
        print(f"✅ Pinecone returned {len(matches)} results")
        return matches

    except Exception as e:
        print(f"❌ Error in query_pinecone(): {e}")
        raise RuntimeError(f"Pinecone query failed: {e}")
