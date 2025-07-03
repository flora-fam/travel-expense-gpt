from query import query_pinecone
from openai import OpenAI
import os

# ✅ Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_expense_gpt(question: str, top_k: int = 5):
    try:
        # ✅ Get Pinecone matches
        matches = query_pinecone(question, top_k)

        # ✅ Extract content from metadata — handle fallback keys
        context_chunks = [
            match['metadata'].get('content') or  # lowercase key
            match['metadata'].get('text_chunk') or
            str(match['metadata'])  # fallback for debugging
            for match in matches
        ]

        context = "\n---\n".join(context_chunks)

        # ✅ Structured system prompt
        prompt = f"""
You are a travel and expense assistant. Use the context below to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""

        # ✅ Call GPT with correct model
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # <-- you have access to this one
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        print("❌ GPT error:", str(e))
        return f"Error: {str(e)}"
