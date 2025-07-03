# expense_gpt_response.py

from query import query_pinecone
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_expense_gpt(question: str, top_k: int = 5):
    try:
        matches = query_pinecone(question, top_k)
        context_chunks = [match['metadata'].get('Content', '') for match in matches]
        context = "\n---\n".join(context_chunks)

        prompt = f"""
You are a travel and expense assistant. Use the context below to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        print("❌ GPT error:", str(e))
        return f"Error: {str(e)}"
