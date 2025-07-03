# expense_gpt_response.py

import openai
from query import query_pinecone

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

        print("📤 Sending to GPT:", prompt[:500])  # only show the start of prompt
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content
        print("✅ GPT answered.")
        return answer

    except Exception as e:
        print("❌ Error in query_expense_gpt:", str(e))
        return f"Error: {str(e)}"
