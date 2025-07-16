from query import query_pinecone
from openai import OpenAI
import os
import pandas as pd  # NEW: needed for mutual approval logic

# ✅ Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_expense_gpt(question: str, top_k: int = 5):
    try:
        # ✅ Get Pinecone matches
        matches = query_pinecone(question, top_k)

        # ✅ Extract content from metadata — handle fallback keys
        context_chunks = []
        rows = []

        for match in matches:
            metadata = match["metadata"]
            content = metadata.get("content") or metadata.get("text_chunk") or str(metadata)
            context_chunks.append(content)
            rows.append(metadata)  # NEW: collect all rows

        context = "\n---\n".join(context_chunks)

        # ✅ Load into DataFrame for mutual approval detection
        df = pd.DataFrame(rows)
        df["Employee ID"] = df["Employee ID"].astype(str).str.strip()
        df["Default Approver ID"] = df["Default Approver ID"].astype(str).str.strip()

        pairs = set(zip(df["Employee ID"], df["Default Approver ID"]))
        mutual_pairs = set()
        for a, b in pairs:
            if (b, a) in pairs:
                mutual_pairs.add(tuple(sorted((a, b))))

        mutual_pairs_text = "\n".join([f"{a} ⇄ {b}" for a, b in sorted(mutual_pairs)]) or "None detected"

        # ✅ Structured system prompt with mutual pairs injected
        prompt = f"""
You are a travel and expense audit assistant. Use the context below to answer the user's question.

Context:
{context}

Mutual Approval Pairs (based on bi-directional approval logic):
{mutual_pairs_text}

Question:
{question}

Answer:"""

        # ✅ Call GPT with correct model
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        print("❌ GPT error:", str(e))
        return f"Error: {str(e)}"
