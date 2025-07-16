from query import query_pinecone
from openai import OpenAI
import os
import pandas as pd  # Needed for logic and formatting

# ✅ Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_expense_gpt(question: str, top_k: int = 5):
    try:
        # ✅ Get Pinecone matches
        matches = query_pinecone(question, top_k)

        # ✅ Extract content and metadata
        context_chunks = []
        rows = []

        for match in matches:
            metadata = match["metadata"]
            content = metadata.get("content") or metadata.get("text_chunk") or str(metadata)
            context_chunks.append(content)
            rows.append(metadata)

        context = "\n---\n".join(context_chunks)

        # ✅ Load into DataFrame for business logic
        df = pd.DataFrame(rows)
        df["Employee ID"] = df["Employee ID"].astype(str).str.strip()
        df["Default Approver ID"] = df["Default Approver ID"].astype(str).str.strip()

        # ✅ Bi-directional mutual approval detection
        pairs = set(zip(df["Employee ID"], df["Default Approver ID"]))
        mutual_pairs = set()
        for a, b in pairs:
            if (b, a) in pairs:
                mutual_pairs.add(tuple(sorted((a, b))))

        mutual_pairs_text = "\n".join([f"{a} ⇄ {b}" for a, b in sorted(mutual_pairs)]) or "None detected"

        # ✅ Top 5 approvers by volume
        top_approvers = df["Default Approver"].value_counts().head(5).reset_index()
        top_approvers.columns = ["Approver", "Approval Count"]
        approver_text = "\n".join(f"{a.strip()} – {c} approvals" for a, c in top_approvers.values)

        # ✅ Compose GPT prompt
        prompt = f"""
You are a travel and expense audit assistant. Use the context below to answer the user's question.

Context:
{context}

Mutual Approval Pairs (based on bi-directional approval logic):
{mutual_pairs_text}

Top Approvers by Volume:
{approver_text}

Question:
{question}

Answer:
"""

        # ✅ Call GPT
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        print("❌ GPT error:", str(e))
        return f"Error: {str(e)}"
