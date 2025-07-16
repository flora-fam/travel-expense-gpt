from query import query_pinecone
from openai import OpenAI
import os
import pandas as pd
import networkx as nx  # NEW: for ring detection

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_expense_gpt(question: str, top_k: int = 5):
    try:
        matches = query_pinecone(question, top_k)

        context_chunks = []
        rows = []
        for match in matches:
            metadata = match["metadata"]
            content = metadata.get("content") or metadata.get("text_chunk") or str(metadata)
            context_chunks.append(content)
            rows.append(metadata)

        context = "\n---\n".join(context_chunks)
        df = pd.DataFrame(rows)

        # Clean fields
        df["Employee ID"] = df["Employee ID"].astype(str).str.strip()
        df["Default Approver ID"] = df["Default Approver ID"].astype(str).str.strip()
        df["Cost Center"] = df["Cost Center"].astype(str).str.strip()

        # === MUTUAL APPROVALS ===
        pairs = set(zip(df["Employee ID"], df["Default Approver ID"]))
        mutual_pairs = set()
        for a, b in pairs:
            if (b, a) in pairs:
                mutual_pairs.add(tuple(sorted((a, b))))

        # Cost center lookup per employee
        cost_center_map = df.groupby("Employee ID")["Cost Center"].agg(lambda x: x.mode()[0] if not x.mode().empty else "").to_dict()

        # Cost-center filtered mutuals
        cost_center_mutuals = []
        for a, b in mutual_pairs:
            cc_a = cost_center_map.get(a)
            cc_b = cost_center_map.get(b)
            if cc_a and cc_a == cc_b:
                cost_center_mutuals.append((a, b, cc_a))

        # === APPROVAL RINGS (GRAPH CYCLES) ===
        G = nx.DiGraph()
        G.add_edges_from(pairs)

        raw_cycles = list(nx.simple_cycles(G))
        approval_rings = [cycle for cycle in raw_cycles if len(cycle) >= 3]

        formatted_rings = []
        for cycle in approval_rings:
            ring_text = " → ".join(cycle + [cycle[0]])  # close the loop
            # Optional: attach shared cost center if all in same one
            shared_ccs = [cost_center_map.get(e) for e in cycle]
            cc = shared_ccs[0] if all(c == shared_ccs[0] for c in shared_ccs) else None
            if cc:
                ring_text += f" (Cost Center: {cc})"
            formatted_rings.append(ring_text)

        # === SUMMARIES ===
        mutual_pairs_text = "\n".join([f"{a} ⇄ {b}" for a, b in sorted(mutual_pairs)]) or "None detected"
        cost_center_mutuals_text = "\n".join([f"{a} ⇄ {b} in Cost Center: {cc}" for a, b, cc in cost_center_mutuals]) or "None detected"
        approval_rings_text = "\n".join(formatted_rings) or "None detected"

        # Top 5 approvers
        top_approvers = df["Default Approver"].value_counts().head(5).reset_index()
        top_approvers.columns = ["Approver", "Approval Count"]
        approver_text = "\n".join(f"{a.strip()} – {c} approvals" for a, c in top_approvers.values)

        # === FINAL PROMPT ===
        prompt = f"""
You are a travel and expense audit assistant. Use the context below to answer the user's question.

Context:
{context}

Mutual Approval Pairs (bi-directional logic):
{mutual_pairs_text}

High-Risk Mutual Approval Pairs (same cost center):
{cost_center_mutuals_text}

Detected Approval Rings (≥3-person circular chains):
{approval_rings_text}

Top Approvers by Volume:
{approver_text}

Question:
{question}

Answer:
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        print("❌ GPT error:", str(e))
        return f"Error: {str(e)}"
