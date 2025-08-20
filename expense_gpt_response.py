# expense_gpt_response.py

import os
import pandas as pd
import numpy as np
import networkx as nx
from openai import OpenAI
from pinecone import Pinecone

# ---------- Config ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "auditexpense2")        # point to your big index
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")              # default namespace

client = OpenAI(api_key=OPENAI_KEY)
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)

# ---------- Embedding ----------
def _embed_query(text: str) -> list:
    """Embed the query with the SAME dimension as your index (1024)."""
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=1024
    ).data[0].embedding
    return emb

# ---------- MMR reranking (diversify retrieval) ----------
def _l2norm(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / n

def _cos_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return _l2norm(A) @ _l2norm(B).T

def _mmr(doc_embs: np.ndarray, query_emb: np.ndarray, k: int, lam: float = 0.5):
    if doc_embs.size == 0:
        return []
    q = query_emb.reshape(1, -1)
    sim_q = _cos_sim(doc_embs, q).flatten()

    selected = []
    cand = list(range(len(doc_embs)))

    # start with most relevant
    first = int(np.argmax(sim_q))
    selected.append(first)
    cand.remove(first)

    while len(selected) < min(k, len(doc_embs)) and cand:
        sel_embs = doc_embs[np.array(selected)]
        sim_to_sel = _cos_sim(doc_embs[cand], sel_embs).max(axis=1)
        scores = lam * sim_q[cand] - (1 - lam) * sim_to_sel
        best_local = int(np.argmax(scores))
        best_idx = cand[best_local]
        selected.append(best_idx)
        cand.remove(best_idx)
    return selected

# ---------- Retrieval + context ----------
def _retrieve_context(question: str, top_k: int = 10, candidate_k: int = 50,
                      per_doc_cap: int = 2, lam: float = 0.5):
    """Query Pinecone widely, diversify, cap per Doc_ID, and return selected metadata rows + context text."""
    qvec = _embed_query(question)

    res = index.query(
        vector=qvec,
        top_k=candidate_k,
        include_metadata=True,
        include_values=True,
        namespace=NAMESPACE
    )
    matches = res.get("matches", []) or []
    if not matches:
        return [], ""

    # collect vectors + metadata
    vecs, rows = [], []
    for m in matches:
        vals = m.get("values")
        md = m.get("metadata") or {}
        if not vals or not md:
            continue
        # normalize likely metadata keys you use downstream
        # content field (different loaders may use different keys)
        md_norm = dict(md)
        md_norm["__content__"] = (
            md.get("Content") or md.get("content") or md.get("text_chunk") or ""
        )
        # Doc_ID helps diversity enforcement
        md_norm["__doc_id__"] = md.get("Doc_ID") or md.get("document_id") or md.get("DocID") or "unknown"
        vecs.append(vals)
        rows.append(md_norm)

    if not vecs:
        return [], ""

    doc_embs = np.array(vecs, dtype=np.float32)
    q = np.array(qvec, dtype=np.float32)

    # MMR to pick diverse candidates (oversample a bit, then cap)
    selected = _mmr(doc_embs, q, k=min(top_k * 3, len(doc_embs)), lam=lam)

    # per-doc cap + de-dupe
    per_doc_counts = {}
    seen_snip = set()
    chosen_rows = []
    context_chunks = []

    for idx in selected:
        r = rows[idx]
        doc_id = str(r.get("__doc_id__", "unknown"))
        txt = (r.get("__content__") or "").strip()
        if not txt:
            continue

        # dedupe short prefix
        key = (doc_id, txt[:160])
        if key in seen_snip:
            continue

        # enforce per-doc cap
        count = per_doc_counts.get(doc_id, 0)
        if count >= per_doc_cap:
            continue

        chosen_rows.append(r)
        context_chunks.append(txt)
        seen_snip.add(key)
        per_doc_counts[doc_id] = count + 1

        if len(context_chunks) >= top_k:
            break

    context = "\n---\n".join(context_chunks)
    return chosen_rows, context

# ---------- Main public function ----------
def query_expense_gpt(question: str, top_k: int = 10):
    try:
        # 1) retrieve diversified context (and rows for analytics)
        rows, context = _retrieve_context(
            question=question,
            top_k=top_k,
            candidate_k=max(30, top_k * 5),
            per_doc_cap=2,
            lam=0.5
        )

        if not rows or not context:
            # be explicit to avoid hallucinations
            return "I don’t know based on the provided context."

        # 2) Build DataFrame for analytics (mutual approvals, rings, etc.)
        df = pd.DataFrame(rows)

        # Map your expected columns; fill missing with safe defaults
        def col(name, fallback=""):
            return df[name] if name in df.columns else pd.Series([fallback] * len(df))

        # Normalize key columns (string + strip)
        df["Employee ID"] = col("Employee ID").astype(str).str.strip()
        df["Default Approver ID"] = col("Default Approver ID").astype(str).str.strip()
        df["Cost Center"] = col("Cost Center").astype(str).str.strip()

        # Optional human-readable approver name (if present)
        if "Default Approver" not in df.columns:
            df["Default Approver"] = col("Default Approver")

        # ---- MUTUAL APPROVALS (bi-directional) ----
        pairs = set(zip(df["Employee ID"], df["Default Approver ID"]))
        mutual_pairs = set()
        for a, b in pairs:
            if (b, a) in pairs:
                mutual_pairs.add(tuple(sorted((a, b))))

        # Cost-center per employee
        # mode() guard: some employees may have multiple rows
        cc_map = (
            df.groupby("Employee ID")["Cost Center"]
              .agg(lambda x: x.mode()[0] if not x.mode().empty else "")
              .to_dict()
        )

        # Higher-risk mutuals (same cost center)
        cc_mutuals = []
        for a, b in mutual_pairs:
            cc_a, cc_b = cc_map.get(a, ""), cc_map.get(b, "")
            if cc_a and cc_a == cc_b:
                cc_mutuals.append((a, b, cc_a))

        # ---- APPROVAL RINGS (3+ cycles) ----
        G = nx.DiGraph()
        G.add_edges_from(pairs)
        raw_cycles = list(nx.simple_cycles(G))
        approval_rings = [cycle for cycle in raw_cycles if len(cycle) >= 3]

        formatted_rings = []
        for cycle in approval_rings:
            ring_text = " → ".join(list(cycle) + [cycle[0]])  # close loop
            shared = [cc_map.get(e, "") for e in cycle]
            if shared and all(cc == shared[0] for cc in shared):
                ring_text += f" (Cost Center: {shared[0]})"
            formatted_rings.append(ring_text)

        # ---- Summaries for the prompt ----
        mutual_pairs_text = "\n".join([f"{a} ⇄ {b}" for a, b in sorted(mutual_pairs)]) or "None detected"
        cc_mutuals_text = "\n".join([f"{a} ⇄ {b} in Cost Center: {cc}" for a, b, cc in cc_mutuals]) or "None detected"
        rings_text = "\n".join(formatted_rings) or "None detected"

        # Top approvers by volume (if present)
        approver_text = "N/A"
        if "Default Approver" in df.columns and not df["Default Approver"].isna().all():
            top_approvers = df["Default Approver"].astype(str).str.strip().value_counts().head(5)
            approver_text = "\n".join(f"{name} – {cnt} approvals" for name, cnt in top_approvers.items()) or "N/A"

        # ---- Prompt (strictly use context) ----
        prompt = f"""
You are a travel and expense audit assistant.
Use ONLY the provided context and computed analytics. If the context does not support the answer, say:
"I don’t know based on the provided context."

Context:
{context}

Computed analytics available to you (derived strictly from the context rows):
- Mutual Approval Pairs (bi-directional):
{mutual_pairs_text}

- High-Risk Mutuals (same cost center):
{cc_mutuals_text}

- Detected Approval Rings (≥3-person cycles):
{rings_text}

- Top Approvers by Volume:
{approver_text}

User question:
{question}

Rules:
- Do not invent data. If unsure, say you don't know based on the provided context.
- Prefer concise, structured outputs (bullets or short sections).
"""

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content

    except Exception as e:
        print("❌ GPT error:", str(e))
        return f"Error: {str(e)}"
