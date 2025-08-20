# expense_gpt_response.py
# Diversified retrieval from Pinecone (MMR), then analytics + GPT narrative.
# Works with INDEX_NAME, PINECONE_API_KEY, OPENAI_API_KEY from env.

import os
import re
import pandas as pd
import numpy as np
import networkx as nx
from openai import OpenAI
from pinecone import Pinecone

# ---------- Config ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "auditexpense2")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

client = OpenAI(api_key=OPENAI_KEY)
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)

# ---------- Embedding ----------
def _embed_query(text: str) -> list:
    """Embed with same dimension as index (1024). Exported for app.py debug."""
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=1024
    ).data[0].embedding
    return emb

# ---------- MMR reranking ----------
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

    # pick most relevant first
    first = int(np.argmax(sim_q))
    selected.append(first)
    cand.remove(first)

    while len(selected) < min(k, len(doc_embs)) and cand:
        sel_embs = doc_embs[np.array(selected)]
        sim_to_sel = _cos_sim(doc_embs[cand], sel_embs).max(axis=1)
        scores = lam * sim_q[cand] - (1 - lam) * sim_to_sel
        pick_local = int(np.argmax(scores))
        pick = cand[pick_local]
        selected.append(pick)
        cand.remove(pick)
    return selected

# ---------- Retrieval + context ----------
def _retrieve_context(question: str, top_k: int = 10, candidate_k: int = 50,
                      per_doc_cap: int = 2, lam: float = 0.5):
    """
    Query Pinecone widely, diversify with MMR, cap per Doc_ID, and return:
      - rows: list of normalized metadata dicts per match
      - context: stitched text chunks for GPT to cite
    """
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

    vecs, rows = [], []
    for m in matches:
        vals = m.get("values")
        md = m.get("metadata") or {}
        if not vals or not md:
            continue
        # Normalize common fields and keep a unified content/doc_id
        norm = dict(md)
        norm["__]()
