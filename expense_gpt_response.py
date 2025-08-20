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
        norm["__content__"] = md.get("Content") or md.get("content") or md.get("text_chunk") or ""
        norm["__doc_id__"] = md.get("Doc_ID") or md.get("document_id") or md.get("DocID") or "unknown"
        vecs.append(vals)
        rows.append(norm)

    if not vecs:
        return [], ""

    doc_embs = np.array(vecs, dtype=np.float32)
    q = np.array(qvec, dtype=np.float32)

    # Oversample, then prune with per-doc cap
    selected = _mmr(doc_embs, q, k=min(top_k * 3, len(doc_embs)), lam=lam)

    per_doc_counts = {}
    seen_snip = set()
    chosen_rows, context_chunks = [], []
    for idx in selected:
        r = rows[idx]
        doc_id = str(r.get("__doc_id__", "unknown"))
        txt = (r.get("__content__") or "").strip()
        if not txt:
            continue
        key = (doc_id, txt[:160])
        if key in seen_snip:
            continue
        if per_doc_counts.get(doc_id, 0) >= per_doc_cap:
            continue

        chosen_rows.append(r)
        context_chunks.append(txt)
        per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
        seen_snip.add(key)

        if len(context_chunks) >= top_k:
            break

    context = "\n---\n".join(context_chunks)
    return chosen_rows, context

# ---------- Helpers: column mapping + cleaning ----------
def _first_col(df: pd.DataFrame, candidates: list[str], default: str = None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

_amount_rx = re.compile(r"[^0-9\.-]+")

def _to_amount(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series.astype(str).str.replace(_amount_rx, "", regex=True), errors="coerce")

def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

# ---------- Analytics from retrieved rows ----------
def _build_analytics(df: pd.DataFrame) -> dict:
    """
    Compute useful aggregates from the retrieved rows only (not whole corpus):
      - total_spend, n_txns, n_employees
      - top_spenders (by sum)
      - top_categories (by sum)
      - top_merchants (by sum)
      - monthly_trend (if date column present)
    All numbers are derived strictly from the retrieved context rows.
    """
    out = {}

    # Map likely columns
    emp_col = _first_col(df, ["Employee Name", "Employee", "Employee_Name", "EmployeeID", "Employee ID"])
    emp_id_col = _first_col(df, ["Employee ID", "EmployeeID"])
    cat_col = _first_col(df, ["Category", "Expense Category", "Expense Type", "Type"])
    merch_col = _first_col(df, ["Merchant", "Vendor", "Merchant Name"])
    amt_col = _first_col(df, ["Amount", "Total Amount", "Transaction Amount", "Net Amount", "USD Amount", "Base Amount"])
    date_col = _first_col(df, ["Date", "Transaction Date", "Submit Date", "Posted Date"])

    # Amounts
    if amt_col:
        df["_Amount"] = _to_amount(df[amt_col])
    else:
        df["_Amount"] = pd.Series([np.nan] * len(df))

    # Dates
    if date_col:
        df["_Date"] = _to_date(df[date_col])
    else:
        df["_Date"] = pd.NaT

    # Totals
    out["total_spend"] = float(np.nansum(df["_Amount"].values)) if len(df) else 0.0
    out["n_txns"] = int(len(df))
    if emp_id_col:
        out["n_employees"] = int(df[emp_id_col].nunique())
    elif emp_col:
        out["n_employees"] = int(df[emp_col].nunique())
    else:
        out["n_employees"] = None

    # Top spenders
    top_spenders = []
    if emp_col or emp_id_col:
        key = emp_col or emp_id_col
        grp = df.groupby(key)["_Amount"].sum().sort_values(ascending=False).head(10)
        top_spenders = [{"name": str(k), "spend": float(v)} for k, v in grp.items()]
    out["top_spenders"] = top_spenders

    # Top categories
    top_categories = []
    if cat_col:
        grp = df.groupby(cat_col)["_Amount"].sum().sort_values(ascending=False).head(10)
        top_categories = [{"category": str(k), "spend": float(v)} for k, v in grp.items()]
    out["top_categories"] = top_categories

    # Top merchants
    top_merchants = []
    if merch_col:
        grp = df.groupby(merch_col)["_Amount"].sum().sort_values(ascending=False).head(10)
        top_merchants = [{"merchant": str(k), "spend": float(v)} for k, v in grp.items()]
    out["top_merchants"] = top_merchants

    # Monthly trend
    monthly_trend = []
    if "_Date" in df.columns and df["_Date"].notna().any():
        tmp = df.dropna(subset=["_Date"]).copy()
        tmp["_Month"] = tmp["_Date"].dt.to_period("M").dt.to_timestamp()
        grp = tmp.groupby("_Month")["_Amount"].sum().sort_values()
        monthly_trend = [{"month": d.strftime("%Y-%m"), "spend": float(v)} for d, v in grp.items()]
    out["monthly_trend"] = monthly_trend

    return out

# ---------- Mutual approvals + approval rings ----------
def _mutuals_and_rings(df: pd.DataFrame) -> dict:
    # normalize key cols used for graph analysis
    emp_id = _first_col(df, ["Employee ID", "EmployeeID"]) or "Employee ID"
    appr_id = _first_col(df, ["Default Approver ID", "Approver ID", "ApproverID"]) or "Default Approver ID"
    cost_center = _first_col(df, ["Cost Center", "CostCenter", "Department"]) or "Cost Center"

    if emp_id not in df.columns: df[emp_id] = ""
    if appr_id not in df.columns: df[appr_id] = ""
    if cost_center not in df.columns: df[cost_center] = ""

    df[emp_id] = df[emp_id].astype(str).str.strip()
    df[appr_id] = df[appr_id].astype(str).str.strip()
    df[cost_center] = df[cost_center].astype(str).str.strip()

    pairs = set(zip(df[emp_id], df[appr_id]))
    mutual_pairs = set()
    for a, b in pairs:
        if (b, a) in pairs:
            mutual_pairs.add(tuple(sorted((a, b))))

    cc_map = (
        df.groupby(emp_id)[cost_center]
          .agg(lambda x: x.mode()[0] if not x.mode().empty else "")
          .to_dict()
    )
    cc_mutuals = []
    for a, b in mutual_pairs:
        cc_a, cc_b = cc_map.get(a, ""), cc_map.get(b, "")
        if cc_a and cc_a == cc_b:
            cc_mutuals.append((a, b, cc_a))

    # Approval rings (3+ cycles)
    G = nx.DiGraph()
    G.add_edges_from(pairs)
    raw_cycles = list(nx.simple_cycles(G))
    rings = [cycle for cycle in raw_cycles if len(cycle) >= 3]

    return {
        "mutual_pairs": sorted(list(mutual_pairs)),
        "cc_mutuals": sorted(cc_mutuals),
        "rings": rings
    }

# ---------- Main entry ----------
def query_expense_gpt(question: str, top_k: int = 10):
    try:
        # 1) Retrieve diversified rows + stitched context
        rows, context = _retrieve_context(
            question=question,
            top_k=top_k,
            candidate_k=max(30, top_k * 5),
            per_doc_cap=2,
            lam=0.5
        )
        if not rows or not context:
            return "I don’t know based on the provided context."

        # 2) Build DataFrame and analytics
        df = pd.DataFrame(rows)
        analytics = _build_analytics(df)
        gov = _mutuals_and_rings(df)

        # 3) Summaries for prompt
        mutual_pairs_text = "\n".join([f"{a} ⇄ {b}" for a, b in gov["mutual_pairs"]]) or "None detected"
        cc_mutuals_text = "\n".join([f"{a} ⇄ {b} (Cost Center: {cc})" for a, b, cc in gov["cc_mutuals"]]) or "None detected"
        rings_text = "\n".join([" → ".join(list(c) + [c[0]]) for c in gov["rings"]]) or "None detected"

        def _fmt_top(items, key_name, k=5):
            return "\n".join([f"{i+1}. {row[key_name]} — ${row['spend']:,.2f}"
                              for i, row in enumerate(items[:k])]) or "N/A"

        top_spenders_text = _fmt_top(
            [{"name": x["name"], "spend": x["spend"]} for x in analytics["top_spenders"]],
            "name"
        )
        top_categories_text = _fmt_top(
            [{"category": x["category"], "spend": x["spend"]} for x in analytics["top_categories"]],
            "category"
        )
        top_merchants_text = _fmt_top(
            [{"merchant": x["merchant"], "spend": x["spend"]} for x in analytics["top_merchants"]],
            "merchant"
        )
        monthly_text = "\n".join([f"{row['month']}: ${row['spend']:,.2f}" for row in analytics["monthly_trend"]]) or "N/A"

        # 4) Prompt (tell GPT to use computed numbers only)
        prompt = f"""
You are a travel & expense audit assistant.
Use ONLY the provided context and the computed analytics below. Do NOT invent numbers.
If the analytics do not contain what is needed, say: "I don’t know based on the provided context."

Context (snippets from Pinecone):
{context}

Computed analytics (derived strictly from the context rows):
- Totals: spend=${analytics['total_spend']:,.2f}, transactions={analytics['n_txns']}, employees={analytics['n_employees']}
- Top spenders:
{top_spenders_text}

- Top categories:
{top_categories_text}

- Top merchants:
{top_merchants_text}

- Monthly trend:
{monthly_text}

- Governance patterns:
  * Mutual Approval Pairs: 
{mutual_pairs_text}
  * High-Risk Mutuals (same cost center):
{cc_mutuals_text}
  * Approval Rings (3+ cycles):
{rings_text}

User question:
{question}

Rules:
- Base any figures you state on the computed analytics above.
- If a requested metric isn't available from these analytics, explain that the data isn't present in the retrieved context.
- Be concise and structured.
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
