# expense_gpt_response.py
# Global analytics if a CSV is available; otherwise diversified RAG subset.
# Env vars used:
#   OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME (e.g., auditexpense2), PINECONE_NAMESPACE (optional)
#   EXPENSES_CSV_URL (optional) OR EXPENSES_CSV_PATH (optional) for global analytics

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

CSV_URL = os.getenv("EXPENSES_CSV_URL")     # set one of these for global analytics
CSV_PATH = os.getenv("EXPENSES_CSV_PATH")   # e.g., /data/expenses.csv

client = OpenAI(api_key=OPENAI_KEY)
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)

# ---------- Embedding ----------
def _embed_query(text: str) -> list:
    """Embed with the same dimension as your index (1024). Exported for app.py debug endpoints."""
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=1024
    ).data[0].embedding
    return emb

# ---------- MMR reranking (for diversified RAG subset) ----------
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
        pick_local = int(np.argmax(scores))
        pick = cand[pick_local]
        selected.append(pick)
        cand.remove(pick)
    return selected

# ---------- Retrieval + context (RAG subset path) ----------
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

        # Normalize and keep unified content/doc_id
        norm = dict(md)
        # Build fallback content from metadata if no 'content' field
        content = (
            md.get("Content") or md.get("content") or md.get("text_chunk") or ""
        ).strip()
        if not content:
            # Construct a readable line from key fields you showed in your sample
            content = (
                f"Doc_ID:{md.get('Doc_ID') or md.get('document_id') or ''} | "
                f"Employee:{(md.get('Employee') or '').strip()} ({(md.get('Employee ID') or '').strip()}) | "
                f"Approver:{(md.get('Default Approver') or '').strip()} ({(md.get('Default Approver ID') or '').strip()}) | "
                f"Approved Amount (rpt):{md.get('Approved Amount (rpt)')} | "
                f"Expense Amount (rpt):{md.get('Expense Amount (rpt)')} | "
                f"Expense Type:{(md.get('Expense Type') or '').strip()} | "
                f"Vendor:{(md.get('Vendor') or '').strip()} | "
                f"Cost Center:{(md.get('Cost Center') or '').strip()} | "
                f"Transaction Date:{md.get('Transaction Date')}"
            ).strip()

        norm["__content__"] = content
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

# ---------- Column helpers ----------
def _first_col(df: pd.DataFrame, candidates: list[str], default: str | None = None):
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

# ---------- Analytics (shared) ----------
def _build_analytics(df: pd.DataFrame) -> dict:
    """
    Compute aggregates:
      - total_spend, n_txns, n_employees
      - top_spenders, top_categories, top_merchants
      - monthly_trend
    Uses your schema: prefers 'Approved Amount (rpt)' then 'Expense Amount (rpt)'.
    """
    out = {}

    # Prioritize your fields
    emp_col = _first_col(df, ["Employee", "Employee Name", "Employee_Name", "EmployeeID", "Employee ID"])
    emp_id_col = _first_col(df, ["Employee ID", "EmployeeID"])
    cat_col = _first_col(df, ["Expense Type", "Parent Expense Type Name", "Spend Category Code", "Category"])
    merch_col = _first_col(df, ["Vendor", "Merchant", "Merchant Name"])
    # Amount preference: Approved -> Expense
    amt_col = _first_col(df, ["Approved Amount (rpt)", "Expense Amount (rpt)", "Total Amount", "Transaction Amount", "USD Amount", "Base Amount"])
    date_col = _first_col(df, ["Transaction Date", "Report Date", "Created Date", "Date", "Paid Date/Time"])

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
    key = emp_col or emp_id_col
    if key:
        grp = df.groupby(key)["_Amount"].sum().sort_values(ascending=False).head(10)
        top_spenders = [{"name": str(k).strip(), "spend": float(v)} for k, v in grp.items()]
    out["top_spenders"] = top_spenders

    # Top categories
    top_categories = []
    if cat_col:
        grp = df.groupby(cat_col)["_Amount"].sum().sort_values(ascending=False).head(10)
        top_categories = [{"category": str(k).strip(), "spend": float(v)} for k, v in grp.items()]
    out["top_categories"] = top_categories

    # Top merchants
    top_merchants = []
    if merch_col:
        grp = df.groupby(merch_col)["_Amount"].sum().sort_values(ascending=False).head(10)
        top_merchants = [{"merchant": str(k).strip(), "spend": float(v)} for k, v in grp.items()]
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

def _mutuals_and_rings(df: pd.DataFrame) -> dict:
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

# ---------- Global CSV (optional) ----------
_GLOBAL_DF: pd.DataFrame | None = None

def _load_global_df() -> pd.DataFrame | None:
    global _GLOBAL_DF
    if _GLOBAL_DF is not None:
        return _GLOBAL_DF
    try:
        if CSV_PATH and os.path.exists(CSV_PATH):
            _GLOBAL_DF = pd.read_csv(CSV_PATH)
        elif CSV_URL:
            _GLOBAL_DF = pd.read_csv(CSV_URL)
        else:
            _GLOBAL_DF = None
    except Exception as e:
        print("⚠️ Failed to load global CSV:", e)
        _GLOBAL_DF = None
    return _GLOBAL_DF

# ---------- Main entry ----------
def query_expense_gpt(question: str, top_k: int = 10):
    try:
        # 0) Try to load global CSV once (if provided)
        gdf = _load_global_df()

        # 1) Always retrieve a diversified subset for context + governance patterns
        rows, context = _retrieve_context(
            question=question,
            top_k=top_k,
            candidate_k=max(30, top_k * 5),
            per_doc_cap=2,
            lam=0.5
        )
        if not rows and not gdf is not None:
            return "I don’t know based on the provided context."

        df_subset = pd.DataFrame(rows)
        # Subset analytics (for narrative and when no CSV is provided)
        subset_analytics = _build_analytics(df_subset) if len(df_subset) else None
        subset_gov = _mutuals_and_rings(df_subset) if len(df_subset) else {"mutual_pairs": [], "cc_mutuals": [], "rings": []}

        # Global analytics if CSV is available (exact over your dataset)
        global_analytics = _build_analytics(gdf) if gdf is not None and len(gdf) else None

        # ---- Summaries for prompt ----
        def fmt_pairs(pairs): 
            return "\n".join([f"{a} ⇄ {b}" for a, b in pairs]) if pairs else "None detected"
        def fmt_cc_mutuals(items):
            return "\n".join([f"{a} ⇄ {b} (Cost Center: {cc})" for a, b, cc in items]) if items else "None detected"
        def fmt_rings(rings):
            return "\n".join([" → ".join(list(c) + [c[0]]) for c in rings]) if rings else "None detected"

        subset_mutuals = fmt_pairs(subset_gov["mutual_pairs"])
        subset_cc_mutuals = fmt_cc_mutuals(subset_gov["cc_mutuals"])
        subset_rings = fmt_rings(subset_gov["rings"])

        def _fmt_top(items, key_name, k=5, label="name"):
            if not items:
                return "N/A"
            lines = []
            for i, row in enumerate(items[:k]):
                klabel = row.get(key_name) or row.get(label) or "Unknown"
                spend = row.get("spend", 0.0)
                lines.append(f"{i+1}. {str(klabel).strip()} — ${spend:,.2f}")
            return "\n".join(lines) if lines else "N/A"

        # Subset (RAG) texts
        if subset_analytics:
            rag_top_spenders = _fmt_top(subset_analytics["top_spenders"], "name")
            rag_top_categories = _fmt_top(subset_analytics["top_categories"], "category")
            rag_top_merchants = _fmt_top(subset_analytics["top_merchants"], "merchant")
            rag_monthly = ("\n".join([f"{row['month']}: ${row['spend']:,.2f}" for row in subset_analytics["monthly_trend"]])
                           if subset_analytics["monthly_trend"] else "N/A")
            rag_totals = f"spend=${subset_analytics['total_spend']:,.2f}, transactions={subset_analytics['n_txns']}, employees={subset_analytics['n_employees']}"
        else:
            rag_top_spenders = rag_top_categories = rag_top_merchants = rag_monthly = "N/A"
            rag_totals = "N/A"

        # Global texts
        if global_analytics:
            g_top_spenders = _fmt_top(global_analytics["top_spenders"], "name")
            g_top_categories = _fmt_top(global_analytics["top_categories"], "category")
            g_top_merchants = _fmt_top(global_analytics["top_merchants"], "merchant")
            g_monthly = ("\n".join([f"{row['month']}: ${row['spend']:,.2f}" for row in global_analytics["monthly_trend"]])
                         if global_analytics["monthly_trend"] else "N/A")
            g_totals = f"spend=${global_analytics['total_spend']:,.2f}, transactions={global_analytics['n_txns']}, employees={global_analytics['n_employees']}"
        else:
            g_top_spenders = g_top_categories = g_top_merchants = g_monthly = g_totals = None  # not available

        # ---- Prompt ----
        # We give the model BOTH: exact global analytics (if CSV provided) and subset analytics (from RAG).
        # Rule: Prefer GLOBAL analytics if available; otherwise use subset analytics.
        prompt = f"""
You are a travel & expense audit assistant.

Use the following data sources in this order of preference:
1) GLOBAL analytics (exact, computed from the full CSV provided at startup), if present.
2) SUBSET analytics (computed from the retrieved Pinecone rows), if global is not present.
Never invent numbers. If neither contains the needed metric, say:
"I don’t know based on the provided data."

Context snippets (from Pinecone; helpful for citing examples):
{context}

GLOBAL analytics (exact over full dataset): {'[present]' if global_analytics else '[not available]'}
- Totals: {g_totals or 'N/A'}
- Top spenders:
{g_top_spenders or 'N/A'}

- Top categories:
{g_top_categories or 'N/A'}

- Top merchants:
{g_top_merchants or 'N/A'}

- Monthly trend:
{g_monthly or 'N/A'}

SUBSET analytics (from retrieved context only): {'[present]' if subset_analytics else '[not available]'}
- Totals: {rag_totals}
- Top spenders:
{rag_top_spenders}

- Top categories:
{rag_top_categories}

- Top merchants:
{rag_top_merchants}

- Monthly trend:
{rag_monthly}

Governance patterns from SUBSET rows:
- Mutual Approval Pairs: 
{subset_mutuals}
- High-Risk Mutuals (same cost center):
{subset_cc_mutuals}
- Approval Rings (3+ cycles):
{subset_rings}

User question:
{question}

Rules:
- Prefer GLOBAL analytics if available; otherwise use SUBSET analytics.
- Be concise and structured (bullets/tables ok).
- If a requested metric isn’t available from the provided analytics, say you don’t know.
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
