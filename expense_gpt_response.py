# expense_gpt_response.py
# Diversified RAG + optional GLOBAL summaries stored inside Pinecone (no CSV required).
# Env vars:
#   OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME (e.g., auditexpense2), PINECONE_NAMESPACE (optional)

import os
import re
import json
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
    """Embed with same dimension as your index (1024)."""
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=1024
    ).data[0].embedding
    return emb

# ---------- Retrieval + context (RAG subset) ----------
def _retrieve_context(question: str, top_k: int = 10, candidate_k: int = 200,
                      per_doc_cap: int = 2):
    """
    Query Pinecone for a large candidate set, then:
      1) pick by highest score while capping per Doc_ID to avoid duplicates
      2) if still short, do a second pass ignoring the cap so we always reach top_k
    NOTE: This version does NOT require embedding values (no MMR). It uses scores.
    """
    # 1) Embed the question
    qvec = _embed_query(question)

    # 2) Query wide. Do NOT ask for values; many indexes don't return them.
    res = index.query(
        vector=qvec,
        top_k=candidate_k,
        include_metadata=True,
        include_values=False,   # <-- important: we don't rely on raw vectors
        namespace=NAMESPACE
    )

    # Pinecone response can be object-like or dict-like
    try:
        matches = res.matches  # type: ignore[attr-defined]
    except Exception:
        matches = (res.get("matches") or [])  # type: ignore[index]

    if not matches:
        return [], ""

    # 3) Normalize into a list of (score, row_dict, content, doc_id)
    items = []
    for m in matches:
        try:
            md = m.metadata or {}            # object style
            score = float(m.score)           # object style
        except Exception:
            md = m.get("metadata") or {}     # dict style
            score = float(m.get("score", 0.0))

        # Build content string (fallback to key fields if no 'content')
        content = (
            md.get("Content")
            or md.get("content")
            or md.get("text_chunk")
            or ""
        )
        content = str(content).strip()

        if not content:
            # Construct readable fallback from your schema
            content = (
                f"Doc_ID:{md.get('Doc_ID') or md.get('ID') or md.get('document_id') or ''} | "
                f"Employee:{(md.get('Employee') or '').strip()} ({(md.get('Employee ID') or md.get('EmployeeID') or '').strip()}) | "
                f"Approver:{(md.get('Default Approver') or '').strip()} ({(md.get('Default Approver ID') or '').strip()}) | "
                f"Approved Amount (rpt):{md.get('Approved Amount (rpt)')} | "
                f"Expense Amount (rpt):{md.get('Expense Amount (rpt)')} | "
                f"Expense Type:{(md.get('Expense Type') or '').strip()} | "
                f"Vendor:{(md.get('Vendor') or '').strip()} | "
                f"Cost Center:{(md.get('Cost Center') or '').strip()} | "
                f"Transaction Date:{md.get('Transaction Date')}"
            ).strip()

        doc_id = (
            md.get("Doc_ID")
            or md.get("ID")
            or md.get("document_id")
            or md.get("DocID")
            or "unknown"
        )

        row = dict(md)
        row["__content__"] = content
        row["__doc_id__"] = str(doc_id)

        items.append((score, row, content, str(doc_id)))

    if not items:
        return [], ""

    # 4) Sort by score (desc)
    items.sort(key=lambda x: x[0], reverse=True)

    # 5) First pass: enforce per-doc cap
    per_doc_counts = {}
    chosen_rows, context_chunks = [], []
    seen_snip = set()

    for _, row, content, doc_id in items:
        if len(context_chunks) >= top_k:
            break
        if not content:
            continue
        if per_doc_counts.get(doc_id, 0) >= per_doc_cap:
            continue
        key = (doc_id, content[:160])
        if key in seen_snip:
            continue

        chosen_rows.append(row)
        context_chunks.append(content)
        per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
        seen_snip.add(key)

    # 6) Second pass: if still short, ignore per-doc cap to fill to top_k
    if len(context_chunks) < top_k:
        for _, row, content, doc_id in items:
            if len(context_chunks) >= top_k:
                break
            if not content:
                continue
            key = (doc_id, content[:160])
            if key in seen_snip:
                continue
            chosen_rows.append(row)
            context_chunks.append(content)
            seen_snip.add(key)

    context = "\n---\n".join(context_chunks)
    return chosen_rows, context

# ---------- Column helpers & analytics on retrieved subset ----------
_amount_rx = re.compile(r"[^0-9\.-]+")

def _first_col(df: pd.DataFrame, candidates: list[str], default: str | None = None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def _to_amount(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series.astype(str).str.replace(_amount_rx, "", regex=True), errors="coerce")

def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def _build_subset_analytics(df: pd.DataFrame) -> dict:
    """
    Aggregates from the retrieved subset only (NOT the whole corpus).
    Used for examples/explanations; not for authoritative totals.
    """
    out = {}

    emp_col = _first_col(df, ["Employee", "Employee Name", "Employee_Name", "EmployeeID", "Employee ID"])
    emp_id_col = _first_col(df, ["Employee ID", "EmployeeID"])
    cat_col = _first_col(df, ["Expense Type", "Parent Expense Type Name", "Spend Category Code", "Category"])
    merch_col = _first_col(df, ["Vendor", "Merchant", "Merchant Name"])
    amt_col = _first_col(df, ["Approved Amount (rpt)", "Expense Amount (rpt)", "Total Amount", "Transaction Amount", "USD Amount", "Base Amount"])
    date_col = _first_col(df, ["Transaction Date", "Report Date", "Created Date", "Date", "Paid Date/Time"])

    if amt_col:
        df["_Amount"] = _to_amount(df[amt_col])
    else:
        df["_Amount"] = pd.Series([np.nan] * len(df))

    if date_col:
        df["_Date"] = _to_date(df[date_col])
    else:
        df["_Date"] = pd.NaT

    out["total_spend"] = float(np.nansum(df["_Amount"].values)) if len(df) else 0.0
    out["n_txns"] = int(len(df))
    if emp_id_col:
        out["n_employees"] = int(df[emp_id_col].nunique())
    elif emp_col:
        out["n_employees"] = int(df[emp_col].nunique())
    else:
        out["n_employees"] = None

    top_spenders = []
    key = emp_col or emp_id_col
    if key:
        grp = df.groupby(key)["_Amount"].sum().sort_values(ascending=False).head(10)
        top_spenders = [{"name": str(k).strip(), "spend": float(v)} for k, v in grp.items()]
    out["top_spenders"] = top_spenders

    top_categories = []
    if cat_col:
        grp = df.groupby(cat_col)["_Amount"].sum().sort_values(ascending=False).head(10)
        top_categories = [{"category": str(k).strip(), "spend": float(v)} for k, v in grp.items()]
    out["top_categories"] = top_categories

    top_merchants = []
    if merch_col:
        grp = df.groupby(merch_col)["_Amount"].sum().sort_values(ascending=False).head(10)
        top_merchants = [{"merchant": str(k).strip(), "spend": float(v)} for k, v in grp.items()]
    out["top_merchants"] = top_merchants

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

# ---------- GLOBAL summaries (stored in Pinecone) ----------
SUMMARY_IDS = [
    "__summary__:totals",
    "__summary__:top_spenders",
    "__summary__:top_categories",
    "__summary__:top_merchants",
    "__summary__:monthly_trend",
]

def _fetch_global_summaries():
    """Fetches summary docs by fixed IDs from Pinecone (namespace aware)."""
    try:
        res = index.fetch(ids=SUMMARY_IDS, namespace=NAMESPACE)
        vectors = getattr(res, "vectors", None)
        if vectors is None:
            vectors = res.get("vectors", {})  # type: ignore
        if not vectors:
            return None

        out = {}
        for sid, vec in vectors.items():
            md = getattr(vec, "metadata", None) or vec.get("metadata") or {}
            out[sid] = md
        return out if out else None
    except Exception:
        return None

def _format_global_text(glob):
    """Make human-readable blocks from the fetched summaries."""
    if not glob:
        return None, None, None, None, None

    totals = glob.get("__summary__:totals") or {}
    sp = totals.get("total_spend")
    ntx = totals.get("n_txns")
    ne = totals.get("n_employees")
    totals_text = f"spend=${sp:,.2f}, transactions={ntx}, employees={ne}" if sp is not None else "N/A"

    def _fmt_list(items, key):
        if not items:
            return "N/A"
        lines = []
        for i, row in enumerate(items[:10]):
            label = row.get(key) or row.get("name") or row.get("category") or row.get("merchant") or "Unknown"
            spend = row.get("spend", 0.0)
            lines.append(f"{i+1}. {str(label).strip()} — ${spend:,.2f}")
        return "\n".join(lines) if lines else "N/A"

    top_spenders_text  = _fmt_list((glob.get("__summary__:top_spenders") or {}).get("items"),  "name")
    top_categories_text = _fmt_list((glob.get("__summary__:top_categories") or {}).get("items"), "category")
    top_merchants_text  = _fmt_list((glob.get("__summary__:top_merchants") or {}).get("items"),  "merchant")

    trend_items = (glob.get("__summary__:monthly_trend") or {}).get("items")
    if trend_items:
        monthly_text = "\n".join([f"{row.get('month')}: ${row.get('spend', 0.0):,.2f}" for row in trend_items])
    else:
        monthly_text = "N/A"

    return totals_text, top_spenders_text, top_categories_text, top_merchants_text, monthly_text

# ---------- Main entry ----------
def query_expense_gpt(question: str, top_k: int = 10):
    try:
        # 1) Try to get global summaries from Pinecone (no CSV required)
        glob = _fetch_global_summaries()
        g_totals, g_spenders, g_cats, g_merch, g_monthly = _format_global_text(glob) if glob else (None, None, None, None, None)

        # 2) Retrieve subset for examples + governance
        rows, context = _retrieve_context(
            question=question,
            top_k=top_k,
            candidate_k=max(50, top_k * 10),  # query wider
            per_doc_cap=2
        )

        df_subset = pd.DataFrame(rows) if rows else pd.DataFrame()
        subset_analytics = _build_subset_analytics(df_subset) if len(df_subset) else None
        subset_gov = _mutuals_and_rings(df_subset) if len(df_subset) else {"mutual_pairs": [], "cc_mutuals": [], "rings": []}

        def fmt_pairs(pairs):
            return "\n".join([f"{a} ⇄ {b}" for a, b in pairs]) if pairs else "None detected"
        def fmt_cc_mutuals(items):
            return "\n".join([f"{a} ⇄ {b} (Cost Center: {cc})" for a, b, cc in items]) if items else "None detected"
        def fmt_rings(rings):
            return "\n".join([" → ".join(list(c) + [c[0]]) for c in rings]) if rings else "None detected"

        subset_mutuals = fmt_pairs(subset_gov["mutual_pairs"])
        subset_cc_mutuals = fmt_cc_mutuals(subset_gov["cc_mutuals"])
        subset_rings = fmt_rings(subset_gov["rings"])

        # RAG subset texts (for examples/explanations)
        if subset_analytics:
            rag_top_spenders = "\n".join([f"{i+1}. {row['name']} — ${row['spend']:,.2f}" for i, row in enumerate(subset_analytics["top_spenders"][:5])]) or "N/A"
            rag_top_categories = "\n".join([f"{i+1}. {row['category']} — ${row['spend']:,.2f}" for i, row in enumerate(subset_analytics["top_categories"][:5])]) or "N/A"
            rag_top_merchants = "\n".join([f"{i+1}. {row['merchant']} — ${row['spend']:,.2f}" for i, row in enumerate(subset_analytics["top_merchants"][:5])]) or "N/A"
            rag_monthly = "\n".join([f"{row['month']}: ${row['spend']:,.2f}" for row in subset_analytics["monthly_trend"]]) or "N/A"
            rag_totals = f"spend=${subset_analytics['total_spend']:,.2f}, transactions={subset_analytics['n_txns']}, employees={subset_analytics['n_employees']}"
        else:
            rag_top_spenders = rag_top_categories = rag_top_merchants = rag_monthly = "N/A"
            rag_totals = "N/A"

        # 3) Prompt — Prefer GLOBAL summaries if available; otherwise subset analytics.
        prompt = f"""
You are a travel & expense audit assistant.

Use the following data sources in this order:
1) GLOBAL summaries (exact, precomputed and stored inside Pinecone) — if present.
2) SUBSET analytics (computed from the retrieved Pinecone rows) — if global summaries are not present.
Never invent numbers. If neither contains the needed metric, say:
"I don’t know based on the provided data."

Context snippets (from Pinecone; for examples/citations):
{context}

GLOBAL summaries: {'[present]' if g_totals else '[not available]'}
- Totals: {g_totals or 'N/A'}
- Top spenders:
{g_spenders or 'N/A'}

- Top categories:
{g_cats or 'N/A'}

- Top merchants:
{g_merch or 'N/A'}

- Monthly trend:
{g_monthly or 'N/A'}

SUBSET analytics (from retrieved rows only):
- Totals: {rag_totals}
- Top spenders:
{rag_top_spenders}

- Top categories:
{rag_top_categories}

- Top merchants:
{rag_top_merchants}

- Monthly trend:
{rag_monthly}

Governance patterns (from subset):
- Mutual Approval Pairs:
{subset_mutuals}
- High-Risk Mutuals (same cost center):
{subset_cc_mutuals}
- Approval Rings (3+ cycles):
{subset_rings}

User question:
{question}

Rules:
- Prefer GLOBAL summaries when present; otherwise use SUBSET analytics.
- Be concise and structured. If a requested metric isn’t present, state that clearly.
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
