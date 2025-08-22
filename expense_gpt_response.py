# expense_gpt_response.py
# Hybrid approach: Vector search + CSV analysis for complete accuracy
# Env vars:
#   OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME (e.g., auditexpense2), PINECONE_NAMESPACE (optional)

import os
import re
import pandas as pd
import numpy as np
import networkx as nx
from openai import OpenAI
from pinecone import Pinecone
import io

# ---------- Config ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "auditexpense2")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

client = OpenAI(api_key=OPENAI_KEY)
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)

# Global variable to cache CSV data
_cached_csv_data = None

# ---------- CSV Analysis Functions ----------
def _load_csv_data():
    """Load and cache the CSV data for complete analysis."""
    global _cached_csv_data
    
    if _cached_csv_data is not None:
        return _cached_csv_data
    
    try:
        # Try to read the CSV file from current directory
        # In your deployment, make sure TNE Data_Updated_Flora.csv is in the same directory
        df = pd.read_csv('TNE Data_Updated_Flora.csv', encoding='cp1252')
        
        # Clean and standardize column names
        df.columns = df.columns.str.strip()
        
        # Convert amount columns to numeric
        amount_cols = ['Expense Amount (rpt)', 'Approved Amount (rpt)']
        for col in amount_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert ID columns to strings for consistent comparison
        id_cols = ['Employee ID', 'Default Approver ID']
        for col in id_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Convert date columns
        date_cols = ['Transaction Date', 'Created Date', 'Paid Date', 'Report Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        _cached_csv_data = df
        print(f"Loaded CSV data: {len(df)} records")
        return df
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def _is_csv_analysis_question(question: str) -> bool:
    """Determine if question requires complete CSV analysis for accuracy."""
    csv_keywords = [
        # Governance violations requiring 100% accuracy
        "mutual approval", "approved each other", "self approval", "self-approval",
        "threshold dodging", "circular approval", "approval ring",
        
        # Financial totals requiring exact counts
        "total expense", "total amount", "total spending", "grand total",
        "how many records", "count of", "total count", "dataset size",
        "total value", "sum of", "aggregate",
        
        # Compliance requiring complete audit trail
        "missing receipts", "receipt compliance", "policy violation",
        "approval delays over", "processing time", "unapproved expenses",
        
        # Data quality requiring full scan
        "duplicate", "missing data", "data quality", "inconsistent",
        "zero amount", "negative amount", "data integrity",
        
        # Exact vendor/employee analysis
        "all vendors", "complete list", "every employee", "full breakdown",
        "exact number", "precise count", "audit trail",
        
        # Critical audit queries
        "governance violation", "control breakdown", "fraud detection",
        "compliance issue", "audit finding", "risk assessment"
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in csv_keywords)

def _analyze_mutual_approvals(df):
    """Find all mutual approval relationships in the complete dataset."""
    if df is None or len(df) == 0:
        return {"mutual_pairs": [], "total_cases": 0, "analysis": "No data available"}
    
    # Create approval relationship mapping
    approval_pairs = {}
    
    for _, row in df.iterrows():
        emp_id = str(row.get('Employee ID', '')).strip()
        app_id = str(row.get('Default Approver ID', '')).strip()
        
        if emp_id and app_id and emp_id != app_id:
            key = f"{emp_id}-{app_id}"
            if key not in approval_pairs:
                approval_pairs[key] = []
            
            approval_pairs[key].append({
                'employee': row.get('Employee', ''),
                'approver': row.get('Default Approver', ''),
                'amount': row.get('Expense Amount (rpt)', 0),
                'date': row.get('Transaction Date', ''),
                'report_id': row.get('Report ID', '')
            })
    
    # Find mutual approvals
    mutual_cases = []
    processed_pairs = set()
    
    for key, transactions in approval_pairs.items():
        emp_id, app_id = key.split('-')
        reverse_key = f"{app_id}-{emp_id}"
        pair_key = tuple(sorted([emp_id, app_id]))
        
        if reverse_key in approval_pairs and pair_key not in processed_pairs:
            processed_pairs.add(pair_key)
            
            forward_txns = transactions
            reverse_txns = approval_pairs[reverse_key]
            
            total_forward_amount = sum(t['amount'] for t in forward_txns if t['amount'])
            total_reverse_amount = sum(t['amount'] for t in reverse_txns if t['amount'])
            
            mutual_cases.append({
                'employee_1_id': emp_id,
                'employee_2_id': app_id,
                'employee_1_name': forward_txns[0]['employee'] if forward_txns else '',
                'employee_2_name': reverse_txns[0]['employee'] if reverse_txns else '',
                'forward_transactions': len(forward_txns),
                'reverse_transactions': len(reverse_txns),
                'forward_amount': total_forward_amount,
                'reverse_amount': total_reverse_amount,
                'total_amount': total_forward_amount + total_reverse_amount
            })
    
    return {
        "mutual_pairs": mutual_cases,
        "total_cases": len(mutual_cases),
        "analysis": f"Found {len(mutual_cases)} mutual approval relationships in complete dataset"
    }

def _analyze_complete_dataset(question: str, df):
    """Perform complete dataset analysis for critical audit questions."""
    if df is None:
        return "Error: Unable to load complete dataset for analysis"
    
    question_lower = question.lower()
    
    # Handle specific analysis types
    if "mutual approval" in question_lower or "approved each other" in question_lower:
        result = _analyze_mutual_approvals(df)
        
        if result["total_cases"] == 0:
            return "**No mutual approvals detected** in the complete dataset of {:,} records.".format(len(df))
        
        response = f"**Found {result['total_cases']} mutual approval relationships** in {:,} total records:\n\n"
        
        for i, case in enumerate(result["mutual_pairs"], 1):
            response += f"{i}. **{case['employee_1_name']} (ID: {case['employee_1_id']})** ⇄ **{case['employee_2_name']} (ID: {case['employee_2_id']})**\n"
            response += f"   - {case['employee_1_name']} → {case['employee_2_name']}: {case['forward_transactions']} transactions, ${case['forward_amount']:,.2f}\n"
            response += f"   - {case['employee_2_name']} → {case['employee_1_name']}: {case['reverse_transactions']} transactions, ${case['reverse_amount']:,.2f}\n"
            response += f"   - **Total mutual approval amount: ${case['total_amount']:,.2f}**\n\n"
        
        response += "**Audit Risk:** Mutual approvals represent potential conflicts of interest and control violations requiring immediate review."
        return response
    
    elif "total" in question_lower and ("record" in question_lower or "count" in question_lower):
        return f"The complete dataset contains **{len(df):,} expense records** with a total expense amount of **${df['Expense Amount (rpt)'].sum():,.2f}**."
    
    elif "self approval" in question_lower or "self-approval" in question_lower:
        self_approvals = df[df['Employee ID'] == df['Default Approver ID']]
        if len(self_approvals) == 0:
            return f"**No self-approvals detected** in the complete dataset of {len(df):,} records."
        
        total_amount = self_approvals['Expense Amount (rpt)'].sum()
        return f"**Found {len(self_approvals)} self-approval violations** totaling **${total_amount:,.2f}** across {len(df):,} total records. This represents a critical control breakdown requiring immediate investigation."
    
    elif "threshold dodging" in question_lower or ("4800" in question_lower and "5000" in question_lower):
        threshold_expenses = df[(df['Expense Amount (rpt)'] >= 4800) & (df['Expense Amount (rpt)'] <= 5000)]
        
        if len(threshold_expenses) == 0:
            return "**No threshold dodging detected** - no expenses found between $4,800 and $5,000 in the complete dataset."
        
        by_type = threshold_expenses.groupby('Expense Type').agg({
            'Expense Amount (rpt)': ['count', 'sum']
        }).round(2)
        
        response = f"**Found {len(threshold_expenses)} potential threshold dodging cases** between $4,800-$5,000:\n\n"
        
        for expense_type, data in by_type.iterrows():
            count = data[('Expense Amount (rpt)', 'count')]
            total = data[('Expense Amount (rpt)', 'sum')]
            response += f"- **{expense_type}**: {count} transactions, ${total:,.2f}\n"
        
        response += f"\n**Total suspicious amount: ${threshold_expenses['Expense Amount (rpt)'].sum():,.2f}**"
        response += "\n**Audit Risk:** Pattern suggests potential policy circumvention to avoid higher approval thresholds."
        return response
    
    elif "top" in question_lower and "spender" in question_lower:
        # Extract number from question (default to 10)
        import re
        numbers = re.findall(r'\d+', question)
        top_n = int(numbers[0]) if numbers else 10
        
        top_spenders = df.groupby(['Employee', 'Employee ID'])['Expense Amount (rpt)'].sum().sort_values(ascending=False).head(top_n)
        
        response = f"**Top {top_n} Spenders** (complete dataset analysis):\n\n"
        for i, ((name, emp_id), amount) in enumerate(top_spenders.items(), 1):
            response += f"{i}. **{name}** (ID: {emp_id}) — ${amount:,.2f}\n"
        
        response += f"\n**Analysis based on complete dataset of {len(df):,} records**"
        return response
    
    # Add more specific analysis types as needed...
    
    # Generic complete dataset summary
    total_amount = df['Expense Amount (rpt)'].sum()
    unique_employees = df['Employee ID'].nunique()
    unique_vendors = df['Vendor'].nunique() if 'Vendor' in df.columns else 0
    
    return f"""**Complete Dataset Analysis**:
- **Total Records**: {len(df):,}
- **Total Expense Amount**: ${total_amount:,.2f}
- **Unique Employees**: {unique_employees:,}
- **Unique Vendors**: {unique_vendors:,}
- **Date Range**: {df['Transaction Date'].min()} to {df['Transaction Date'].max()}

For specific audit analysis, please ask about mutual approvals, self-approvals, threshold dodging, or other compliance patterns."""

# ---------- Embedding ----------
def _embed_query(text: str) -> list:
    """Embed with same dimension as your index (1024)."""
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=1024
    ).data[0].embedding
    return emb

# ---------- Total record count functions ----------
def _get_total_record_count():
    """Get the actual total number of records in Pinecone index."""
    try:
        stats = index.describe_index_stats()
        if hasattr(stats, 'total_vector_count'):
            return stats.total_vector_count
        elif isinstance(stats, dict):
            return stats.get('total_vector_count', 0)
        else:
            # Fallback - try to get from namespaces
            namespaces = getattr(stats, 'namespaces', {}) or stats.get('namespaces', {})
            if NAMESPACE and NAMESPACE in namespaces:
                ns_stats = namespaces[NAMESPACE]
                return getattr(ns_stats, 'vector_count', 0) or ns_stats.get('vector_count', 0)
            elif '' in namespaces:  # default namespace
                ns_stats = namespaces['']
                return getattr(ns_stats, 'vector_count', 0) or ns_stats.get('vector_count', 0)
            else:
                return sum(getattr(ns, 'vector_count', 0) or ns.get('vector_count', 0) 
                          for ns in namespaces.values())
    except Exception as e:
        print(f"Error getting total count: {e}")
        return None

def _is_count_question(question: str) -> bool:
    """Detect if the question is asking for total record count."""
    count_keywords = [
        "how many", "total records", "dataset size", "count", "total", 
        "number of records", "size of dataset", "how big", "records in",
        "entries in", "transactions in", "expenses in"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in count_keywords)

# ---------- Enhanced analytics functions ----------
def _get_large_representative_sample(question: str, sample_size: int = 500):
    """
    Get a large, diverse sample for analytics by using multiple query strategies.
    This gives us a much more representative dataset without pre-computed summaries.
    """
    try:
        # Strategy 1: Use the original question
        primary_sample, _ = _retrieve_context(
            question=question, 
            top_k=sample_size // 3,
            candidate_k=sample_size * 2,
            per_doc_cap=5
        )
        
        # Strategy 2: Query for high-value transactions
        high_value_sample, _ = _retrieve_context(
            question="high value expensive transactions large amounts", 
            top_k=sample_size // 3,
            candidate_k=sample_size * 2,
            per_doc_cap=5
        )
        
        # Strategy 3: Query for diverse employees/categories
        diverse_sample, _ = _retrieve_context(
            question="employee spending categories travel meals expenses", 
            top_k=sample_size // 3,
            candidate_k=sample_size * 2,
            per_doc_cap=5
        )
        
        # Combine and deduplicate by Doc_ID
        all_rows = primary_sample + high_value_sample + diverse_sample
        seen_ids = set()
        unique_rows = []
        
        for row in all_rows:
            doc_id = row.get("__doc_id__", "unknown")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_rows.append(row)
                
        # Limit to target sample size
        return unique_rows[:sample_size]
        
    except Exception as e:
        print(f"Error getting large sample: {e}")
        # Fallback to single query
        rows, _ = _retrieve_context(question, top_k=sample_size, candidate_k=sample_size * 3, per_doc_cap=10)
        return rows

def _is_analytics_question(question: str) -> bool:
    """Detect if the question needs a larger sample for better analytics."""
    analytics_keywords = [
        "top spenders", "top categories", "top merchants", "monthly trend",
        "biggest spenders", "highest spenders", "most expensive", "spending by",
        "breakdown by", "summary", "analytics", "who spent", "which employee",
        "top 10", "top 5", "ranking", "leaderboard", "global summaries",
        "show me summaries", "overview", "spending patterns", "expense analysis"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in analytics_keywords)

# ---------- Retrieval + context (RAG subset) ----------
def _retrieve_context(question: str, top_k: int = 10, candidate_k: int = 200,
                      per_doc_cap: int = 2):
    """
    Query Pinecone wide, rank by score, cap per Doc_ID (using metadata or match.id),
    then fill to top_k.
    """
    qvec = _embed_query(question)

    res = index.query(
        vector=qvec,
        top_k=candidate_k,
        include_metadata=True,
        include_values=False,   # we don't rely on raw vectors
        namespace=NAMESPACE
    )

    # Handle object-like or dict-like response
    try:
        matches = res.matches  # type: ignore[attr-defined]
    except Exception:
        matches = (res.get("matches") or [])  # type: ignore[index]

    if not matches:
        return [], ""

    items = []
    for m in matches:
        # Extract metadata / score / match id robustly
        try:
            md = m.metadata or {}              # object style
            score = float(m.score)
            match_id = getattr(m, "id", None)
        except Exception:
            md = m.get("metadata") or {}       # dict style
            score = float(m.get("score", 0.0))
            match_id = m.get("id")

        # Content to show the LLM (fallback if no 'content'-like field)
        content = (
            md.get("Content")
            or md.get("content")
            or md.get("text_chunk")
            or ""
        )
        content = str(content).strip()
        if not content:
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

        # CRITICAL FIX: Choose a proper doc id, falling back to the match id
        doc_id = (
            md.get("Doc_ID")
            or md.get("ID")
            or md.get("document_id")
            or md.get("DocID")
            or match_id               # <--- fallback to Pinecone match id
            or "unknown"
        )

        row = dict(md)
        row["__content__"] = content
        row["__doc_id__"] = str(doc_id)

        items.append((score, row, content, str(doc_id)))

    if not items:
        return [], ""

    # Sort by score (desc)
    items.sort(key=lambda x: x[0], reverse=True)

    # First pass: enforce per-doc cap
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

    # Second pass: if still short, ignore per-doc cap
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
    for a, b in sorted(mutual_pairs):
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
    """Fetch summary docs by fixed IDs from Pinecone (namespace aware)."""
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
    """Human-readable blocks from fetched summaries."""
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

    top_spenders_text   = _fmt_list((glob.get("__summary__:top_spenders") or {}).get("items"),  "name")
    top_categories_text = _fmt_list((glob.get("__summary__:top_categories") or {}).get("items"), "category")
    top_merchants_text  = _fmt_list((glob.get("__summary__:top_merchants") or {}).get("items"),  "merchant")

    trend_items = (glob.get("__summary__:monthly_trend") or {}).get("items")
    monthly_text = "N/A"
    if trend_items:
        monthly_text = "\n".join([f"{row.get('month')}: ${row.get('spend', 0.0):,.2f}" for row in trend_items])

    return totals_text, top_spenders_text, top_categories_text, top_merchants_text, monthly_text

# ---------- Main entry ----------
def query_expense_gpt(question: str, top_k: int = 10):
    try:
        # FIRST: Check if this requires complete CSV analysis for 100% accuracy
        if _is_csv_analysis_question(question):
            print(f"Using CSV analysis for critical audit question: {question}")
            csv_data = _load_csv_data()
            return _analyze_complete_dataset(question, csv_data)
        
        # SECOND: Check if this is a basic count question
        if _is_count_question(question):
            total_count = _get_total_record_count()
            if total_count is not None:
                return f"The dataset contains a total of **{total_count:,} expense records**."
            else:
                return "I couldn't retrieve the total record count from the database."
        
        # THIRD: Handle analytics questions with large representative sampling
        if _is_analytics_question(question):
            # Get a large, diverse sample for robust analytics
            rows = _get_large_representative_sample(question, sample_size=800)
            context = "Large representative sample retrieved for analytics"
            
        else:
            # Normal retrieval for specific questions
            rows, context = _retrieve_context(
                question=question,
                top_k=top_k,
                candidate_k=max(50, top_k * 10),
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
            rag_top_spenders   = "\n".join([f"{i+1}. {row['name']} — ${row['spend']:,.2f}" for i, row in enumerate(subset_analytics["top_spenders"][:10])]) or "N/A"
            rag_top_categories = "\n".join([f"{i+1}. {row['category']} — ${row['spend']:,.2f}" for i, row in enumerate(subset_analytics["top_categories"][:10])]) or "N/A"
            rag_top_merchants  = "\n".join([f"{i+1}. {row['merchant']} — ${row['spend']:,.2f}" for i, row in enumerate(subset_analytics["top_merchants"][:10])]) or "N/A"
            rag_monthly        = "\n".join([f"{row['month']}: ${row['spend']:,.2f}" for row in subset_analytics["monthly_trend"]]) or "N/A"
            rag_totals         = f"spend=${subset_analytics['total_spend']:,.2f}, transactions={subset_analytics['n_txns']}, employees={subset_analytics['n_employees']}"
        else:
            rag_top_spenders = rag_top_categories = rag_top_merchants = rag_monthly = "N/A"
            rag_totals = "N/A"

        # Enhanced prompt
        total_count = _get_total_record_count()
        count_info = f"Total records in database: {total_count:,}" if total_count else "Total records: unknown"
        
        sample_info = ""
        if _is_analytics_question(question):
            coverage_pct = (len(df_subset) / total_count * 100) if total_count else 0
            sample_info = f"\n**ANALYTICS**: Based on {len(df_subset)} records ({coverage_pct:.1f}% representative sample) from {total_count:,} total."
        
        prompt = f"""
You are a travel & expense audit assistant.

IMPORTANT: {count_info}{sample_info}

Context snippets (from Pinecone):
{context if not _is_analytics_question(question) else "Representative sample for analytics"}

SUBSET analytics (from retrieved records):
- Totals: {rag_totals}
- Top spenders:
{rag_top_spenders}

- Top categories:
{rag_top_categories}

- Top merchants:
{rag_top_merchants}

- Monthly trend:
{rag_monthly}

Governance patterns:
- Mutual Approval Pairs:
{subset_mutuals}
- High-Risk Mutuals (same cost center):
{subset_cc_mutuals}
- Approval Rings (3+ cycles):
{subset_rings}

User question:
{question}

Rules:
- For analytics questions, present results as robust insights from the representative sample
- Be confident in the analysis while noting it's based on a substantial sample
- Focus on actionable insights rather than disclaimers about completeness
- Remember: Total database has {total_count:,} records
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
