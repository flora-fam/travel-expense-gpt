def _is_analytics_question(question: str) -> bool:
    """Detect if the question needs a larger sample for better analytics."""
    analytics_keywords = [
        "top spenders", "top categories", "top merchants", "monthly trend",
        "biggest spenders", "highest spenders", "most expensive", "spending by",
        "breakdown by", "summary", "analytics", "who spent", "which employee"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in analytics_keywords)

# Replace your existing query_expense_gpt function with this enhanced version:
def query_expense_gpt(question: str, top_k: int = 10):
    try:
        # FIRST: Check if this is a count question
        if _is_count_question(question):
            total_count = _get_total_record_count()
            if total_count is not None:
                return f"The dataset contains a total of **{total_count:,} expense records**."
            else:
                return "I couldn't retrieve the total record count from the database."
        
        # SECOND: Check if this needs a larger sample for analytics
        if _is_analytics_question(question):
            # Use much larger sample for better analytics
            top_k = max(100, top_k * 10)  # Get at least 100 records for analytics
            candidate_k = max(1000, top_k * 10)  # Search through more candidates
        
        # 1) GLOBAL summaries (if present)
        glob = _fetch_global_summaries()
        g_totals, g_spenders, g_cats, g_merch, g_monthly = _format_global_text(glob) if glob else (None, None, None, None, None)

        # 2) Retrieve subset for examples + governance
        rows, context = _retrieve_context(
            question=question,
            top_k=top_k,
            candidate_k=max(50, candidate_k if _is_analytics_question(question) else top_k * 10),
            per_doc_cap=3 if _is_analytics_question(question) else 2  # Allow more per document for analytics
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

        # 3) Enhanced prompt for better analytics handling
        total_count = _get_total_record_count()
        count_info = f"Total records in database: {total_count:,}" if total_count else "Total records: unknown"
        
        sample_size_note = ""
        if _is_analytics_question(question):
            sample_size_note = f"\n**ANALYTICS MODE**: Retrieved {len(df_subset)} records from {total_count:,} total for representative analysis."
        
        prompt = f"""
You are a travel & expense audit assistant.

IMPORTANT: {count_info}
(The subset below is {len(df_subset)} records retrieved for context/examples){sample_size_note}

Use the following data sources in this order:
1) GLOBAL summaries (exact, precomputed and stored inside Pinecone) — if present.
2) SUBSET analytics (computed from the retrieved Pinecone rows) — if global summaries are not present.
Never invent numbers. If neither contains the needed metric, say:
"I don't know based on the provided data."

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
- For analytics questions, acknowledge the sample size but provide the best analysis possible from the retrieved data.
- Be concise and structured. If a requested metric isn't present, state that clearly.
- Remember: Total database has {total_count:,} records if asked about dataset size.
- If using subset data for analytics, mention this is from a sample of {len(df_subset)} records.
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
