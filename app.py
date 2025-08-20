# app.py

import os
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone

from query import query_pinecone
from expense_gpt_response import query_expense_gpt, _embed_query  # uses your updated retrieval

app = FastAPI()


# ---------- Request schema ----------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 10


# ---------- Routes ----------
# Pinecone search only (no GPT)
@app.post("/query-expense-docs")
def query_expense_docs(request: QueryRequest):
    results = query_pinecone(request.question, request.top_k)
    return {"results": results}


# GPT + Pinecone context (main endpoint your GPT tool calls)
@app.post("/ask-expense-gpt")
def ask_expense_gpt(request: QueryRequest):
    answer = query_expense_gpt(request.question, request.top_k)
    return {"answer": answer}


# Health check
@app.get("/")
def health_check():
    return {"status": "ok"}


# ---------- Debug helpers ----------
@app.get("/debug-config")
def debug_config():
    """Shows which index/namespace and key presence; also returns Pinecone vector counts."""
    info = {
        "index_name": os.getenv("INDEX_NAME", "auditexpense2"),
        "namespace": os.getenv("PINECONE_NAMESPACE", ""),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "pinecone_key_set": bool(os.getenv("PINECONE_API_KEY")),
    }
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        idx = pc.Index(info["index_name"])
        stats = idx.describe_index_stats()
        info["total_vector_count"] = stats.get("total_vector_count")
        info["namespaces"] = stats.get("namespaces")
    except Exception as e:
        info["pinecone_error"] = str(e)
    return info


@app.post("/debug-query")
def debug_query(request: QueryRequest):
    """Runs a raw Pinecone query (no GPT) to inspect hits/Doc_ID diversity."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("INDEX_NAME", "auditexpense2")
    namespace = os.getenv("PINECONE_NAMESPACE", "")
    idx = pc.Index(index_name)

    qvec = _embed_query(request.question)  # 1024-dim
    res = idx.query(
        vector=qvec,
        top_k=max(20, request.top_k * 3),
        include_metadata=True,
        include_values=False,
        namespace=namespace
    )
    hits = []
    for m in (res.get("matches") or [])[:15]:
        md = m.get("metadata") or {}
        hits.append({
            "id": m.get("id"),
            "score": round(m.get("score", 0.0), 4),
            "Doc_ID": md.get("Doc_ID") or md.get("document_id"),
            "snippet": (md.get("Content") or md.get("content") or md.get("text_chunk") or "")[:140]
        })
    return {"index_name": index_name, "namespace": namespace, "hits": hits}


# ---------- Local run (Render uses its own start command) ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
