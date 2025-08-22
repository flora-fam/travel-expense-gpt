# app.py

import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from pinecone import Pinecone

from query import query_pinecone
from expense_gpt_response import (
    query_expense_gpt,
    _embed_query,       # used by /debug-query
    _retrieve_context,  # used by /debug/retrieve
)

app = FastAPI()

# ---------- Request schema ----------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 10


# ---------- Health ----------
@app.get("/")
def health_check():
    return {"status": "ok"}


# ---------- Pinecone search only (no GPT) ----------
@app.post("/query-expense-docs")
def query_expense_docs(request: QueryRequest):
    results = query_pinecone(request.question, request.top_k)
    return {"results": results}


# ---------- GPT + Pinecone context (main endpoint) ----------
@app.post("/ask-expense-gpt")
def ask_expense_gpt(request: QueryRequest):
    answer = query_expense_gpt(request.question, request.top_k)
    return {"answer": answer}


# ---------- DEBUG: environment + Pinecone stats ----------
@app.get("/debug/pinecone")
def debug_pinecone():
    index_name = os.getenv("INDEX_NAME", "auditexpense2")
    namespace = os.getenv("PINECONE_NAMESPACE", "")
    api_key = os.getenv("PINECONE_API_KEY", "")
    key_ok = bool(api_key and api_key.startswith("pcsk_"))

    try:
        pc = Pinecone(api_key=api_key)
        idx = pc.Index(index_name)
        stats = idx.describe_index_stats()
        return {
            "index_name": index_name,
            "namespace": namespace,
            "pinecone_key_ok": key_ok,
            "total_vector_count": stats.get("total_vector_count"),
            "dimension": stats.get("dimension"),
            "namespaces": list((stats.get("namespaces") or {}).keys()),
        }
    except Exception as e:
        return {
            "index_name": index_name,
            "namespace": namespace,
            "pinecone_key_ok": key_ok,
            "error": str(e),
        }


# ---------- DEBUG: prove retrieval depth (no GPT involvement) ----------
@app.get("/debug/retrieve")
def debug_retrieve(
    q: str = Query(..., description="test query"),
    top_k: int = 12,
    candidate_k: int = 200
):
    rows, _ = _retrieve_context(q, top_k=top_k, candidate_k=candidate_k)
    doc_ids = [r.get("__doc_id__", "unknown") for r in rows]
    return {
        "returned_chunks": len(rows),
        "unique_doc_ids": len(set(doc_ids)),
        "doc_ids_sample": doc_ids[:10],
    }


# ---------- DEBUG: raw Pinecone query (score/doc diversity check) ----------
@app.post("/debug-query")
def debug_query(request: QueryRequest):
    """
    Runs a raw Pinecone query (no GPT) to inspect hits/Doc_ID diversity.
    Helpful to confirm retrieval is working and varied.
    """
    try:
        pk = os.getenv("PINECONE_API_KEY")
        iname = os.getenv("INDEX_NAME", "auditexpense2")
        namespace = os.getenv("PINECONE_NAMESPACE", "")
        if not pk or not iname:
            return {
                "error": "Missing PINECONE_API_KEY or INDEX_NAME",
                "index_name": iname,
                "namespace": namespace,
            }

        pc = Pinecone(api_key=pk)
        idx = pc.Index(iname)

        qvec = _embed_query(request.question)  # 1024-dim to match your index
        res = idx.query(
            vector=qvec,
            top_k=max(20, request.top_k * 3),
            include_metadata=True,
            include_values=False,  # we don't rely on raw values
            namespace=namespace,
        )

        # Handle object/dict response shapes
        try:
            matches = res.matches  # type: ignore[attr-defined]
        except Exception:
            matches = (res.get("matches") or [])  # type: ignore[index]

        hits = []
        for m in matches[: max(15, request.top_k)]:
            try:
                # object-style from Pinecone client
                md = m.metadata or {}
                mid = getattr(m, "id", None)
                score = float(getattr(m, "score", 0.0))
            except Exception:
                # dict-style fallback
                md = m.get("metadata") or {}
                mid = m.get("id")
                score = float(m.get("score", 0.0))

            snippet = (md.get("Content") or md.get("content") or md.get("text_chunk") or "")
            docid = md.get("Doc_ID") or md.get("ID") or md.get("document_id")
            hits.append({
                "id": mid,
                "score": round(score, 4),
                "Doc_ID": docid,
                "snippet": str(snippet)[:140],
            })

        return {"index_name": iname, "namespace": namespace, "hits": hits}

    except Exception as e:
        return {"error": str(e)}


# ---------- Local run (Render uses its own start command) ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
