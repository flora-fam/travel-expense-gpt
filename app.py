# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from query import query_pinecone
from expense_gpt_response import query_expense_gpt  # GPT integration

app = FastAPI()

# ✅ Input schema
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

# ✅ Route 1: Pinecone search only
@app.post("/query-expense-docs")
def query_expense_docs(request: QueryRequest):
    results = query_pinecone(request.question, request.top_k)
    return {"results": results}

# ✅ Route 2: GPT + Pinecone context
@app.post("/ask-expense-gpt")
def ask_expense_gpt(request: QueryRequest):
    answer = query_expense_gpt(request.question, request.top_k)
    return {"answer": answer}

# ✅ Route 3: Root health check for Render
@app.get("/")
def health_check():
    return {"status": "ok"}

# ✅ Ensure Render detects correct port
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
