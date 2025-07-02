from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from pinecone import Pinecone

# ✅ Hardcoded API keys (your real values)
openai.api_key = "OPENAI_API_KEY"
pc = Pinecone(api_key="PINECONE_API_KEY")

index = pc.Index("auditexpense2")  # ✅ Your active Pinecone index

# ✅ Flask app
app = Flask(__name__)
CORS(app)

@app.route("/query", methods=["POST"])
def query():
    try:
        user_prompt = request.json.get("query")
        if not user_prompt:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        # 🔹 Embed user query
        response = openai.embeddings.create(
            input=user_prompt,
            model="text-embedding-3-large",
            dimensions=1024
        )
        embedding = response.data[0].embedding

        # 🔹 Query Pinecone
        pinecone_results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )

        return jsonify({
            "query": user_prompt,
            "results": pinecone_results["matches"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
