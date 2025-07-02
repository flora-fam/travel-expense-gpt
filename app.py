from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from pinecone import Pinecone

# ✅ Environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# ✅ Pinecone v7 object-based init
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("auditexpense2")

# ✅ Flask app setup
app = Flask(__name__)
CORS(app)

@app.route("/query", methods=["POST"])
def query():
    try:
        user_prompt = request.json.get("query")
        if not user_prompt:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        # 🔹 Embed the query
        response = openai.embeddings.create(
            input=user_prompt,
            model="text-embedding-3-large",
            dimensions=1024
        )
        embedding = response.data[0].embedding

        # 🔹 Query Pinecone
        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )

        return jsonify({
            "query": user_prompt,
            "results": results["matches"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
