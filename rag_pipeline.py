# rag_pipeline.py

import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# =========================
# 🔑 API KEY
# =========================
os.environ["GROQ_API_KEY"] = YOUR_API_KEY"  # ← put your key here

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# =========================
# 🧠 EMBEDDING MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 📦 LOAD FAISS INDEX
# =========================
index = faiss.read_index("faiss_index.bin")

# =========================
# 📄 LOAD TEXT DATA
# =========================
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

# =========================
# 🔍 RETRIEVAL FUNCTION
# =========================
def retrieve(query, k=3):
    q_emb = model.encode([query])
    _, indices = index.search(np.array(q_emb), k)
    return [texts[i] for i in indices[0]]

# =========================
# 🤖 LLM FUNCTION
# =========================
def ask_llm(context, query):
    prompt = f"""
You are a professional medical assistant.

Instructions:
- Give answer in bullet points
- Be clear and concise
- Extract side effects clearly
- If not directly mentioned, infer from warnings
- Avoid long paragraphs

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# =========================
# 🚀 MAIN RAG FUNCTION
# =========================
def run(query):
    docs = retrieve(query, k=3)
    context = "\n".join(docs)
    answer = ask_llm(context, query)
    return answer