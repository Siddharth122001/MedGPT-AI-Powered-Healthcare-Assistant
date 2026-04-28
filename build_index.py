import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load your dataset file
file_path = "C:/Users/SiddharthSingh/Downloads/drug-label-0001-of-0013.json"

with open(file_path, encoding="utf-8") as f:
    data = json.load(f)

texts = []

# Extract useful info
for item in data["results"][:5000]:  # limit for speed
    drug = item.get("openfda", {}).get("brand_name", ["Unknown"])[0]
    warnings = " ".join(item.get("warnings", []))
    dosage = " ".join(item.get("dosage_and_administration", []))

    combined = f"Drug: {drug} | Warnings: {warnings} | Dosage: {dosage}"
    texts.append(combined)

print("Total texts:", len(texts))

# Create embeddings
embeddings = model.encode(texts)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index
faiss.write_index(index, "faiss_index.bin")

# Save texts
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ Index built and saved!")