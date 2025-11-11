# scripts/build_index.py
from sentence_transformers import SentenceTransformer
import json, faiss, numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
data, texts = [], []

with open("data/processed/text_chunks.jsonl") as f:
    for line in f:
        record = json.loads(line)
        texts.append(record['text'])
        data.append(record)

embeddings = model.encode(texts, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "data/processed/embeddings.faiss")
json.dump(data, open("data/processed/metadata.json", "w"))

print("✅ Embedding index built.")
