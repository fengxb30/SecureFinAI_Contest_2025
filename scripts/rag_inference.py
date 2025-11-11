# scripts/rag_inference.py
import faiss, json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 加载索引与元数据
index = faiss.read_index("data/processed/embeddings.faiss")
metadata = json.load(open("data/processed/metadata.json"))
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 加载模型
model = AutoModelForCausalLM.from_pretrained("models/lora_finetuned", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def retrieve(query, k=3):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb, dtype=np.float32), k)
    return [metadata[i]['text'] for i in I[0]]

def answer_query(query):
    docs = retrieve(query)
    context = "\n".join(docs)
    prompt = f"""You are FinGPT Compliance Agent.
Answer the question using only the SEC filings below:
Context:\n{context}\n\nQuestion: {query}\nAnswer:"""

    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=300)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

query = "What factors led to the decrease in net income in 2023?"
answer_query(query)

