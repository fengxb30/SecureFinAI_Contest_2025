# scripts/preprocess_data.py
import os, json, re
from bs4 import BeautifulSoup

RAW_DIR = "data/raw/sec_filings/"
OUT_PATH = "data/processed/text_chunks.jsonl"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'&#\d+;', '', text)
    return text.strip()

def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    text = ' '.join([t.get_text(separator=' ') for t in soup.find_all(['p', 'td'])])
    return clean_text(text)

def chunk_text(text, max_len=512):
    words = text.split()
    for i in range(0, len(words), max_len):
        yield ' '.join(words[i:i+max_len])

with open(OUT_PATH, 'w', encoding='utf-8') as fout:
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith(('.htm', '.html', '.txt')): continue
        text = extract_text_from_html(os.path.join(RAW_DIR, fname))
        for i, chunk in enumerate(chunk_text(text)):
            record = {
                "doc_id": fname,
                "chunk_id": i,
                "text": chunk
            }
            fout.write(json.dumps(record) + "\n")

print(f"✅ Processed SEC filings saved to {OUT_PATH}")
