import os, requests, zipfile, io

def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading {url} ...")
        r = requests.get(url)
        with open(dest, 'wb') as f:
            f.write(r.content)

# FinQA 数据
download_file("https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip",
              "data/raw/finqa.zip")

# TAT-QA 数据
download_file("https://github.com/NExTplusplus/TAT-QA/archive/refs/heads/main.zip",
              "data/raw/tatqa.zip")

# FiQA 数据
download_file("https://raw.githubusercontent.com/maziyarpanahi/FiQA/master/FiQA_train.csv",
              "data/raw/fiqa_train.csv")

print("✅ 数据下载完成，请手动从 EDGAR 下载几份 10-K 示例放入 data/raw/sec_filings/")
