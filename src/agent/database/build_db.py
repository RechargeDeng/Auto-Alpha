import json
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

def load_jsonl(path):
    records = []
    bad_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad_lines.append((idx, line, str(e)))

    if bad_lines:
        print(f"[WARN] {len(bad_lines)} invalid json lines in {path}")
        for idx, line, err in bad_lines[:5]:
            print(f"Line {idx}: {err}")
            print(line[:200])
            print("-" * 40)

    return records


def build_embedding_text(item):
    """
    Construct text used for embedding.
    """
    parts = [
        f"Name: {item.get('name', '')}",
        f"Description: {item.get('description', '')}",
        f"Semantic: {item.get('semantic_text', '')}",
    ]
    return "\n".join(p for p in parts if p.strip())

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

def build_faiss_from_jsonl(jsonl_path, save_path):
    records = load_jsonl(jsonl_path)

    texts = []
    metadatas = []

    for item in records:
        texts.append(build_embedding_text(item))
        metadatas.append(item)  # 整条 metadata 保留，方便回溯

    db = FAISS.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings
    )

    db.save_local(save_path)
    print(f"Saved FAISS DB to {save_path}, size={len(texts)}")

    return db

build_faiss_from_jsonl(
    jsonl_path="/Users/deng/Desktop/alpha-gpt/alpha-gpt/rag_fields/layer1_modules.jsonl",
    save_path="/Users/deng/Desktop/alpha-gpt/alpha-gpt/src/agent/database/faiss_layer1_modules"
)

build_faiss_from_jsonl(
    jsonl_path="/Users/deng/Desktop/alpha-gpt/alpha-gpt/rag_fields/layer2_fields.jsonl",
    save_path="/Users/deng/Desktop/alpha-gpt/alpha-gpt/src/agent/database/faiss_layer2_fields"
)

build_faiss_from_jsonl(
    jsonl_path="/Users/deng/Desktop/alpha-gpt/alpha-gpt/rag_fields/op.jsonl",
    save_path="/Users/deng/Desktop/alpha-gpt/alpha-gpt/src/agent/database/faiss_operators"
)