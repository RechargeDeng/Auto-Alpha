import pymupdf
import re


doc = pymupdf.open("/Users/deng/Desktop/alpha-gpt/alpha-gpt/1601.00991v3.pdf")
text = "\n".join(page.get_text() for page in doc)

def split_alpha101(text: str):
    """
    Split Alpha101 paper into chunks by Alpha# definitions.
    """
    pattern = re.compile(r"(Alpha#\d+[\s\S]*?)(?=Alpha#\d+|$)")
    matches = pattern.findall(text)

    chunks = []
    for m in matches:
        header = re.search(r"Alpha#(\d+)", m)
        alpha_id = int(header.group(1)) if header else None

        chunks.append({
            "alpha_id": alpha_id,
            "content": m.strip()
        })
    return chunks


alpha_chunks = split_alpha101(text)
print(f"Extracted {len(alpha_chunks)} alpha chunks")
alpha_chunks = alpha_chunks[2:]

def build_alpha101_documents(alpha_chunks):
    docs = []

    for c in alpha_chunks:
        alpha_id = c["alpha_id"]
        content = c["content"]

        docs.append({
            "text": content,
            "metadata": {
                "kb_type": "paper",
                "paper": "Alpha101",
                "alpha_id": alpha_id,
                "source": "Kakushadze 2015",
                "semantic_hint": f"alpha {alpha_id} quantitative trading signal formula"
            }
        })
    return docs


paper_docs = build_alpha101_documents(alpha_chunks)

def clean_text(text: str) -> str:
    lines = text.splitlines()
    lines = [l.strip() for l in lines if len(l.strip()) > 0]
    return "\n".join(lines)


for d in paper_docs:
    d["text"] = clean_text(d["text"])

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

from langchain.vectorstores import FAISS

texts = [d["text"] for d in paper_docs]
metadatas = [d["metadata"] for d in paper_docs]

paper_db = FAISS.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embeddings
)

paper_db.save_local("/Users/deng/Desktop/alpha-gpt/alpha-gpt/src/agent/database/faiss_alpha101_paper_db")
