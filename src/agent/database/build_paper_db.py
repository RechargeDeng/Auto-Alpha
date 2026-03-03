import pymupdf  # fitz
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append('/Users/deng/Desktop/alpha-gpt/alpha-gpt/src/agent/prompts')
from build_paper_prompts import PAPER_EXTRACTION_PROMPT

def load_pdf_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


pdf_path = "/Users/deng/Desktop/alpha-gpt/alpha-gpt/1601.00991v3.pdf"
paper_text = load_pdf_text(pdf_path)

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0.0
)

response = llm.invoke(
    PAPER_EXTRACTION_PROMPT.format(paper_text=paper_text)
)

content = response.content.strip()

# 防御性 JSON 解析
json_start = content.find("{")
json_end = content.rfind("}") + 1

paper_structured = json.loads(content[json_start:json_end])

def make_kb_entry(
    kb_type: str,
    paper_id: str,
    name: str,
    description: str,
    semantic_text: str
):
    return {
        "kb_type": kb_type,
        "paper_id": paper_id,
        "name": name,
        "description": description,
        "semantic_text": semantic_text
    }

paper_id = "alpha101"

kb_entries = []

# --- core mechanisms ---
for m in paper_structured.get("core_mechanisms", []):
    kb_entries.append(
        make_kb_entry(
            kb_type="paper_mechanism",
            paper_id=paper_id,
            name=m["mechanism_name"],
            description=m["description"],
            semantic_text=f"{m['mechanism_name']} {m['intuition']} market microstructure"
        )
    )

# --- variables ---
for v in paper_structured.get("variables_and_proxies", []):
    kb_entries.append(
        make_kb_entry(
            kb_type="paper_variable",
            paper_id=paper_id,
            name=v["variable_name"],
            description=v["definition"],
            semantic_text=f"{v['variable_name']} {v['role_in_mechanism']} proxy variable"
        )
    )

# --- empirical findings ---
for r in paper_structured.get("empirical_findings", []):
    kb_entries.append(
        make_kb_entry(
            kb_type="paper_result",
            paper_id=paper_id,
            name="Empirical Finding",
            description=r["finding"],
            semantic_text=f"{r['finding']} {r.get('direction','')} {r.get('time_horizon','')}"
        )
    )

# --- signal hints ---
for h in paper_structured.get("signal_design_hints", []):
    kb_entries.append(
        make_kb_entry(
            kb_type="paper_signal_hint",
            paper_id=paper_id,
            name="Signal Design Hint",
            description=h["hint"],
            semantic_text=f"{h['hint']} related to {h['related_mechanism']}"
        )
    )

paper_jsonl_path = "/Users/deng/Desktop/alpha-gpt/alpha-gpt/rag_fields/rag_papers_alpha101.jsonl"

with open(paper_jsonl_path, "w", encoding="utf-8") as f:
    for entry in kb_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Saved {len(kb_entries)} paper KB entries")

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

texts = []
metadatas = []

for item in kb_entries:
    texts.append(
        f"Name: {item['name']}\n"
        f"Description: {item['description']}\n"
        f"Semantic: {item['semantic_text']}"
    )
    metadatas.append(item)

paper_db = FAISS.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embeddings
)

paper_db.save_local("/Users/deng/Desktop/alpha-gpt/alpha-gpt/src/agent/database/faiss_papers_alpha101")