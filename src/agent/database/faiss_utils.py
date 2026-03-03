#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_faiss_db(path: str) -> FAISS:
    """
    Load a FAISS vector store saved with LangChain.

    Args:
        path: directory containing index.faiss and index.pkl

    Returns:
        FAISS vector store instance
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )