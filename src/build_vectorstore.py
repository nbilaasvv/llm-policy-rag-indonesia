"""FAISS vector store construction and loading."""

from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTORSTORE_DIR = "vectorstore_reformasi_2045"


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL):
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(documents, output_dir: str = DEFAULT_VECTORSTORE_DIR,
                      embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    if not documents:
        raise ValueError("Tidak ada dokumen untuk dimasukkan ke vectorstore.")
    embeddings = get_embeddings(embedding_model)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(output_dir)
    print(f"✔ Vectorstore disimpan di: {output_dir}")
    return vectorstore


def load_vectorstore(output_dir: str = DEFAULT_VECTORSTORE_DIR,
                     embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    if not Path(output_dir).exists():
        raise FileNotFoundError(f"Vectorstore tidak ditemukan: {output_dir}")
    embeddings = get_embeddings(embedding_model)
    return FAISS.load_local(
        output_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
