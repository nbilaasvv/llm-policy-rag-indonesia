"""RAG querying utilities using Gemini via an environment variable."""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DEFAULT_MODEL = "gemini-2.5-flash"


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0):
    """Create Gemini client using GOOGLE_API_KEY from the environment."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY belum tersedia. Set environment variable terlebih dahulu "
            "(jangan hard-code API key di notebook)."
        )
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm=None):
    llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_template(
        """Jawab pertanyaan berdasarkan konteks berikut.

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:"""
    )
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def query_rag(question: str, retriever, llm=None, k: int = 5) -> str:
    """Run a direct RAG query without rebuilding the chain."""
    llm = llm or get_llm()
    docs = retriever.invoke(question)
    context = format_docs(docs[:k])
    prompt = f"""Anda adalah analis kebijakan publik dan peneliti tata kelola digital.

Gunakan hanya informasi yang terdapat pada KONTEKS untuk menjawab PERTANYAAN.
Jika informasi tidak tersedia secara eksplisit, lakukan sintesis berdasarkan isi dokumen,
bukan asumsi pribadi.

KONTEKS:
{context}

PERTANYAAN:
{question}

Jawab dalam Bahasa Indonesia baku, objektif, akademik, dan berbasis konteks."""
    return llm.invoke(prompt).content


def classify_document(doc, llm=None):
    llm = llm or get_llm()
    prompt = f"""Klasifikasikan teks berikut ke salah satu kategori:
- Masalah SDM
- Infrastruktur Teknologi
- Regulasi/Kebijakan
- Dinamika Global

Teks:
{doc.page_content}

Jawab hanya dengan nama kategorinya."""
    return llm.invoke(prompt).content.strip()


def automated_coding(vectorstore, llm=None, k: int = 1000):
    import pandas as pd
    docs = vectorstore.similarity_search("", k=k)
    results = []
    for doc in docs:
        results.append({
            "text": doc.page_content[:200],
            "category": classify_document(doc, llm),
            "metadata": doc.metadata,
        })
    return pd.DataFrame(results)


def keyword_frequency(vectorstore, keywords, k: int = 500):
    import re
    import pandas as pd
    docs = vectorstore.similarity_search("", k=k)
    text = " ".join(d.page_content for d in docs)
    counts = {
        word: len(re.findall(re.escape(word), text, re.IGNORECASE))
        for word in keywords
    }
    return pd.DataFrame(counts.items(), columns=["Keyword", "Frequency"])
