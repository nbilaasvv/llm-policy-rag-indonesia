"""Document ingestion utilities: PDF/CSV extraction and chunking."""

from pathlib import Path
import re
import pandas as pd
import pdfplumber
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Normalize extracted text while preserving Indonesian characters."""
    if not text:
        return ""
    text = re.sub(r"\n\d+\n", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF, using pdfplumber then PyPDF2 as fallback."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File tidak ditemukan: {file_path}")
        return ""

    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            text_parts = [page.extract_text() or "" for page in pdf.pages]
        text = clean_text("\n".join(text_parts))
        if text:
            print(f"✔ pdfplumber: {file_path}")
            return text
    except Exception as exc:
        print(f"⚠ pdfplumber gagal untuk {file_path}: {exc}")

    try:
        reader = PdfReader(str(path))
        text = clean_text("\n".join(page.extract_text() or "" for page in reader.pages))
        if text:
            print(f"✔ PyPDF2: {file_path}")
            return text
    except Exception as exc:
        print(f"⚠ PyPDF2 gagal untuk {file_path}: {exc}")

    print(f"❌ Gagal membaca PDF: {file_path}")
    return ""


def process_csv(file_path: str, text_column: str = "text") -> str:
    """Read a CSV, validate the text column, deduplicate rows, and combine text."""
    df = pd.read_csv(file_path)
    if text_column not in df.columns:
        raise ValueError(f"Kolom '{text_column}' tidak ditemukan: {file_path}")
    df = df[[text_column]].dropna().drop_duplicates()
    print(f"✔ {file_path}: {len(df)} baris teks")
    return clean_text(" ".join(df[text_column].astype(str)))


def create_chunks(text: str, label: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split text into LangChain Documents with a source/category label."""
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [
        Document(page_content=chunk, metadata={"kategori": label})
        for chunk in splitter.split_text(text)
    ]


def ingest_sources(sources: dict, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Extract and chunk a mapping of source labels to file paths.

    File types supported: PDF and CSV.
    """
    documents = []
    for label, file_path in sources.items():
        suffix = Path(file_path).suffix.lower()
        try:
            if suffix == ".pdf":
                text = extract_pdf_text(file_path)
            elif suffix == ".csv":
                text = process_csv(file_path)
            else:
                print(f"⚠ Format tidak didukung: {file_path}")
                continue
            chunks = create_chunks(text, label, chunk_size, chunk_overlap)
            documents.extend(chunks)
            print(f"  → {label}: {len(chunks)} chunk")
        except Exception as exc:
            print(f"❌ {label}: {exc}")
    return documents
