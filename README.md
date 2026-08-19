# Refactored LLM Policy RAG

## Struktur
- `ingest.py`: ekstraksi PDF/CSV, cleaning, chunking.
- `build_vectorstore.py`: embedding + FAISS build/load.
- `rag_query.py`: Gemini LLM, RAG chain, query, coding, keyword frequency.
- `analysis.py`: sentiment, word frequency/wordcloud, co-occurrence.
- `LLM_PREANALISIS_REFACTORED.ipynb`: notebook utama yang sudah dipangkas.

## API key
API key lama yang tertanam di notebook harus **direvoke secara manual** pada layanan Google yang menerbitkannya. Refactor ini sengaja tidak menyimpan key tersebut.

Setelah revoke, buat key baru dan expose sebagai environment variable:
`GOOGLE_API_KEY`.

Jangan commit `.env`; gunakan `.env.example` sebagai template.

## Catatan
Notebook mempertahankan sumber data dan tujuan analisis dari notebook awal, tetapi menghapus eksperimen/duplikasi cell yang berulang.
