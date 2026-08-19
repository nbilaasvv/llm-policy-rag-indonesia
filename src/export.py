"""Export structured analysis outputs (CSV) for Power BI dashboards.

Produces:
  - word_frequency.csv     (word, frequency)
  - sentiment_results.csv  (text, sentiment, confidence, sentiment_score)
  - cooccurrence_edges.csv (word1, word2, weight)
  - keyword_frequency.csv  (Keyword, Frequency)  [optional, needs a built vectorstore]

Usage:
    python export.py --csv scraping_news.csv --out data/processed
    python export.py --csv scraping_news.csv --out data/processed --skip-keyword-freq
"""

import argparse
from pathlib import Path

import pandas as pd

from analysis import add_sentiment_columns, word_frequencies, build_cooccurrence_network
from build_vectorstore import load_vectorstore
from rag_query import keyword_frequency


def export_word_frequency(df: pd.DataFrame, out_dir: Path, text_col="text", top_n=50):
    freqs = word_frequencies(df[text_col], top_n=top_n)
    out = pd.DataFrame(freqs, columns=["word", "frequency"])
    out.to_csv(out_dir / "word_frequency.csv", index=False)
    print(f"✔ word_frequency.csv ({len(out)} rows)")
    return out


def export_sentiment(df: pd.DataFrame, out_dir: Path, text_col="text"):
    result = add_sentiment_columns(df, text_col=text_col)
    result.to_csv(out_dir / "sentiment_results.csv", index=False)
    print(f"✔ sentiment_results.csv ({len(result)} rows)")
    return result


def export_cooccurrence(df: pd.DataFrame, out_dir: Path, text_col="text",
                         window_size=4, top_n=40, min_freq=2):
    graph, _ = build_cooccurrence_network(
        df[text_col], window_size=window_size, top_n=top_n, min_freq=min_freq
    )
    edges = [
        {"word1": u, "word2": v, "weight": d["weight"]}
        for u, v, d in graph.edges(data=True)
    ]
    out = pd.DataFrame(edges)
    out.to_csv(out_dir / "cooccurrence_edges.csv", index=False)
    print(f"✔ cooccurrence_edges.csv ({len(out)} rows)")
    return out


def export_keyword_frequency(vectorstore_dir: str, keywords: list, out_dir: Path):
    vectorstore = load_vectorstore(vectorstore_dir)
    out = keyword_frequency(vectorstore, keywords)
    out.to_csv(out_dir / "keyword_frequency.csv", index=False)
    print(f"✔ keyword_frequency.csv ({len(out)} rows)")
    return out


def main():
    parser = argparse.ArgumentParser(description="Export analysis outputs for Power BI.")
    parser.add_argument("--csv", default="scraping_news.csv", help="Corpus CSV dengan kolom teks")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--vectorstore", default="vectorstore_reformasi_2045")
    parser.add_argument("--keywords", nargs="*",
                         default=["AI", "Digital-First", "Korupsi", "Cybersecurity"])
    parser.add_argument("--skip-keyword-freq", action="store_true",
                         help="Lewati export ini jika vectorstore belum dibangun")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[[args.text_col]].dropna().drop_duplicates()

    export_word_frequency(df, out_dir, text_col=args.text_col)
    export_sentiment(df, out_dir, text_col=args.text_col)
    export_cooccurrence(df, out_dir, text_col=args.text_col)

    if not args.skip_keyword_freq:
        try:
            export_keyword_frequency(args.vectorstore, args.keywords, out_dir)
        except FileNotFoundError as exc:
            print(f"⚠ Dilewati: {exc}")

    print(f"\nSemua file CSV siap di: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
