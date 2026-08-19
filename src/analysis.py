"""Local corpus analysis: sentiment, word frequency/cloud, and co-occurrence."""

import re
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from nltk.corpus import stopwords
from transformers import pipeline
from wordcloud import WordCloud


def get_stop_words():
    return set(stopwords.words("indonesian")).union(
        set(stopwords.words("english"))
    )


def clean_text_for_analysis(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_sentiment_analyzers():
    return (
        pipeline(
            "sentiment-analysis",
            model="w11wo/indonesian-roberta-base-sentiment-classifier",
        ),
        pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        ),
    )


def analyze_sentiment(text, sentiment_id, sentiment_en):
    from langdetect import detect
    if not text or not str(text).strip():
        return "NEUTRAL", 0.0
    text = str(text)[:512]
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    result = (sentiment_id if lang == "id" else sentiment_en)(text)[0]
    return result["label"], result["score"]


def add_sentiment_columns(df: pd.DataFrame, text_col="text"):
    sentiment_id, sentiment_en = build_sentiment_analyzers()
    result = df.copy()
    values = result[text_col].apply(
        lambda x: analyze_sentiment(x, sentiment_id, sentiment_en)
    )
    result[["sentiment", "confidence"]] = values.apply(pd.Series)
    result["sentiment_score"] = result["sentiment"].map(
        {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0,
         "positive": 1, "negative": -1, "neutral": 0}
    )
    return result


def word_frequencies(text_series, top_n=30):
    stop_words = get_stop_words()
    words = clean_text_for_analysis(" ".join(text_series.astype(str))).split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return Counter(words).most_common(top_n)


def plot_wordcloud(text_series, top_n=50, title="Word Cloud"):
    frequencies = dict(word_frequencies(text_series, top_n))
    wc = WordCloud(width=1200, height=600, background_color="white",
                   max_words=top_n, random_state=42)
    wc.generate_from_frequencies(frequencies)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def build_cooccurrence_network(text_series, window_size=4, top_n=40, min_freq=2):
    stop_words = get_stop_words()
    words = clean_text_for_analysis(
        " ".join(text_series.astype(str))
    ).split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    word_freq = Counter(words)
    top_words = set(w for w, _ in word_freq.most_common(top_n))

    pair_counts = Counter()
    for i in range(len(words)):
        window = [w for w in words[i:i + window_size] if w in top_words]
        pair_counts.update(combinations(window, 2))

    graph = nx.Graph()
    for pair, weight in pair_counts.items():
        if weight >= min_freq:
            graph.add_edge(*pair, weight=weight)
    return graph, word_freq


def visualize_network(graph, word_freq, title="Co-occurrence Network"):
    if not graph.nodes:
        print("Graph kosong — turunkan min_freq/top_n.")
        return
    fig, ax = plt.subplots(figsize=(14, 12))
    pos = nx.spring_layout(graph, k=0.6, seed=42)
    node_sizes = [word_freq[node] * 25 for node in graph.nodes()]
    node_colors = [word_freq[node] for node in graph.nodes()]
    edge_widths = [data["weight"] * 0.3 for _, _, data in graph.edges(data=True)]

    nodes = nx.draw_networkx_nodes(
        graph, pos, node_size=node_sizes, node_color=node_colors,
        cmap=plt.cm.viridis, alpha=0.9, ax=ax
    )
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.4, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold", ax=ax)
    fig.colorbar(nodes, ax=ax, label="Word Frequency")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
