from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

# Try to use textstat; fallback to a simple heuristic
def _import_textstat():
    try:
        import textstat
        return textstat
    except Exception:
        return None

TEXTSTAT = _import_textstat()

def flesch_reading_ease(text: str) -> float:
    if TEXTSTAT:
        try:
            return TEXTSTAT.flesch_reading_ease(text)
        except Exception:
            pass
    # fallback: simple heuristic based on average sentence length
    if not text:
        return 0.0
    sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    words = max(1, len(text.split()))
    # heuristic: higher value better readability
    asl = words / sentences
    # map typical ranges into 0-100-like value
    score = max(0.0, min(100.0, 206.835 - 1.015 * asl - 84.6 * 1.5))
    return score

def top_keywords_from_tfidf(corpus: List[str], top_n=5) -> List[str]:
    vec = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
    X = vec.fit_transform(corpus)
    feature_names = vec.get_feature_names_out()
    # For each doc, get top features
    top_keywords = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        if row.sum() == 0:
            top_keywords.append([])
            continue
        idx = row.argsort()[::-1][:top_n]
        top_keywords.append([feature_names[j] for j in idx if row[j] > 0])
    return top_keywords, vec, X

def compute_features(df: pd.DataFrame, inplace=True) -> Tuple[pd.DataFrame, TfidfVectorizer, any]:
    """
    Input: df with 'url' and 'body_text'
    Returns: features_df, tfidf_vectorizer, tfidf_matrix
    """
    out = df.copy() if inplace else df.copy()
    out["body_text"] = out["body_text"].fillna("").astype(str)
    out["word_count"] = out["body_text"].apply(lambda t: len(t.split()))
    out["sentence_count"] = out["body_text"].apply(lambda t: max(1, t.count(".") + t.count("!") + t.count("?")))
    out["flesch_reading_ease"] = out["body_text"].apply(flesch_reading_ease)
    keywords, vec, X = top_keywords_from_tfidf(out["body_text"].tolist(), top_n=5)
    out["top_keywords"] = keywords
    return out, vec, X

def find_duplicates(tfidf_matrix, urls: List[str], sim_threshold=0.9):
    """
    returns list of dicts: {"url1":..,"url2":..,"similarity":..}
    """
    duplicates = []
    if tfidf_matrix is None or tfidf_matrix.shape[0] < 2:
        return duplicates
    # compute cosine similarities; use dense if small dataset
    sims = cosine_similarity(tfidf_matrix)
    n = sims.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            sim = float(sims[i, j])
            if sim >= sim_threshold:
                duplicates.append({"url1": urls[i], "url2": urls[j], "similarity": round(sim, 4)})
    return duplicates
