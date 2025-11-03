from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import re
import requests
from typing import Tuple

# Basic HTML -> text extraction utility functions
def clean_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_title_and_body_from_html(html: str) -> Tuple[str, str]:
    if not html or not isinstance(html, str):
        return "", ""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    # collect paragraphs and article text
    texts = []
    # try <article> then <main> then <p>
    article = soup.find("article")
    if article:
        texts = [p.get_text(" ", strip=True) for p in article.find_all(["p", "div"])]
    else:
        main = soup.find("main")
        if main:
            texts = [p.get_text(" ", strip=True) for p in main.find_all(["p", "div"])]
        else:
            texts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body = " ".join(texts)
    return clean_whitespace(title), clean_whitespace(body)

def parse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe with at least 'url' and optional 'html_content'.
    Returns a dataframe with url, title, body_text, word_count.
    """
    out = df.copy()
    if "title" not in out.columns:
        out["title"] = ""
    if "body_text" not in out.columns:
        out["body_text"] = ""
    for idx, row in out.iterrows():
        try:
            if pd.notna(row.get("html_content")) and row.get("html_content"):
                title, body = extract_title_and_body_from_html(row.get("html_content"))
                out.at[idx, "title"] = title
                out.at[idx, "body_text"] = body
            else:
                # leave blank; don't attempt network scraping here
                out.at[idx, "title"] = row.get("title", "")
                out.at[idx, "body_text"] = row.get("body_text", "")
        except Exception:
            out.at[idx, "title"] = out.at[idx, "title"] if out.at[idx, "title"] else ""
            out.at[idx, "body_text"] = out.at[idx, "body_text"] if out.at[idx, "body_text"] else ""
    # compute word count
    out["body_text"] = out["body_text"].fillna("").astype(str)
    out["word_count"] = out["body_text"].apply(lambda t: len(t.split()))
    return out

# Minimal scraping helper for single-url analyze_url (rate limiting/delay not enforced here)
def analyze_url(url: str, timeout=8):
    if not url:
        raise ValueError("URL empty")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SEO-Detector/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    title, body = extract_title_and_body_from_html(resp.text)
    df = pd.DataFrame([{"url": url, "title": title, "body_text": body, "word_count": len(body.split())}])
    return df
