import os
import json
import re
import html
from pathlib import Path
from difflib import SequenceMatcher
from bs4 import BeautifulSoup

DATA_DIR      = "data"
RAW_HTML_DIR  = os.path.join(DATA_DIR, "raw_html")
CHUNKS_DIR    = os.path.join(DATA_DIR, "mdkey_chunks")
REPORT_DIR    = os.path.join(DATA_DIR, "comparison_reports")

os.makedirs(REPORT_DIR, exist_ok=True)


def normalize(text):
    text = html.unescape(text)

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(" ")

    # Remove timestamps like [00:01]->[00:33]
    text = re.sub(r"\[\d{2}:\d{2}\]\s*->\s*\[\d{2}:\d{2}\]", " ", text)

    # Remove standalone timestamps like [00:01]
    text = re.sub(r"\[\d{2}:\d{2}\]", " ", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def sliding_match(chunk, document, window_padding=300):
    """
    Finds the best matching window in the HTML text.

    Returns:
        score,
        best_snippet
    """

    chunk_len = len(chunk)

    window = max(chunk_len + window_padding, chunk_len)

    best_score = 0
    best_text = ""

    step = max(50, chunk_len // 10)

    for i in range(0, max(1, len(document) - window), step):
        piece = document[i:i + window]
        score = similarity(chunk, piece)

        if score > best_score:
            best_score = score
            best_text = piece

            if score == 1:
                break

    return best_score, best_text


html_files = list(Path(RAW_HTML_DIR).glob("*.html"))

for html_file in html_files:

    json_file = Path(CHUNKS_DIR) / (html_file.stem + ".json")

    if not json_file.exists():
        print(f"Missing JSON for {html_file.name}")
        continue

    print(f"Checking {html_file.name}")

    ##########################################
    # Load HTML
    ##########################################

    with open(html_file, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    html_text = normalize(soup.get_text(" "))

    ##########################################
    # Load JSON
    ##########################################

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    report = []

    for idx, chunk in enumerate(data["chunks"], start=1):

        chunk_text = normalize(chunk["text"])

        # Exact containment
        if chunk_text in html_text:

            score = 1.0
            status = "PERFECT MATCH"

            snippet = chunk_text[:300]

        else:

            score, snippet = sliding_match(chunk_text, html_text)

            if score >= 0.90:
                status = "90% MATCH"

            else:
                status = "POSSIBLE HALLUCINATION"

        report.append({
            "chunk_id": chunk.get("chunk_id"),
            "position": idx,
            "score": round(score, 4),
            "status": status,
            "title": chunk.get("title"),
            "summary": chunk.get("summary"),
            "best_match_snippet": snippet[:500]
        })

    report_file = Path(REPORT_DIR) / (html_file.stem + "_comparison.json")

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved -> {report_file}")

print("Done.")