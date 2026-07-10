"""
news_pipeline.py
================
Parse -> chunk -> index HTML news articles for a RAG pipeline.

    HTML  -->  extract_metadata()   read header block above article body
          -->  clean_html()         BeautifulSoup noise removal
          -->  to_markdown()        Docling structural conversion
          -->  chunk_markdown()     MDKeyChunker + ChatAI enrichment
          -->  index_chunks()       ChromaDB upsert

HOW METADATA WORKS (answer to supervisor's question)
-----------------------------------------------------
Each HTML file we collect has a small header block at the very top of the
<body> — written by us, not scraped from the news site — that looks like:

    Title: Is A Massive ICE Surge Coming To New York?
    News Source: Forbes
    Political Leaning: Center
    News Type: News
    Published Date: 08 June 2026
    Source Link: https://...
    Topic: Deportation Campaign

This header is OUR annotation layer: we manually assigned the political
leaning, outlet name, and document type BEFORE processing. It sits above
the article body as plain <br/>-separated text.

The pipeline reads this header FIRST with extract_metadata(), then
strips it from the HTML before Docling ever sees the file. Docling
therefore only converts the clean article body to Markdown — it never
touches or re-interprets the metadata.

The extracted metadata is then attached to every chunk as structured
fields (event_id, political_perspective, doc_type, outlet, topic, etc.)
and stored as ChromaDB metadata. This is what the supervisor meant by
"one layer above the chunked document content": the metadata lives
outside the text that gets chunked and embedded, not inside it.

The filename (DC5__Center__News.html) is kept as a redundant backup
reference but is NO LONGER the primary source of political_perspective
or doc_type — the header block is.

File naming convention (still useful for sorting/grouping):
    {event_id}__{perspective}__{doc_type}.html
    e.g.  DC5__Center__News.html

.env:
    CHATAI_API_KEY=...
    CHATAI_BASE_URL=https://chat-ai.academiccloud.de/v1
    CHATAI_LLM_MODEL=meta-llama-3.1-70b-instruct
    CHATAI_EMBED_MODEL=e5-mistral-7b-instruct
"""

import os
import re
import json
import dataclasses
import argparse
from pathlib import Path

from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR      = "data"
RAW_HTML_DIR  = os.path.join(DATA_DIR, "raw_html")
PARSED_MD_DIR = os.path.join(DATA_DIR, "parsed_mds")
CHUNKS_DIR    = os.path.join(DATA_DIR, "mdkey_chunks")
CHROMA_DIR    = os.path.join(DATA_DIR, "chroma_db")
TMP_DIR       = os.path.join(DATA_DIR, "_tmp")

for path in [DATA_DIR, RAW_HTML_DIR, PARSED_MD_DIR, CHUNKS_DIR, CHROMA_DIR, TMP_DIR]:
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------------------------
# ChatAI settings
# ---------------------------------------------------------------------------
CHATAI_BASE_URL  = os.getenv("CHATAI_BASE_URL",   "https://chat-ai.academiccloud.de/v1")
CHATAI_API_KEY   = os.getenv("CHATAI_API_KEY",    "")
CHATAI_LLM_MODEL = os.getenv("CHATAI_LLM_MODEL",  "meta-llama-3.1-70b-instruct")
CHATAI_EMB_MODEL = os.getenv("CHATAI_EMBED_MODEL", "e5-mistral-7b-instruct")
CHROMA_COLLECTION = "political_news"


# ===========================================================================
# STEP 0 — EXTRACT METADATA FROM THE HTML HEADER BLOCK
# ===========================================================================

# These are the field labels we write into the HTML header block ourselves.
# They map to the structured corpus axes used throughout the pipeline.
_HEADER_FIELDS = {
    "title":            "title",
    "news source":      "news_source",
    "political leaning": "political_leaning",
    "news type":        "news_type",
    "published date":   "published_date",
    "source link":      "source_link",
    "topic":            "topic",
}


def extract_metadata(html_path: str) -> dict:
    """
    Read and parse the manually-written metadata header from the HTML file.

    The header sits at the top of <body> as plain text lines separated by
    <br/> tags, before the article content begins. Example:

        Title: Is A Massive ICE Surge Coming To New York?
        News Source: Forbes
        Political Leaning: Center
        News Type: News
        Published Date: 08 June 2026
        Source Link: https://...
        Topic: Deportation Campaign

    This is OUR annotation — we assigned political leaning, outlet name,
    and document type manually. The function extracts these values into a
    clean dict so they can be stored as structured metadata on every chunk,
    completely separate from the article text that gets chunked.

    Falls back gracefully to filename-derived values when a field is absent.
    """
    with open(html_path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    soup = BeautifulSoup(raw, "lxml")

    # Collect the raw text of the header block: everything before the first
    # <div>, <article>, <main>, <p class=...>, or <section> that looks like
    # article body. We look for consecutive <br/>-separated lines near the
    # top of the body.
    body = soup.find("body") or soup
    header_text = ""
    for child in body.children:
        # Stop when we hit a block-level element that is likely article content
        if hasattr(child, "name") and child.name in ("div", "article", "main", "section", "p"):
            # Check if it carries class/id suggesting article body
            cls = " ".join(child.get("class") or []).lower()
            eid = (child.get("id") or "").lower()
            if any(k in cls + eid for k in ("article", "story", "content", "body", "post")):
                break
        header_text += (child.get_text(separator="\n") if hasattr(child, "get_text")
                        else str(child))

    # Parse "Key: Value" lines from the header text
    brs = soup.find_all("br")
    meta = {}

    if len(brs) >= 2:
        header_text = ""

        for node in brs[0].next_siblings:
            if node == brs[1]:
                break
            header_text += str(node)

        for line in header_text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, _, value = line.partition(":")
            key_norm = key.strip().lower()
            value = value.strip()

            if key_norm in _HEADER_FIELDS and value:
                meta[_HEADER_FIELDS[key_norm]] = value

    # Fall back to filename parts for any missing corpus-critical fields
    stem  = Path(html_path).stem
    parts = stem.split("__")
    print(f"    [Meta] extracted from header: {meta}")
    if "political_leaning" not in meta:
        meta["political_leaning"] = parts[1] if len(parts) > 1 else "unknown"
        print(f"    [Meta] political_leaning not found in header — using filename: {meta['political_leaning']}")
    if "news_type" not in meta:
        meta["news_type"] = parts[2] if len(parts) > 2 else "unknown"
        print(f"    [Meta] news_type not found in header — using filename: {meta['news_type']}")

    # event_id always comes from the filename (it is our internal corpus ID)
    meta["event_id"] = parts[0] if len(parts) > 0 else "unknown"

    print(f"    [Meta] news_source={meta.get('news_source','?')}  "
          f"perspective={meta.get('political_leaning','?')}  "
          f"news_type={meta.get('news_type','?')}  "
          f"event={meta.get('event_id','?')}")
    return meta


def strip_metadata_header(html_path: str) -> str:
    """
    Return cleaned HTML with the metadata header block removed.

    The header text (Title:, News Source:, etc.) must not reach Docling or
    MDKeyChunker — it is not article content and would pollute chunks and
    embeddings with repeated boilerplate.

    Strategy: remove all text nodes and <br/> tags that appear BEFORE the
    first substantive block element (div / article / main / section) inside
    <body>. If no such block exists, fall back to regex removal.
    """
    with open(html_path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    soup = BeautifulSoup(raw, "lxml")
    body = soup.find("body")

    if body:
        # Remove leading text nodes and <br/> tags until we hit article content
        children = list(body.children)
        for child in children:
            name = getattr(child, "name", None)
            # Remove plain text nodes and <br/> separators
            if name is None or name == "br":
                child.extract()
                continue
            # Stop at the first element that looks like article content
            break

    return str(soup)


# ===========================================================================
# STEP 1 — HTML CLEANING (noise removal)
# ===========================================================================

_REMOVE_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "header", "footer", "aside",
    "form", "button", "select", "option", "input", "textarea",
    "picture",
}

_NOISE_SUBSTRINGS = [
    "ad-", "-ad", "ads", "advert", "advertisement", "sponsor", "sponsored",
    "promo", "banner", "doubleclick", "dfp-", "gam-", "taboola", "outbrain",
    "navbar", "nav-", "breadcrumb", "pagination", "sidebar", "masthead",
    "topbar", "toolbar", "flyout", "drawer",
    "related", "recommended", "also-read", "read-more", "more-stories",
    "more-articles", "trending", "popular", "most-read", "latest-news",
    "news-card", "card-list", "recirculation",
    "social", "share-", "-share", "sharing", "follow-", "tweet",
    "subscribe", "subscription", "newsletter", "cookie", "consent",
    "gdpr", "paywall", "piano-", "sign-up", "signup",
    "comment", "comments", "disqus",
    "overlay", "modal", "popup", "skip-link", "screen-reader",
    "visually-hidden", "tag-cloud", "tag-list", "widget-",
]


def _is_noise(tag) -> bool:
    combined = " ".join(tag.get("class") or []).lower() + " " + (tag.get("id") or "").lower()
    return any(sub in combined for sub in _NOISE_SUBSTRINGS)


def clean_html(html_content: str) -> str:
    """
    Three-pass noise removal on the already-header-stripped HTML string.

    Pass 1 — remove structural tags that are never article content
    Pass 2 — remove tags whose class/id matches ad/noise patterns
    Pass 3 — isolate the article container; discard everything outside it
    """
    soup = BeautifulSoup(html_content, "lxml")

    for node in soup.find_all(string=lambda t: isinstance(t, Comment)):
        node.extract()

    for tag_name in _REMOVE_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()

    for el in list(soup.find_all(True)):
        try:
            if _is_noise(el):
                el.decompose()
        except Exception:
            pass

    article = (
        soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find("main")
        or soup.find(class_=re.compile(r"(article|story|post)[_-]?(body|content|text|main)", re.I))
        or soup.find(id=re.compile(r"(article|story|post)[_-]?(body|content|text|main)", re.I))
        or soup.find(class_=re.compile(r"entry[_-]?content", re.I))
        or soup.find(class_=re.compile(r"main[_-]?content", re.I))
    )

    return f"<html><body>{article}</body></html>" if article else str(soup)


# ===========================================================================
# STEP 2 — HTML -> MARKDOWN  (Docling)
# ===========================================================================

def to_markdown(stem: str, cleaned_html: str) -> str:
    """Convert cleaned HTML to Markdown via Docling. Falls back to plain text."""
    from docling.document_converter import DocumentConverter

    tmp_path = os.path.join(TMP_DIR, f"{stem}_clean.html")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(cleaned_html)

    try:
        result = DocumentConverter().convert(tmp_path)
        md     = result.document.export_to_markdown()
        print(f"    [Docling] OK  ({len(md):,} chars)")
        return md
    except Exception as exc:
        print(f"    [Docling] FAILED ({exc})  ->  plain-text fallback")
        return _plain_text_fallback(cleaned_html)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _plain_text_fallback(cleaned_html: str) -> str:
    soup  = BeautifulSoup(cleaned_html, "lxml")
    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name.startswith("h"):
            lines.append(f"\n{'#' * int(el.name[1])} {text}\n")
        else:
            lines.append(text)
    return "\n\n".join(lines)


# ===========================================================================
# STEP 3 — SEMANTIC CHUNKING  (MDKeyChunker + ChatAI)
# ===========================================================================

def _configure_mdkeychunker():
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    os.environ["LLM_BASE_URL"] = CHATAI_BASE_URL
    os.environ["LLM_API_KEY"]  = CHATAI_API_KEY
    os.environ["LLM_MODEL"]    = CHATAI_LLM_MODEL
    os.environ.pop("OPENAI_API_KEY",  None)
    os.environ.pop("OPENAI_BASE_URL", None)


def chunk_to_dict(chunk) -> dict:
    """
    Convert an MDKeyChunker Chunk to a plain dict preserving ALL fields.
    Uses Chunk.to_dict() which returns __dict__ directly (all fields included).
    """
    if hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    if dataclasses.is_dataclass(chunk) and not isinstance(chunk, type):
        return dataclasses.asdict(chunk)
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    return vars(chunk).copy()


def chunk_markdown(md_path: str) -> list:
    """Run MDKeyChunker on a Markdown file. Returns list of chunk dicts."""
    from mdkeychunker import Pipeline, Config

    _configure_mdkeychunker()
    pipeline   = Pipeline(Config.from_env())
    raw_chunks = pipeline.process_file(md_path)

    result = []
    for chunk in raw_chunks:
        d = chunk_to_dict(chunk)
        if not (d.get("text") or "").strip():
            continue
        result.append(d)

    print(f"    [MDKeyChunker] {len(result)} non-empty chunks")
    return result


# ===========================================================================
# STEP 4 — BUILD EMBEDDING TEXT
# ===========================================================================

def build_embedding_text(chunk: dict) -> str:
    """
    Build the enriched passage string for the embedding model.
    Only populated fields are included. 'passage:' prefix required by e5-mistral.
    """
    title    = (chunk.get("title")   or "").strip()
    summary  = (chunk.get("summary") or "").strip()
    keywords = [k for k in (chunk.get("keywords") or []) if k]
    text     = (chunk.get("text")    or "").strip()

    parts = []
    if title:
        parts.append(title)
    if summary and summary != title:
        parts.append(summary)
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords))
    if text:
        parts.append(text)

    body = " | ".join(parts) if parts else text or "[empty]"
    return f"passage: {body}"


# ===========================================================================
# STEP 5 — CHROMADB INDEXING
# ===========================================================================

def get_chroma_collection():
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    ef = OpenAIEmbeddingFunction(
        api_base=CHATAI_BASE_URL, api_key=CHATAI_API_KEY, model_name=CHATAI_EMB_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=ef)


def index_chunks(chunks: list, filename: str, meta: dict, collection) -> int:
    """
    Upsert enriched chunks into ChromaDB.

    The 'meta' dict comes from extract_metadata() — it is stored on every
    chunk as structured fields, completely separate from the embedded text.
    ChromaDB metadata values must be str/int/float/bool (no lists or None).
    """
    if not chunks:
        return 0

    def _to_str(v) -> str:
        if v is None:
            return ""
        if isinstance(v, list):
            return ", ".join(
                item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                for item in v
            )
        return str(v)

    ids, docs, metas = [], [], []
    stem = Path(filename).stem

    for i, chunk in enumerate(chunks):
        ids.append(f"{stem}__chunk_{i}")
        docs.append(build_embedding_text(chunk))
        metas.append({
            # --- Corpus-level metadata (from header block, NOT from chunk text) ---
            "source":                filename,
            # "event_id":              meta.get("event_id", "unknown"),
            "political_leaning": meta.get("political_leaning", "unknown"),
            "news_type":              meta.get("news_type", "unknown"),
            "news_source":                meta.get("news_source", "unknown"),
            "topic":                 meta.get("topic", "unknown"),
            "published_date":        meta.get("published_date", "unknown"),
            "title":         meta.get("title",         "unknown"),
            "source_link":           meta.get("source_link", "unknown"),
            # --- Chunk-level fields (from MDKeyChunker) ---
            "chunk_id":              _to_str(chunk.get("chunk_id")),
            "section_title":         _to_str(chunk.get("section_title")),
            "content_types":         _to_str(chunk.get("content_types")),
            "start_line":            int(chunk.get("start_line")     or 0),
            "end_line":              int(chunk.get("end_line")       or 0),
            "position_index":        int(chunk.get("position_index") or i),
            "previous_chunk_id":     _to_str(chunk.get("previous_chunk_id")),
            "next_chunk_id":         _to_str(chunk.get("next_chunk_id")),
            "token_count":           int(chunk.get("token_count")    or 0),
            "title":                 _to_str(chunk.get("title"))[:500],
            "summary":               _to_str(chunk.get("summary"))[:500],
            "keywords":              _to_str(chunk.get("keywords"))[:500],
            "key":                   _to_str(chunk.get("key"))[:200],
            "related_keys":          _to_str(chunk.get("related_keys"))[:500],
            "entities":              _to_str(chunk.get("entities"))[:500],
            "questions":             _to_str(chunk.get("questions"))[:500],
        })

    BATCH = 100
    for start in range(0, len(ids), BATCH):
        collection.upsert(
            ids       = ids  [start : start + BATCH],
            documents = docs [start : start + BATCH],
            metadatas = metas[start : start + BATCH],
        )

    print(f"    [ChromaDB] indexed {len(ids)} chunks")
    return len(ids)


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def process_file(html_path: str, force: bool = False) -> dict:
    """Full pipeline for one HTML news file."""
    filename = os.path.basename(html_path)
    stem     = Path(html_path).stem

    print(f"\n{'─' * 64}")
    print(f"  FILE : {filename}")
    print(f"{'─' * 64}")

    json_out = os.path.join(CHUNKS_DIR, f"{stem}.json")
    if os.path.exists(json_out) and not force:
        print(f"  Already processed -> {json_out}  (use --force to redo)")
        return {"file": filename, "skipped": True}

    # 0. Extract metadata from the HTML header block
    print("  [0/4] Extracting corpus metadata from header block ...")
    meta = extract_metadata(html_path)

    # Strip the metadata header from HTML before passing to Docling
    stripped_html = strip_metadata_header(html_path)

    # 1. Clean HTML (noise removal)
    print("  [1/4] Cleaning HTML ...")
    clean = clean_html(stripped_html)

    # 2. HTML -> Markdown
    print("  [2/4] Docling: HTML -> Markdown ...")
    md = to_markdown(stem, clean)
    md_path = os.path.join(PARSED_MD_DIR, f"{stem}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    # 3. Chunk + enrich
    print("  [3/4] MDKeyChunker: chunking + enriching ...")
    chunks = chunk_markdown(md_path)

    if not chunks:
        print("  WARNING: no chunks produced — skipping index step.")
        return {"file": filename, "skipped": False, "chunks": 0}

    # Attach corpus metadata to every chunk dict
    # Metadata comes from the extracted header block, NOT re-parsed from text
    # for c in chunks:
    #     c["source"]                = filename
    #     c["event_id"]              = meta.get("event_id", "unknown")
    #     c["political_leaning"] = meta.get("political_leaning", "unknown")
    #     c["news_type"]              = meta.get("news_type", "unknown")
    #     c["news_source"]                = meta.get("news_source", "unknown")
    #     c["topic"]                 = meta.get("topic", "")
    #     c["published_date"]        = meta.get("published_date", "")
    #     c["title"]         = meta.get("title", "")
    
    output_json = {
    "metadata": {
        "source": filename,
        # "event_id": meta.get("event_id", "unknown"),
        "political_leaning": meta.get("political_leaning", "unknown"),
        "news_type": meta.get("news_type", "unknown"),
        "news_source": meta.get("news_source", "unknown"),
        "topic": meta.get("topic", ""),
        "published_date": meta.get("published_date", ""),
        "title": meta.get("title", ""),
        "source_link": meta.get("source_link", "")
    },
    "chunks": chunks
    }

    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(output_json, fh, ensure_ascii=False, indent=2)
    print(f"    {len(chunks)} chunks saved -> {json_out}")

    # Preview first chunk
    c0 = chunks[0]
    # print(f"\n    -- metadata preview (stored above chunk content) --")
    # print(f"    event_id              : {c0.get('event_id')}")
    # print(f"    political_leaning : {c0.get('political_leaning')}")
    # print(f"    news_type             : {c0.get('news_type')}")
    # print(f"    news_source           : {c0.get('news_source')}")
    # print(f"    topic                 : {c0.get('topic')}")
    # print(f"    published_date        : {c0.get('published_date')}")
    # print(f"    -- chunk content --")
    # print(f"    chunk_id              : {c0.get('chunk_id')}")
    # print(f"    title                 : {str(c0.get('title',''))[:80]}")
    # print(f"    text[:80]             : {str(c0.get('text',''))[:80].replace(chr(10),' ')}")

    # 4. Index into ChromaDB
    print("\n  [4/4] Indexing into ChromaDB ...")
    n = index_chunks(chunks, filename, meta, get_chroma_collection())

    return {"file": filename, "skipped": False, "chunks": len(chunks), "indexed": n}


def run_batch(html_dir: str, force: bool = False):
    html_files = sorted(f for f in os.listdir(html_dir) if f.lower().endswith(".html"))
    if not html_files:
        print(f"No .html files found in: {html_dir}")
        return

    print(f"\nFound {len(html_files)} HTML file(s) in {html_dir}")
    total_chunks = total_indexed = skipped = 0

    for fname in html_files:
        r = process_file(os.path.join(html_dir, fname), force=force)
        if r.get("skipped"):
            skipped += 1
        else:
            total_chunks  += r.get("chunks",  0)
            total_indexed += r.get("indexed", 0)

    print(f"\n{'=' * 64}")
    print(f"  Processed : {len(html_files) - skipped} files")
    print(f"  Skipped   : {skipped}")
    print(f"  Chunks    : {total_chunks}")
    print(f"  Indexed   : {total_indexed} vectors")
    print(f"{'=' * 64}\n")


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse, chunk, and index HTML news articles.")
    ap.add_argument("--dir",   default=RAW_HTML_DIR, help="Folder with .html files")
    ap.add_argument("--file",  default=None,          help="Single HTML file to process")
    ap.add_argument("--force", action="store_true",   help="Re-parse already-processed files")
    args = ap.parse_args()

    if not CHATAI_API_KEY:
        raise SystemExit("ERROR: CHATAI_API_KEY not set in .env")

    if args.file:
        process_file(args.file, force=args.force)
    else:
        os.makedirs(args.dir, exist_ok=True)
        run_batch(args.dir, force=args.force)
