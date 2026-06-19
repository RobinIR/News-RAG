"""
news_pipeline.py
================
Parse -> chunk -> index HTML news articles for a RAG pipeline.

    HTML  -->  clean_html()       BeautifulSoup noise removal
          -->  to_markdown()      Docling structural conversion
          -->  chunk_markdown()   MDKeyChunker + ChatAI enrichment
          -->  index_chunks()     ChromaDB upsert
Usage
-----
    python news_pipeline.py                         # process data/raw_html/
    python news_pipeline.py --dir /my/html/folder
    python news_pipeline.py --file article.html     # single file
    python news_pipeline.py --force                 # re-parse existing files

.env
----
    CHATAI_API_KEY=<your key>
    CHATAI_BASE_URL=https://chat-ai.academiccloud.de/v1   # default
    CHATAI_LLM_MODEL=meta-llama-3.1-70b-instruct          # default
    CHATAI_EMBED_MODEL=e5-mistral-7b-instruct              # default
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

# ---------------------------------------------------------------------------
# Create directories if they don't exist
# ---------------------------------------------------------------------------
for path in [
    DATA_DIR,
    RAW_HTML_DIR,
    PARSED_MD_DIR,
    CHUNKS_DIR,
    CHROMA_DIR,
    TMP_DIR,
]:
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------------------------
# ChatAI settings  (read from .env)
# ---------------------------------------------------------------------------
CHATAI_BASE_URL  = os.getenv("CHATAI_BASE_URL",   "https://chat-ai.academiccloud.de/v1")
CHATAI_API_KEY   = os.getenv("CHATAI_API_KEY",    "")
CHATAI_LLM_MODEL = os.getenv("CHATAI_LLM_MODEL",  "meta-llama-3.1-70b-instruct")
CHATAI_EMB_MODEL = os.getenv("CHATAI_EMBED_MODEL", "e5-mistral-7b-instruct")
CHROMA_COLLECTION = "political_news"


# ===========================================================================
# STEP 1 — HTML CLEANING                                        (fixes BUG 1)
# ===========================================================================

_REMOVE_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "header", "footer", "aside",
    "form", "button", "select", "option", "input", "textarea",
    "picture",
}

_NOISE_SUBSTRINGS = [
    # ads / sponsored
    "ad-", "-ad", "ads", "advert", "advertisement", "sponsor", "sponsored",
    "promo", "promotion", "banner", "doubleclick", "dfp-", "gam-",
    "taboola", "outbrain", "chumbox",
    # navigation / chrome
    "navbar", "nav-", "-nav", "breadcrumb", "pagination", "pager",
    "sidebar", "side-bar", "masthead", "topbar", "toolbar", "flyout",
    "drawer", "offcanvas",
    # related / recommended cards
    "related", "recommended", "also-read", "read-more",
    "more-stories", "more-articles", "more-news",
    "trending", "popular", "most-read", "latest-news",
    "news-card", "newscard", "card-list", "recirculation", "recirc",
    # social / sharing
    "social", "share-", "-share", "sharing", "follow-", "tweet",
    # subscription / cookie / paywall
    "subscribe", "subscription", "newsletter", "email-signup",
    "cookie", "consent", "gdpr", "paywall", "piano-", "sign-up", "signup",
    # comments
    "comment", "comments", "disqus", "livefyre",
    # misc
    "overlay", "modal", "popup", "lightbox",
    "skip-link", "screen-reader", "visually-hidden", "sr-only",
    "tag-cloud", "tag-list", "widget-",
]


def _is_noise(tag) -> bool:
    classes  = " ".join(tag.get("class") or []).lower()
    tag_id   = (tag.get("id") or "").lower()
    combined = classes + " " + tag_id
    return any(sub in combined for sub in _NOISE_SUBSTRINGS)


def clean_html(html_path: str) -> str:
    """
    Three-pass HTML noise removal before Docling:
      Pass 1 — structural tags that can never be article body
      Pass 2 — class/id heuristic (ads, nav, sidebars, related cards, ...)
      Pass 3 — isolate the article container; discard everything outside it
    """
    with open(html_path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    soup = BeautifulSoup(raw, "lxml")

    # Remove HTML comments (often contain ad-injection markup)
    for node in soup.find_all(string=lambda t: isinstance(t, Comment)):
        node.extract()

    # Pass 1 — structural
    for tag_name in _REMOVE_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()

    # Pass 2 — class/id heuristic  (snapshot list to avoid live-tree issues)
    for el in list(soup.find_all(True)):
        try:
            if _is_noise(el):
                el.decompose()
        except Exception:
            pass

    # Pass 3 — locate the article container
    article = (
        soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find("main")
        or soup.find(class_=re.compile(
            r"(article|story|post)[_-]?(body|content|text|main|detail)", re.I))
        or soup.find(id=re.compile(
            r"(article|story|post)[_-]?(body|content|text|main|detail)", re.I))
        or soup.find(class_=re.compile(r"entry[_-]?content",   re.I))
        or soup.find(class_=re.compile(r"content[_-]?body",    re.I))
        or soup.find(class_=re.compile(r"main[_-]?content",    re.I))
        or soup.find(class_=re.compile(r"article[_-]?wrapper", re.I))
    )

    if article:
        return f"<html><body>{article}</body></html>"

    return str(soup)


# ===========================================================================
# STEP 2 — HTML -> MARKDOWN  (Docling)
# ===========================================================================

def to_markdown(html_path: str, cleaned_html: str) -> str:
    """Convert cleaned HTML to Markdown via Docling. Falls back to plain text."""
    from docling.document_converter import DocumentConverter

    stem = Path(html_path).stem
    os.makedirs(TMP_DIR, exist_ok=True)
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
    """
    MDKeyChunker reads all settings from env vars via Config.from_env().
    Point every key at ChatAI and remove any OpenAI redirects.
    """
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    os.environ["LLM_BASE_URL"] = CHATAI_BASE_URL
    os.environ["LLM_API_KEY"]  = CHATAI_API_KEY
    os.environ["LLM_MODEL"]    = CHATAI_LLM_MODEL
    os.environ.pop("OPENAI_API_KEY",  None)
    os.environ.pop("OPENAI_BASE_URL", None)


def chunk_to_dict(chunk) -> dict:
    """
    Convert an MDKeyChunker Chunk object to a plain dict WITH ALL FIELDS.
    """
    # Primary: Chunk.to_dict() is defined in models.py and returns __dict__
    if hasattr(chunk, "to_dict"):
        return chunk.to_dict()

    # Secondary: dataclasses.asdict() — works on any @dataclass, handles
    # nested dataclass fields automatically
    if dataclasses.is_dataclass(chunk) and not isinstance(chunk, type):
        return dataclasses.asdict(chunk)

    # Tertiary: Pydantic v2 / v1 (future-proofing if they ever migrate)
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump()
    if hasattr(chunk, "dict"):
        return chunk.dict()

    # Last resort: vars() — still returns everything, just less safe
    return vars(chunk).copy()


def chunk_markdown(md_path: str) -> list:
    """
    Run MDKeyChunker pipeline on a Markdown file.
    Returns a list of dicts; each dict contains EVERY field from the Chunk.
    Chunks with empty text bodies are dropped.
    """
    from mdkeychunker import Pipeline, Config

    _configure_mdkeychunker()
    pipeline   = Pipeline(Config.from_env())
    raw_chunks = pipeline.process_file(md_path)   # returns list[Chunk]

    result = []
    for chunk in raw_chunks:
        d = chunk_to_dict(chunk)

        # Safety: skip chunks whose body text is empty
        if not (d.get("text") or "").strip():
            print(f"    [MDKeyChunker] skipping empty chunk (id={d.get('chunk_id','')})")
            continue

        result.append(d)

    print(f"    [MDKeyChunker] {len(result)} non-empty chunks  "
          f"(fields per chunk: {sorted(result[0].keys()) if result else 'n/a'})")
    return result


# ===========================================================================
# STEP 4 — BUILD EMBEDDING TEXT
# ===========================================================================

def build_embedding_text(chunk: dict) -> str:
    """
    Build the enriched passage string for the embedding model.

    Rules:
    - Only include a section when it has actual content (no ". . Keywords: .")
    - Never repeat the title in the summary slot if they are identical
    - "passage:" prefix is required by e5-mistral (asymmetric model)
    - " | " separates semantic sections cleanly
    """
    title    = (chunk.get("title") or "").strip()
    summary  = (chunk.get("summary") or "").strip()
    keywords = [k for k in (chunk.get("keywords") or []) if k]
    text     = (chunk.get("text") or "").strip()

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
        api_base   = CHATAI_BASE_URL,
        api_key    = CHATAI_API_KEY,
        model_name = CHATAI_EMB_MODEL,
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
    )


def index_chunks(chunks: list, filename: str, collection) -> int:
    """
    Upsert enriched chunks into ChromaDB.

    ChromaDB metadata values must be str / int / float / bool — no lists or
    None.  We serialise list fields (keywords, content_types, related_keys,
    entities, questions) as comma-separated strings so nothing is dropped.
    """
    if not chunks:
        return 0

    stem  = Path(filename).stem
    parts = stem.split("__")
    event_id    = parts[0] if len(parts) > 0 else "unknown"
    perspective = parts[1] if len(parts) > 1 else "unknown"
    doc_type    = parts[2] if len(parts) > 2 else "unknown"

    def _to_str(v) -> str:
        """Flatten any value to a ChromaDB-safe string."""
        if v is None:
            return ""
        if isinstance(v, list):
            # lists of strings or dicts (entities)
            return ", ".join(
                item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                for item in v
            )
        return str(v)

    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{stem}__chunk_{i}")
        docs.append(build_embedding_text(chunk))
        metas.append({
            # ── corpus axes (for filtering in retrieval) ────────────────────
            "source":                filename,
            "event_id":              event_id,
            "political_perspective": perspective,
            "doc_type":              doc_type,
            # ── ALL MDKeyChunker Chunk fields ───────────────────────────────
            # Chunker-set
            "section_title":         _to_str(chunk.get("section_title")),
            "content_types":         _to_str(chunk.get("content_types")),
            "start_line":            int(chunk.get("start_line") or 0),
            "end_line":              int(chunk.get("end_line")   or 0),
            # Finalize-set
            "chunk_id":              _to_str(chunk.get("chunk_id")),
            "position_index":        int(chunk.get("position_index") or i),
            "previous_chunk_id":     _to_str(chunk.get("previous_chunk_id")),
            "next_chunk_id":         _to_str(chunk.get("next_chunk_id")),
            "token_count":           int(chunk.get("token_count") or 0),
            # LLM-set
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

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    json_out = os.path.join(CHUNKS_DIR, f"{stem}.json")

    if os.path.exists(json_out) and not force:
        print(f"  Already processed -> {json_out}")
        print(f"  Pass --force to re-parse.")
        return {"file": filename, "skipped": True}

    # 1. Clean HTML
    print("  [1/4] Cleaning HTML ...")
    clean = clean_html(html_path)

    # 2. HTML -> Markdown
    print("  [2/4] Docling: HTML -> Markdown ...")
    md = to_markdown(html_path, clean)
    os.makedirs(PARSED_MD_DIR, exist_ok=True)
    md_path = os.path.join(PARSED_MD_DIR, f"{stem}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"    Markdown saved -> {md_path}")

    # 3. Chunk + enrich
    print("  [3/4] MDKeyChunker: chunking + enriching ...")
    chunks = chunk_markdown(md_path)

    if not chunks:
        print("  WARNING: no chunks produced — skipping index step.")
        return {"file": filename, "skipped": False, "chunks": 0}

    # Attach corpus metadata (fields MDKeyChunker cannot know about)
    stem_parts = stem.split("__")
    for c in chunks:
        c["source"]                = filename
        c["event_id"]              = stem_parts[0] if len(stem_parts) > 0 else "unknown"
        c["political_perspective"] = stem_parts[1] if len(stem_parts) > 1 else "unknown"
        c["doc_type"]              = stem_parts[2] if len(stem_parts) > 2 else "unknown"

    # Save full chunk JSON (ALL fields present)
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    print(f"    {len(chunks)} chunks saved -> {json_out}")
    print(f"    Fields in each chunk: {sorted(chunks[0].keys())}")

    # Diagnostic: print first chunk to verify nothing is missing
    c0 = chunks[0]
    print(f"\n    -- first chunk preview --")
    print(f"    chunk_id          : {c0.get('chunk_id')}")
    print(f"    position_index    : {c0.get('position_index')}")
    print(f"    section_title     : {c0.get('section_title')}")
    print(f"    content_types     : {c0.get('content_types')}")
    print(f"    token_count       : {c0.get('token_count')}")
    print(f"    previous_chunk_id : {c0.get('previous_chunk_id')}")
    print(f"    next_chunk_id     : {c0.get('next_chunk_id')}")
    print(f"    related_keys      : {c0.get('related_keys')}")
    print(f"    title             : {str(c0.get('title',''))[:80]}")
    print(f"    summary           : {str(c0.get('summary',''))[:80]}")
    print(f"    keywords          : {c0.get('keywords')}")
    print(f"    key               : {c0.get('key')}")
    print(f"    text[:80]         : {str(c0.get('text',''))[:80].replace(chr(10),' ')}")
    print(f"    embed[:100]       : {build_embedding_text(c0)[:100]}")
    print()

    # 4. Index into ChromaDB
    print("  [4/4] Indexing into ChromaDB ...")
    n = index_chunks(chunks, filename, get_chroma_collection())

    return {"file": filename, "skipped": False, "chunks": len(chunks), "indexed": n}


def run_batch(html_dir: str, force: bool = False):
    """Process every .html file in html_dir."""
    html_files = sorted(
        f for f in os.listdir(html_dir) if f.lower().endswith(".html")
    )
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
    print(f"  DONE")
    print(f"  Processed : {len(html_files) - skipped} files")
    print(f"  Skipped   : {skipped} (already parsed)")
    print(f"  Chunks    : {total_chunks}")
    print(f"  Indexed   : {total_indexed} vectors")
    print(f"  ChromaDB  : {CHROMA_DIR}/")
    print(f"  JSONs     : {CHUNKS_DIR}/")
    print(f"  Markdown  : {PARSED_MD_DIR}/")
    print(f"{'=' * 64}\n")


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Parse, chunk, and index HTML news articles.")
    ap.add_argument("--dir",   default=RAW_HTML_DIR,
                    help=f"Folder with .html files (default: {RAW_HTML_DIR})")
    ap.add_argument("--file",  default=None,
                    help="Process a single HTML file")
    ap.add_argument("--force", action="store_true",
                    help="Re-parse files even when a chunk JSON already exists")
    args = ap.parse_args()

    if not CHATAI_API_KEY:
        raise SystemExit(
            "ERROR: CHATAI_API_KEY is not set.\n"
            "Add it to a .env file:  CHATAI_API_KEY=<your key>"
        )

    if args.file:
        process_file(args.file, force=args.force)
    else:
        os.makedirs(args.dir, exist_ok=True)
        run_batch(args.dir, force=args.force)