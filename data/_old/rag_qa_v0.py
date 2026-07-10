"""
rag_qa_generator.py
===================
Generate a gold-standard RAG Q&A JSON from all chunk JSONs in
data/mdkey_chunks/  using the existing ChromaDB index built by
news_pipeline.py.

Pipeline
--------
1. Load all chunk JSONs from data/mdkey_chunks/
2. Extract every LLM-generated question from each chunk
3. For each question, retrieve top-k chunks from ChromaDB
   (with known RAG flaws mitigated — see FLAW MAP below)
4. Generate an answer using a configurable LLM (ChatAI or Anthropic)
5. Save results as data/rag_qa_output.json

FLAW MAP  (problems in naive RAG + how we fix each one)
--------------------------------------------------------
FLAW 1 – Embedding only bare question text
  → We embed a rich query:  "query: <question> [event:<id> doc_type:<t>]"
    using the e5-mistral asymmetric prefix so the query vector lands in
    the same space as the "passage: ..." document vectors.

FLAW 2 – Retrieving without metadata filters
  → We always filter by event_id when the question comes from a chunk
    that has one, so answers can't bleed across unrelated news events.

FLAW 3 – Fixed top-k regardless of question complexity
  → Single-hop questions (factual / specific entity) use k=3.
    Multi-hop / analytical questions use k=6.

FLAW 4 – No deduplication of retrieved passages
  → After retrieval we deduplicate by chunk_id before sending to the LLM.

FLAW 5 – Context window overflow / irrelevant padding
  → We trim each retrieved passage to MAX_PASSAGE_TOKENS tokens and
    include at most MAX_CONTEXT_CHUNKS passages in the prompt.

FLAW 6 – Lost neighbouring context (chunking seam problem)
  → For every retrieved chunk we also include its previous_chunk_id and
    next_chunk_id neighbours from the in-memory chunk store.

FLAW 7 – Annotator == retrieval evaluator (evaluation bias)
  → The answer-generation model is configured separately from the
    embedding / retrieval model via QA_LLM_* env vars.  Defaults to
    the same ChatAI endpoint but a different model; swap to Anthropic
    by setting QA_LLM_PROVIDER=anthropic.

FLAW 8 – Empty / unenriched chunks injected as questions
  → Chunks without LLM-generated questions, titles, or summaries are
    skipped (they are raw report text with no semantic enrichment).

FLAW 9 – Questions from a chunk retrieved without its metadata context
  → The answer prompt always includes the source chunk's own title,
    summary, event_id, doc_type, and political_perspective so the LLM
    has grounding even if retrieval partially fails.

FLAW 10 – No question deduplication across chunks of the same document
  → We deduplicate questions at the event + normalised-text level before
    running retrieval, so identical phrasings from neighbouring chunks
    don't generate duplicate records.

.env keys
---------
# Shared (embedding + retrieval — same as news_pipeline.py)
CHATAI_API_KEY=...
CHATAI_BASE_URL=https://chat-ai.academiccloud.de/v1
CHATAI_LLM_MODEL=meta-llama-3.1-70b-instruct
CHATAI_EMBED_MODEL=e5-mistral-7b-instruct

# Answer-generation model  (default: same ChatAI endpoint, llama-3.1-70b)
QA_LLM_PROVIDER=chatai          # or: anthropic
QA_LLM_MODEL=gpt-oss-120b
# Only needed when QA_LLM_PROVIDER=anthropic:
ANTHROPIC_API_KEY=...
"""

import os
import re
import json
import time
import uuid
import hashlib
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Directory layout  (mirrors news_pipeline.py)
# ---------------------------------------------------------------------------
DATA_DIR       = "data"
CHUNKS_DIR     = os.path.join(DATA_DIR, "mdkey_chunks")
CHROMA_DIR     = os.path.join(DATA_DIR, "chroma_db")
OUTPUT_PATH    = os.path.join(DATA_DIR, "rag_qa_output.json")
CHROMA_COLLECTION = "political_news"

# ---------------------------------------------------------------------------
# ChatAI / embedding settings  (read from .env)
# ---------------------------------------------------------------------------
CHATAI_BASE_URL   = os.getenv("CHATAI_BASE_URL",    "https://chat-ai.academiccloud.de/v1")
CHATAI_API_KEY    = os.getenv("CHATAI_API_KEY",     "")
CHATAI_LLM_MODEL  = os.getenv("CHATAI_LLM_MODEL",  "meta-llama-3.1-8b-instruct")
CHATAI_EMB_MODEL  = os.getenv("CHATAI_EMBED_MODEL", "e5-mistral-7b-instruct")

# Answer-generation model config
QA_LLM_PROVIDER   = os.getenv("QA_LLM_PROVIDER",  "chatai")   # chatai | anthropic
QA_LLM_MODEL      = os.getenv("QA_LLM_MODEL",      "gpt-oss-120b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# RAG retrieval knobs
TOP_K_SIMPLE      = 3    # factual / single-hop questions
TOP_K_COMPLEX     = 6    # analytical / multi-entity questions
MAX_PASSAGE_TOKENS = 300 # hard cap per retrieved passage
MAX_CONTEXT_CHUNKS = 6   # max passages sent to LLM

# ---------------------------------------------------------------------------
# Heuristic: is this question complex / multi-hop?
# ---------------------------------------------------------------------------
_COMPLEX_SIGNALS = re.compile(
    r"\b(compar|contrast|differ|both|all|across|between|relation|impact|"
    r"analyz|evaluat|assess|why|how does|to what extent|explain|discuss)\b",
    re.I,
)

def _is_complex(question: str) -> bool:
    return bool(_COMPLEX_SIGNALS.search(question))

# ---------------------------------------------------------------------------
# Token-budget trimmer  (rough: 1 token ≈ 4 chars)
# ---------------------------------------------------------------------------
def _trim(text: str, max_tokens: int) -> str:
    limit = max_tokens * 4
    return text[:limit] + ("…" if len(text) > limit else "")

# ---------------------------------------------------------------------------
# Stable question ID
# ---------------------------------------------------------------------------
def _qid(event_id: str, question: str) -> str:
    h = hashlib.md5(f"{event_id}|{question}".encode()).hexdigest()[:8]
    return f"QA_{event_id}_{h}"

# ===========================================================================
# STEP 1 — Load all chunk JSONs from mdkey_chunks/
# ===========================================================================

def load_all_chunks(chunks_dir: str) -> tuple[list[dict], dict[str, dict]]:
    """
    Returns:
        chunks      – flat list of all chunk dicts (with corpus metadata)
        chunk_index – dict keyed by chunk_id for O(1) neighbour lookup
    """
    all_chunks: list[dict] = []
    chunk_index: dict[str, dict] = {}

    json_files = sorted(Path(chunks_dir).glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {chunks_dir!r}")

    for jf in json_files:
        with open(jf, encoding="utf-8") as fh:
            data = json.load(fh)

        # Support both flat list  [ {...}, ... ]
        # and wrapped object      { "metadata": {...}, "chunks": [...] }
        if isinstance(data, list):
            raw_chunks = data
            file_meta  = {}
        elif isinstance(data, dict) and "chunks" in data:
            raw_chunks = data["chunks"]
            file_meta  = data.get("metadata", {})
        else:
            print(f"  [WARN] Unrecognised structure in {jf.name} — skipping")
            continue

        # Derive corpus axes from filename  (DC1__Left__News)
        stem  = jf.stem
        parts = stem.split("__")
        event_id    = parts[0] if len(parts) > 0 else file_meta.get("event_id", "unknown")
        perspective = parts[1] if len(parts) > 1 else file_meta.get("political_leaning", "unknown")
        doc_type    = parts[2] if len(parts) > 2 else file_meta.get("news_type", "unknown")

        for c in raw_chunks:
            c.setdefault("source",                jf.name)
            c.setdefault("event_id",              event_id)
            c.setdefault("political_perspective",  perspective)
            c.setdefault("doc_type",              doc_type)
            all_chunks.append(c)
            cid = c.get("chunk_id")
            if cid:
                chunk_index[cid] = c

    print(f"[load] {len(json_files)} JSON file(s) → {len(all_chunks)} chunks total")
    return all_chunks, chunk_index

# ===========================================================================
# STEP 2 — Extract & deduplicate questions
# ===========================================================================

def extract_questions(all_chunks: list[dict]) -> list[dict]:
    """
    Pull every LLM-generated question from chunk metadata.
    Returns list of dicts:
      { question, question_id, event_id, doc_type, political_perspective,
        source_chunk_id, required_documents, required_doc_types,
        required_political_perspectives }

    Mitigations applied:
      - FLAW 8:  skip chunks with no enrichment (no title/summary/questions)
      - FLAW 10: deduplicate by (event_id, normalised question text)
    """
    seen: set[str] = set()
    records: list[dict] = []

    for chunk in all_chunks:
        questions = chunk.get("questions") or []
        if not questions:
            continue  # FLAW 8: skip unenriched chunks

        event_id    = chunk.get("event_id", "unknown")
        doc_type    = chunk.get("doc_type", "unknown")
        perspective = chunk.get("political_perspective", "unknown")
        src_cid     = chunk.get("chunk_id", "")
        source      = chunk.get("source", "")

        for q in questions:
            if not q or not q.strip():
                continue
            # FLAW 10: deduplicate
            
            normalized_q = re.sub(r"\s+", " ", q).strip().lower()
            norm_key = f"{event_id}|{normalized_q}"
            if norm_key in seen:
                continue
            seen.add(norm_key)

            records.append({
                "question_id":                      _qid(event_id, q),
                "question":                         q.strip(),
                "event_id":                         event_id,
                "required_documents":               [source],
                "required_doc_types":               [doc_type],
                "required_political_perspectives":  [perspective],
                "_source_chunk_id":                 src_cid,   # internal, stripped at output
                "_chunk_title":                     chunk.get("title", ""),
                "_chunk_summary":                   chunk.get("summary", ""),
                "_chunk_text":                      chunk.get("text", ""),
            })

    print(f"[questions] {len(records)} unique questions extracted")
    return records

# ===========================================================================
# STEP 3 — ChromaDB retrieval  (with RAG flaw mitigations)
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


def retrieve_context(
    question: str,
    event_id: str,
    chunk_index: dict[str, dict],
    collection,
) -> list[dict]:
    """
    Retrieve relevant chunks for a question.

    FLAW 1  – Rich query prefix (e5-mistral asymmetric embedding)
    FLAW 2  – Metadata filter on event_id
    FLAW 3  – Adaptive top-k
    FLAW 4  – Deduplication by chunk_id
    FLAW 5  – Passage trimming
    FLAW 6  – Neighbour injection
    """
    top_k = TOP_K_COMPLEX if _is_complex(question) else TOP_K_SIMPLE

    # FLAW 1: prefix the query the way e5-mistral expects asymmetric queries
    rich_query = f"query: {question}"

    # FLAW 2: filter to same event so we don't bleed across stories
    where_filter = {"event_id": {"$eq": event_id}} if event_id != "unknown" else None

    try:
        result = collection.query(
            query_texts = [rich_query],
            n_results   = top_k,
            where       = where_filter,
            include     = ["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        print(f"  [WARN] ChromaDB query failed: {exc}")
        return []

    retrieved: list[dict] = []
    seen_ids: set[str] = set()

    metadatas = result.get("metadatas", [[]])[0]
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for meta, doc, dist in zip(metadatas, documents, distances):
        cid = meta.get("chunk_id", "")

        # FLAW 4: deduplicate
        if cid and cid in seen_ids:
            continue
        if cid:
            seen_ids.add(cid)

        # FLAW 5: trim passage
        passage = _trim(doc, MAX_PASSAGE_TOKENS)

        retrieved.append({
            "chunk_id":              cid,
            "title":                 meta.get("title", ""),
            "political_perspective": meta.get("political_perspective", ""),
            "doc_type":              meta.get("doc_type", ""),
            "distance":              round(float(dist), 4),
            "text":                  passage,
        })

        # FLAW 6: inject previous/next neighbours
        for neighbour_key in ("previous_chunk_id", "next_chunk_id"):
            nb_id = meta.get(neighbour_key, "")
            if nb_id and nb_id not in seen_ids and nb_id in chunk_index:
                seen_ids.add(nb_id)
                nb = chunk_index[nb_id]
                retrieved.append({
                    "chunk_id":              nb_id,
                    "title":                 nb.get("title", ""),
                    "political_perspective":  nb.get("political_perspective", ""),
                    "doc_type":              nb.get("doc_type", ""),
                    "distance":              round(float(dist) + 0.001, 4),  # tag as neighbour
                    "text":                  _trim(nb.get("text", ""), MAX_PASSAGE_TOKENS),
                })

    # FLAW 5: cap total context chunks
    retrieved = retrieved[:MAX_CONTEXT_CHUNKS]
    return retrieved

# ===========================================================================
# STEP 4 — Answer generation  (configurable LLM)
# ===========================================================================

def _build_prompt(question: str, source_chunk: dict, context_chunks: list[dict]) -> str:
    """
    FLAW 9: always include the source chunk's own enriched metadata so the
    LLM has grounding even when retrieval partially fails.
    """
    lines = [
        "You are a factual news Q&A assistant. Answer ONLY from the provided context.",
        "If the context does not contain enough information, say 'Insufficient context'.",
        "",
        f"Question: {question}",
        "",
        "=== Source document context ===",
        f"Event    : {source_chunk.get('event_id', '')}",
        f"Doc type : {source_chunk.get('doc_type', '')}",
        f"Perspective: {source_chunk.get('political_perspective', '')}",
        f"Title    : {source_chunk.get('title', '')}",
        f"Summary  : {source_chunk.get('summary', '')}",
        "",
        "=== Retrieved passages ===",
    ]

    if not context_chunks:
        lines.append("[No passages retrieved — using source context only]")
        lines.append(_trim(source_chunk.get("text", ""), MAX_PASSAGE_TOKENS))
    else:
        for i, c in enumerate(context_chunks, 1):
            lines.append(
                f"[{i}] ({c.get('doc_type','')} | {c.get('political_perspective','')} "
                f"| dist={c.get('distance','')}) {c.get('title','')}"
            )
            lines.append(c.get("text", ""))
            lines.append("")

    lines += [
        "=== Answer ===",
        "Provide a concise, factual answer (2-4 sentences). Cite passage numbers where relevant.",
    ]
    return "\n".join(lines)


def generate_answer_chatai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=CHATAI_BASE_URL, api_key=CHATAI_API_KEY)
    resp = client.chat.completions.create(
        model    = QA_LLM_MODEL,
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 512,
        temperature = 0.1,
    )
    return resp.choices[0].message.content.strip()


def generate_answer_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model      = QA_LLM_MODEL,
        max_tokens = 512,
        messages   = [{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def generate_answer(prompt: str) -> str:
    if QA_LLM_PROVIDER.lower() == "anthropic":
        return generate_answer_anthropic(prompt)
    return generate_answer_chatai(prompt)

# ===========================================================================
# STEP 5 — Full pipeline
# ===========================================================================

def run(
    chunks_dir: str  = CHUNKS_DIR,
    output_path: str = OUTPUT_PATH,
    dry_run: bool    = False,
    limit: Optional[int] = None,
):
    print(f"\n{'='*64}")
    print(f"  RAG Q&A Generator")
    print(f"  Chunks dir   : {chunks_dir}")
    print(f"  ChromaDB     : {CHROMA_DIR}")
    print(f"  QA provider  : {QA_LLM_PROVIDER}  model={QA_LLM_MODEL}")
    print(f"  Embed model  : {CHATAI_EMB_MODEL}")
    print(f"  Output       : {output_path}")
    if dry_run:
        print("  DRY RUN — no LLM calls will be made")
    print(f"{'='*64}\n")

    # 1. Load chunks
    all_chunks, chunk_index = load_all_chunks(chunks_dir)

    # 2. Extract questions
    question_records = extract_questions(all_chunks)
    if limit:
        question_records = question_records[:limit]
        print(f"[limit] processing first {limit} questions")

    # 3. Connect to ChromaDB
    print("[chroma] connecting to persistent collection …")
    collection = get_chroma_collection()
    print(f"[chroma] collection '{CHROMA_COLLECTION}' has "
          f"{collection.count()} vectors\n")

    # 4. Retrieve + generate
    results: list[dict] = []
    total = len(question_records)

    for idx, rec in enumerate(question_records, 1):
        question    = rec["question"]
        event_id    = rec["event_id"]
        src_cid     = rec["_source_chunk_id"]

        print(f"  [{idx:>3}/{total}] {event_id} | {question[:80]}")

        # Build source chunk metadata dict for the prompt (FLAW 9)
        source_chunk = {
            "event_id":              event_id,
            "doc_type":              rec["required_doc_types"][0] if rec["required_doc_types"] else "",
            "political_perspective": rec["required_political_perspectives"][0] if rec["required_political_perspectives"] else "",
            "title":                 rec["_chunk_title"],
            "summary":               rec["_chunk_summary"],
            "text":                  rec["_chunk_text"],
        }

        # Retrieve context (FLAW 1-6)
        context_chunks = retrieve_context(question, event_id, chunk_index, collection)

        if dry_run:
            answer = "[DRY RUN — no LLM call]"
        else:
            prompt = _build_prompt(question, source_chunk, context_chunks)
            try:
                answer = generate_answer(prompt)
            except Exception as exc:
                answer = f"[ERROR: {exc}]"
                print(f"    [WARN] LLM error: {exc}")
            time.sleep(0.3)   # gentle rate-limit buffer

        # Build output record (strip internal _ keys)
        results.append({
            "question_id":                     rec["question_id"],
            "event_id":                        event_id,
            "question":                        question,
            "required_documents":              rec["required_documents"],
            "required_doc_types":              rec["required_doc_types"],
            "required_political_perspectives": rec["required_political_perspectives"],
            "retrieved_chunk_ids":             [c["chunk_id"] for c in context_chunks],
            "rag_answer":                      answer,
            "qa_model":                        QA_LLM_MODEL,
            "qa_provider":                     QA_LLM_PROVIDER,
            "embed_model":                     CHATAI_EMB_MODEL,
        })

    # 5. Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"\n{'='*64}")
    print(f"  Done.  {len(results)} Q&A records saved → {output_path}")
    print(f"{'='*64}\n")
    return results


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate RAG Q&A from mdkey_chunks/")
    ap.add_argument("--chunks-dir",  default=CHUNKS_DIR,   help="Folder with chunk JSONs")
    ap.add_argument("--output",      default=OUTPUT_PATH,  help="Output JSON path")
    ap.add_argument("--limit",       type=int, default=None, help="Process only first N questions (for testing)")
    ap.add_argument("--dry-run",     action="store_true",  help="Skip LLM calls (retrieval only)")
    args = ap.parse_args()

    if not CHATAI_API_KEY:
        raise SystemExit(
            "ERROR: CHATAI_API_KEY is not set.\n"
            "Add it to your .env file:  CHATAI_API_KEY=<your key>"
        )
    if QA_LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise SystemExit(
            "ERROR: QA_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set."
        )

    run(
        chunks_dir  = args.chunks_dir,
        output_path = args.output,
        dry_run     = args.dry_run,
        limit       = args.limit,
    )
