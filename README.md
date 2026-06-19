# RAGBench — Political News Cross-Document Retrieval System

Flask-based research dashboard for the thesis:
**"Evaluating RAG for Cross-Document Political News Retrieval"**

---

## Architecture

```
HTML Files (raw_html/)
    ↓  BeautifulSoup + Docling
Markdown (parsed_mds/)
    ↓  MDKeyChunker + ChatAI (meta-llama / e5-mistral)
Enriched Chunks (mdkey_chunks/*.json)
    ↓  ChromaDB + ChatAI embeddings
Vector Index (chroma_db/)

```

---

## Setup

### 1. Install dependencies (using virtual environment)

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your CHATAI_API_KEY
```

### 3. Organise your HTML files

Name each file following the convention so metadata is extracted automatically:

```
data/raw_html/TDC__left__news.html
data/raw_html/TDC__centre__news.html
data/raw_html/TDC__right__news.html
# Repeat for Other events TDCL --> (Trump's deportation campaign)
```

### 4. Run the app

Extract the raw html files from news source links including metadata:

```bash
python news_scrap.py
```

Run the news pipeline:

```bash
python news_pipeline.py
```

---


## Chunk Schema (from MDKeyChunker + thesis metadata)

```json
{
  "chunk_id": "a3f2b1c4",
  "text": "…",
  "section_title": "Coalition talks",
  "title": "Fiscal Dispute Overview",
  "summary": "Describes the breakdown over fiscal policy…",
  "keywords": ["coalition", "fiscal policy", "collapse"],
  "entities": [{"name": "Party A", "type": "ORG"}],
  "questions": ["Why did coalition talks fail?"],
  "key": "fiscal dispute",
  "token_count": 187,
  "start_line": 12,
  "end_line": 28,
  "source": "E1__left__commentary.html",
  "event_id": "E1",
  "political_perspective": "left",
  "doc_type": "commentary",
  "page": "3"
}
```

---

## MDKeyChunker + ChatAI configuration

MDKeyChunker is configured via environment variables in `.env`:

```
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://chat-ai.academiccloud.de/v1
LLM_API_KEY=<your key>
LLM_MODEL=meta-llama-3.1-8b-instruct
```

Embeddings use the same ChatAI base URL with `e5-mistral-7b-instruct`.
LLM Model `meta-llama-3.1-8b-instruct`.

---
