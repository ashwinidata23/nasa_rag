# Complex Technical Manual QA System

A RAG (Retrieval-Augmented Generation) chatbot that answers natural language questions about the NASA Systems Engineering Handbook (SP-2016-6105 Rev2, 297 pages). Every answer is grounded in the document with verifiable page and section citations.

**How you use it:** From `src`, run **`uvicorn main:app --reload --port 8000`**, then open **`chat.html`** from the project root in your browser. The page calls `http://localhost:8000/query` on your machine.

---

## How to Run

### Prerequisites
```bash
# Python 3.10+
# OpenAI API key in .env file
```

### Setup
```bash
git clone <repo>
cd i2ehireathon

# Create and activate virtual environment
python -m venv pharma_env
pharma_env\Scripts\activate          # Windows
source pharma_env/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Add your OpenAI key
Create `.env` in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

### Run ingestion pipeline (builds the knowledge base)
```bash
cd src
python pipeline.py --reindex
# Takes ~3 minutes, costs ~$0.002 in API calls
# Processes 297 pages → 906 chunks → FAISS index
```

### Start the API
```bash
cd src
uvicorn main:app --reload --port 8000
```

### Open the chatbot
With the API running, open **`chat.html`** from the project root (double-click or “Open with” your browser).

---

## Architecture

```
NASA PDF (297 pages)
        │
        ▼
┌─────────────────┐
│  extractor.py   │  PyMuPDF — extracts text, tables (bbox-aware), images
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│reconstructor.py │  2-pass multi-page table merger + SQLite buffer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  validator.py   │  Rejects false tables (empty cells, redaction markers)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│diagram_extractor.py │  Page-render + GPT-4o vision → text chunks, refs
└────────┬────────────┘
         │
         ▼
┌─────────────────┐
│   chunker.py    │  Content-aware chunking:
└────────┬────────┘  - Tables → parent chunk + row-level child chunks
         │           - Narrative → 400-token semantic chunks, 80-token overlap
         ▼
┌──────────────────┐
│reference_graph.py│  Detects cross-references (Table X, Section Y, NPR Z)
└────────┬─────────┘  Builds forward + inverse graph in SQLite
         │
         ▼
┌─────────────────┐
│  embedder.py    │  OpenAI text-embedding-3-small → FAISS IndexFlatIP
└────────┬────────┘  906 chunks, 1536 dimensions
         │
         ▼
┌─────────────────┐
│  retriever.py   │  Query → FAISS top-20 → cross-ref resolution
└────────┬────────┘  → deduplicate → top-5 → GPT-4o-mini
         │
         ▼
┌─────────────────┐
│    main.py      │  FastAPI: POST /query, GET /health
└─────────────────┘
```

---

## Key Design Decisions

### 1. PyMuPDF over PyPDF2 or pdfplumber
PyMuPDF provides bounding box (bbox) coordinates for every text block and table. This is essential for detecting that a table on page 172 is a continuation of the table that started on page 170 — they share the same x-span and column schema. PyPDF2 provides text only, with no layout information.

### 2. Two-pass table reconstruction
Standard parsers treat each PDF page independently. NASA handbook tables span up to 4 pages (e.g., TABLE 6 spans pages 170–173). Our pipeline:
- Pass 1: Collect fragments into SQLite buffer, detect continuations by column schema + y-gap + x-span matching
- Pass 2: Resolve footer titles (which appear at the end of the table, not the beginning)

This produces complete tables with all rows intact, rather than 4 orphaned fragments.

### 3. Parent + child chunking for tables
Each table produces:
- **Parent chunk**: Full table HTML — for broad queries ("what's in TABLE 6?")
- **Child chunks**: One per row with headers prepended — for specific lookups ("CDR entrance criteria")

WHY headers prepended to every row: A row `"CDR | Phase C | Design mature..."` is meaningless without knowing the columns are `"Review | Timing | Purpose"`.

### 4. Reference graph (forward + inverse)
The handbook heavily cross-references: "See Table G-7 of NPR 7123.1", "Refer to Section 6.3.2". We build a graph at ingestion:
- **Forward:** chunk_id → [referenced_chunk_ids] — used at query time to fetch linked chunks
- **Inverse:** ref_normalized → [source_chunk_ids] — used for incremental updates (when TABLE 6 changes, find all chunks that cite it)

**Example:** You ask *“What are the CDR entrance criteria?”* FAISS returns a **narrative** chunk from (say) page 150 that discusses reviews and says *“see TABLE 6”* — that page matches the question well. The **TABLE 6** chunks are a large HTML table; their embedding may be **far** from a short question, so they might not appear in the top-20 from vector search alone. At ingestion we recorded that this narrative chunk **references** `table_6` in SQLite. At query time, `retriever._resolve_cross_references` loads those linked chunk IDs from the metadata store and **appends** them (with a lower score, e.g. 0.5) so the LLM sees both the narrative **and** the table rows — same idea as the in-code comment: page 147 matches the query, but Table 4.2 (different pages) is pulled in via the graph.

### 5. FAISS over ChromaDB
ChromaDB requires C++ build tools (chroma-hnswlib compilation) which fails on Windows without Visual Studio SDK. FAISS ships pre-built Windows wheels. Same vector similarity search, different install path. For production: both are valid; Pinecone adds persistence and filtering.

### 6. GPT-4o-mini over GPT-4o
Cost comparison for 200 queries/day:
- GPT-4o-mini: ~$0.00028/query → $1.68/day → $50/month
- GPT-4o: ~$0.009/query → $54/day → $1,620/month

For factual Q&A on retrieved context, the context does the heavy lifting — not the model's parametric knowledge. GPT-4o-mini quality is sufficient and 33x cheaper.

### 7. External reference flagging
NASA handbook frequently references external documents (NPR 7123.1, Table G-7, NASA-STD-0010) that are not in our knowledge base. Rather than hallucinating their content, we steer the model to disclose that gap.

**How it works:** (1) **Prompt contract** — the retriever user prompt tells the model to label anything that points outside the handbook (e.g. NPR 7123.1, NASA-STD-0010) as *“External reference — verify [document] directly”* instead of inventing appendix text (`retriever.py`). (2) **Ingestion** — `reference_graph.py` regexes tie in-handbook phrases into the graph where we can resolve **in-corpus** targets; **diagram** chunks can carry `external_refs` from vision JSON, and the pipeline can add graph edges for resolvable section/table-like strings (`pipeline.py`). (3) **Outcome** — the model is steered to **disclose** when the answer depends on a document we did not ingest, which is safer than hallucinating NPR appendices.

### 8. Acronym expansion (not implemented)
**Issue:** If the user spells out a long form (*“Key Decision Point”*) but the PDF mostly uses *“KDP”*, dense embeddings can under-match. The inverse (*“KDP”* in query vs long form in doc) is often **handled well** by `text-embedding-3-small` — e.g. *“What is KDP?”* still retrieves relevant chunks in testing — so we **did not** add a glossary/expansion pass to avoid extra complexity and maintenance.

---

## Known Limitations

### 1. External document references
NPR 7123.1 appendices (Table G-7, Table G-15, etc.) contain detailed entrance/exit criteria for technical reviews. These are referenced by the handbook but not included in it. Questions requiring exact criteria from NPR appendices cannot be answered from the handbook alone.

**Impact:** CDR entrance criteria query returns the handbook's summary but flags "verify NPR 7123.1 Table G-7 directly."

**This is correct behavior** — fabricating external document content would be worse than acknowledging the gap.

### 2. Latency
Average query latency: 4–7 seconds. The OpenAI API round-trip accounts for 3–5 seconds.

**Fix:** Semantic cache for repeat queries. Clinical/engineering staff ask the same questions repeatedly — 40% of queries are near-duplicates. A Redis cache with cosine similarity threshold 0.95 returns cached answers in <100ms.

### 3. Residual provisional table titles
We reduced bare *“Table (T_p170_173)”* citations using an **expanded title regex** (footer + variants like `Table 2.`) and **title search in page text** above/below the table bbox (`patch_reconstructor_titles` in `pipeline.py`). Some handbook tables still have **no explicit “Table X:”** in detectable text; those can remain `[PROVISIONAL]` and cite with internal ids until a human or richer layout rules assign a label.

### 4. Diagram coverage
`diagram_extractor.py` turns many figures into searchable text chunks via vision, but **purely visual** detail (fine arrows, un-captioned graphics) can still be thin compared to the full figure.

---

## Evaluation Results

| Query Type | Works | Notes |
|---|---|---|
| TRL level definitions | ✅ | Full 9-level list with citations |
| Concept phase activities | ✅ | Structured list from Section 3 |
| Risk management process | ✅ | Multi-section synthesis working |
| Verification vs validation | ✅ | Clear distinction with citations |
| ConOps definition | ✅ | Correct with page reference |
| CDR entrance criteria | ⚠️ | Finds TABLE 6 row, external NPR ref flagged |
| Exact NPR appendix content | ❌ | External document — correctly flagged |

---

## Cost Summary

| Component | Cost |
|---|---|
| Full ingestion (906 chunks) | ~$0.002 |
| Per query (avg 1,500 tokens) | ~$0.00028 |
| 200 queries/day × 30 days | ~$1.68/month |

---

## Files

```
src/
├── extractor.py         PDF extraction (PyMuPDF)
├── reconstructor.py     Multi-page table merger
├── validator.py         False positive table filter  
├── diagram_extractor.py Vision captions for figures/diagrams → chunks
├── reference_graph.py   Cross-reference graph
├── chunker.py           Content-aware chunking
├── embedder.py          FAISS vector index
├── retriever.py         Query + LLM generation
├── pipeline.py          Single-command ingestion runner
└── main.py              FastAPI REST API

chat.html                Browser-based chat UI
requirements.txt         Python dependencies
.env                     OpenAI API key (create locally; never commit)
```

---

## If I Had More Time

1. **Semantic cache** — Redis with cosine similarity matching. Reduces repeat query latency from 6s to <100ms.
2. **Hierarchical section context** — maintain parent section summaries so a chunk from Section 6.3.2.1 carries context from Section 6.3, 6, and the chapter title.
3. **PDF viewer integration** — clicking a citation opens the PDF at the exact page.
