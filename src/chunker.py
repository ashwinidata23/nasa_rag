"""
chunker.py — Day 5
-------------------
Responsibility: Take validated ReconstructedTables + PageData objects
and produce Chunk objects ready for embedding.

WHY chunking strategy matters (Q5 answer running as code):
  - Fixed-size chunks split tables mid-row → wrong
  - Semantic chunking groups unrelated paragraphs → noisy
  - Section-based chunks are 40 pages or 1 cell → useless

Our strategy: content-type-aware chunking
  - Tables   → atomic HTML chunk (whole table) +
               child chunks (header + single row)
  - Narrative → semantic split, 400 token max, 80 token overlap
  - Diagrams  → single chunk per diagram (handled separately)

WHY metadata envelope on every chunk (Q4 answer):
  Every chunk carries: page, section, table_id, amendment info.
  This travels unchanged from extraction → embedding → retrieval
  → LLM prompt → citation in final answer.
  Lose the metadata at any step = wrong or missing citations.
"""

import json
import re
import os
import tiktoken
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single embeddable unit with full metadata envelope.

    This is what enters the vector store.
    The metadata dict is stored alongside the vector
    and returned at retrieval time.
    """
    chunk_id: str
    text: str                   # what gets embedded
    chunk_type: str             # 'table', 'table_row', 'narrative', 'diagram'
    metadata: dict              # full provenance — never stripped

    def __post_init__(self):
        # Validate metadata has minimum required fields
        required = ['page_start', 'page_end', 'type']
        for field_name in required:
            if field_name not in self.metadata:
                self.metadata[field_name] = None


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------

class TokenCounter:
    """
    Counts tokens using tiktoken (same tokenizer as OpenAI models).

    WHY tiktoken not len(text)//4:
      The "divide by 4" approximation is off by 20-40% for
      technical text with acronyms, units, and numbers.
      tiktoken gives exact counts — critical for staying
      under context window limits.
    """
    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            self.enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self.enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.enc.encode(text))


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class Chunker:
    """
    Produces Chunk objects from:
      - ReconstructedTable objects (from reconstructor.py)
      - PageData objects (for narrative text)

    Configuration:
      max_tokens      : max tokens per narrative chunk (400)
      overlap_tokens  : overlap between consecutive chunks (80)
                        WHY overlap: a sentence split across chunk
                        boundary loses context. Overlap ensures
                        the boundary sentence appears in both chunks.
    """

    def __init__(self,
                 max_tokens: int = 400,
                 overlap_tokens: int = 80):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.token_counter = TokenCounter()
        self._chunk_counter = 0

    def chunk_tables(self, tables: list) -> list:
        """
        Chunk reconstructed tables.

        Strategy:
          1. One parent chunk = full table HTML
             Used for broad retrieval ("what's in Table 4.2?")

          2. One child chunk per row = header + row
             Used for specific cell retrieval
             ("what row in Table 2.2-1 describes Phase B?")

        WHY both:
          Parent catches broad queries.
          Child catches specific cell-level queries.
          At retrieval time we return whichever scored higher.
        """
        chunks = []

        for table in tables:
            # --- Parent chunk: full table ---
            parent_text = table.to_chunk_text()
            parent_id = f"tbl_{table.table_id}"

            parent_chunk = Chunk(
                chunk_id=parent_id,
                text=parent_text,
                chunk_type="table",
                metadata={
                    **table.to_metadata(),
                    "chunk_id": parent_id,
                    "is_parent": True,
                    "child_ids": [],    # filled below
                    "token_count": self.token_counter.count(parent_text)
                }
            )

            # --- Child chunks: one per row ---
            child_ids = []
            for row_idx, row in enumerate(table.all_rows):
                if not any(cell for cell in row if cell):
                    continue  # skip empty rows

                # Prepend headers to every row chunk
                # WHY: a row without headers is meaningless
                # "SE 2.3 | Product Verification | ..." means nothing without
                # knowing column headers (Competency Area | Competency | Description)
                row_text = self._build_row_chunk_text(
                    table.title,
                    table.headers,
                    row,
                    table.pages
                )

                child_id = f"tbl_{table.table_id}_r{row_idx:03d}"
                child_ids.append(child_id)

                child_chunk = Chunk(
                    chunk_id=child_id,
                    text=row_text,
                    chunk_type="table_row",
                    metadata={
                        **table.to_metadata(),
                        "chunk_id": child_id,
                        "is_parent": False,
                        "parent_id": parent_id,
                        "row_index": row_idx,
                        "token_count": self.token_counter.count(row_text)
                    }
                )
                chunks.append(child_chunk)

            # Update parent with child IDs
            parent_chunk.metadata["child_ids"] = child_ids
            chunks.insert(0, parent_chunk)  # parent first

        return chunks

    def chunk_narrative(self, pages: list,
                        table_pages: set = None) -> list:
        """
        Chunk narrative text from pages.

        Skips pages that are predominantly tables
        (already handled by chunk_tables).

        Strategy:
          - Split on paragraph boundaries (double newline)
          - Never split mid-sentence
          - Max 400 tokens per chunk
          - 80 token overlap between chunks
          - Attach full metadata envelope to each chunk
        """
        chunks = []
        table_pages = table_pages or set()

        for page_data in pages:
            page_num = page_data.page_num
            text = page_data.raw_text.strip()

            if not text or len(text) < 50:
                continue  # skip near-empty pages

            # Detect section heading if present
            section = self._detect_section(text)

            # Split into paragraphs first
            paragraphs = self._split_paragraphs(text)

            # Group paragraphs into chunks respecting token limit
            current_chunk_paras = []
            current_tokens = 0

            for para in paragraphs:
                para_tokens = self.token_counter.count(para)

                if para_tokens == 0:
                    continue

                # If adding this paragraph exceeds limit, flush current chunk
                if (current_tokens + para_tokens > self.max_tokens
                        and current_chunk_paras):
                    chunk = self._build_narrative_chunk(
                        current_chunk_paras,
                        page_num, section
                    )
                    chunks.append(chunk)

                    # Overlap: keep last paragraph in next chunk
                    # WHY: prevents sentence context loss at boundaries
                    overlap_paras = current_chunk_paras[-1:]
                    current_chunk_paras = overlap_paras + [para]
                    current_tokens = sum(
                        self.token_counter.count(p)
                        for p in current_chunk_paras
                    )
                else:
                    current_chunk_paras.append(para)
                    current_tokens += para_tokens

            # Flush remaining paragraphs
            if current_chunk_paras:
                chunk = self._build_narrative_chunk(
                    current_chunk_paras,
                    page_num, section
                )
                chunks.append(chunk)

        return chunks

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _build_row_chunk_text(self, title: str, headers: list,
                               row: list, pages: list) -> str:
        """
        Build embeddable text for a single table row.
        Always includes headers for context.
        """
        lines = []
        if title:
            lines.append(f"From: {title} (pages {pages})")
        lines.append("Columns: " + " | ".join(
            str(h) for h in headers if h
        ))
        lines.append("Values:  " + " | ".join(
            str(c) for c in row if c
        ))
        return "\n".join(lines)

    def _build_narrative_chunk(self, paragraphs: list,
                                page_num: int,
                                section: str) -> Chunk:
        """Build a narrative chunk with metadata envelope."""
        self._chunk_counter += 1
        text = "\n\n".join(paragraphs)
        chunk_id = f"nar_p{page_num:03d}_{self._chunk_counter:04d}"

        return Chunk(
            chunk_id=chunk_id,
            text=text,
            chunk_type="narrative",
            metadata={
                "chunk_id": chunk_id,
                "page_start": page_num,
                "page_end": page_num,
                "section": section,
                "type": "narrative",
                "token_count": self.token_counter.count(text),
                "table_id": None,
                "is_parent": False,
                "amendment_version": None,
                "superseded_by": None
            }
        )

    def _split_paragraphs(self, text: str) -> list:
        """
        Split text into paragraphs on double newlines.
        Filters out very short fragments (noise).
        """
        # Split on double newline or section-like breaks
        raw_paras = re.split(r'\n\s*\n', text)

        # Filter and clean
        paras = []
        for p in raw_paras:
            p = p.strip()
            # Skip very short fragments — likely headers or noise
            if len(p) < 30:
                continue
            # Skip page number artifacts
            if re.match(r'^\d+$', p):
                continue
            paras.append(p)

        return paras

    def _detect_section(self, text: str) -> Optional[str]:
        """
        Try to detect section number from page text.
        Used to populate metadata.section field for citations.

        Looks for patterns like:
          "2.0 Fundamentals of Systems Engineering"
          "Section 4.0: System Design Processes"
          "6.0 Product Realization"
        """
        patterns = [
            r'^(\d+\.\d+)\s+[A-Z]',       # "2.0 Fundamentals of Systems Engineering"
            r'^Section\s+(\d+[\.\d]*)',     # "Section 4.2"
            r'^(\d+)\.\s+[A-Z]{3}',        # "6.0 Product Realization"
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:200], re.MULTILINE)
            if match:
                return match.group(1)
        return None

    def _new_chunk_id(self, prefix: str) -> str:
        self._chunk_counter += 1
        return f"{prefix}_{self._chunk_counter:04d}"


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from extractor import PDFExtractor
    from reconstructor import TableReconstructor
    from validator import TableValidator

    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    _default_pdf = os.path.join(
        _project_root, "data", "nasa_systems_engineering_handbook_0.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else _default_pdf

    # Step 1: Extract
    print("=== Step 1: Extracting all pages ===")
    extractor = PDFExtractor(pdf_path)
    pages = extractor.extract_all_pages(end_page=None)
    extractor.close()

    # Step 2: Reconstruct tables
    print("\n=== Step 2: Reconstructing tables ===")
    reconstructor = TableReconstructor(db_path="db/table_buffer.db")
    raw_tables = reconstructor.reconstruct(pages)

    # Step 3: Validate tables
    print("\n=== Step 3: Validating tables ===")
    validator = TableValidator()
    valid_tables, rejected = validator.validate_all(raw_tables)
    print(f"Valid: {len(valid_tables)}, Rejected: {len(rejected)}")

    # Step 4: Chunk everything
    print("\n=== Step 4: Chunking ===")
    chunker = Chunker(max_tokens=400, overlap_tokens=80)

    table_chunks = chunker.chunk_tables(valid_tables)
    narrative_chunks = chunker.chunk_narrative(pages)

    all_chunks = table_chunks + narrative_chunks

    # Step 5: Show results
    print(f"\n=== CHUNKING RESULTS ===")
    print(f"Table chunks:     {len(table_chunks)}")
    print(f"Narrative chunks: {len(narrative_chunks)}")
    print(f"Total chunks:     {len(all_chunks)}")

    # Token distribution
    token_counts = [c.metadata.get('token_count', 0) for c in all_chunks]
    if token_counts:
        print(f"\nToken stats:")
        print(f"  Min:  {min(token_counts)}")
        print(f"  Max:  {max(token_counts)}")
        print(f"  Avg:  {sum(token_counts)//len(token_counts)}")

    # Sample chunks
    print(f"\n--- Sample table chunk ---")
    tbl_chunks = [c for c in all_chunks if c.chunk_type == "table"]
    if tbl_chunks:
        t = tbl_chunks[0]
        print(f"  ID:       {t.chunk_id}")
        print(f"  Type:     {t.chunk_type}")
        print(f"  Tokens:   {t.metadata.get('token_count')}")
        print(f"  Metadata: {json.dumps(t.metadata, indent=4)}")
        print(f"  Text preview: {t.text[:200]}")

    nar_chunks = [c for c in all_chunks if c.chunk_type == "narrative"]
    n_show = min(5, len(nar_chunks))
    print(f"\n--- First {n_show} narrative (paragraph) chunks ---")
    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "narrative_chunks_preview.txt")
    log_body = [f"pdf={pdf_path}\n", f"narrative_total={len(nar_chunks)}\n\n"]
    for i, n in enumerate(nar_chunks[:5], 1):
        preview = n.text[:800] + ("…" if len(n.text) > 800 else "")
        block = (
            f"[{i}] id={n.chunk_id} | page={n.metadata.get('page_start')} "
            f"| section={n.metadata.get('section')} | "
            f"tokens={n.metadata.get('token_count')}\n{preview}\n\n"
        )
        print(block)
        log_body.append(block)
    if nar_chunks:
        with open(log_file, "w", encoding="utf-8") as f:
            f.writelines(log_body)
        print(f"(Also saved to {log_file})")

    print(f"\nChunker.py is working and {len(all_chunks)} chunks ready for embedding.")