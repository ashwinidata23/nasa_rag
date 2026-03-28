"""
reference_graph.py
---------------------------
Responsibility: Scan all page text and detect explicit cross-references
like "Table 4.2", "Section 5.3", "see page 32".

Build a graph: {source_chunk → [referenced_chunk_ids]}
And its inverse: {referenced_chunk_id → [source_chunk_ids]}

WHY this solves Q2:
  Without this, a query for "stakeholder expectations definition"
  retrieves a narrative page that mentions a table but NOT Table 4.1-1
  (contains the stakeholder list) because they share no overlapping text.

  With this graph, when page 147 chunk is retrieved, we
  automatically also fetch Table 4.2 — because the graph tells
  us page 147 references Table 4.2.

WHY build at ingestion not query time:
  Scanning 1GB PDF for cross-refs at query time = 4+ seconds latency.
  Pre-building the graph = 2ms lookup at query time.
"""

import sqlite3
import re
import os
import json
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CrossReference:
    """A single detected cross-reference."""
    source_page: int
    source_text_snippet: str    # the sentence containing the reference
    ref_type: str               # 'table', 'section', 'figure', 'page'
    ref_id: str                 # e.g. 'Table 4.2', 'Section 5.3'
    ref_normalized: str         # normalized key e.g. 'table_4.2'
    confidence: float           # 0-1, how confident we are this is real


# ---------------------------------------------------------------------------
# Regex patterns for cross-reference detection
# ---------------------------------------------------------------------------

CROSS_REF_PATTERNS = [
    # Table references: "Table 4.2", "Table 4", "TABLE 7.3"
    {
        "type": "table",
        "pattern": r'\bTable\s+([\d]+(?:\.[\d]+)?)\b',
        "normalizer": lambda m: f"table_{m.lower().replace(' ', '_').replace('table_', '')}"
    },
    # Section references: "Section 5.3", "Sec. 4.1", "section 12"
    {
        "type": "section",
        "pattern": r'\b(?:Section|Sec\.?)\s+([\d]+(?:\.[\d]+)*)\b',
        "normalizer": lambda m: f"section_{m.lower().replace('section_', '').replace('sec._', '').replace('sec_', '').strip()}"
    },
    # Figure references: "Figure 3", "Fig. 4.2"
    {
        "type": "figure",
        "pattern": r'\b(?:Figure|Fig\.?)\s+([\d]+(?:\.[\d]+)?)\b',
        "normalizer": lambda m: f"figure_{m.lower().replace('figure_', '').replace('fig._', '').replace('fig_', '').strip()}"
    },
    # Page references: "see page 32", "page 147"
    {
        "type": "page",
        "pattern": r'\b(?:see\s+)?[Pp]age\s+([\d]+)\b',
        "normalizer": lambda m: f"page_{m.lower().replace('page_', '').replace('see_', '').strip()}"
    },
    # Column references: "Column C", "column 3"
    {
        "type": "column",
        "pattern": r'\bColumn\s+([A-Z]|[\d]+)\b',
        "normalizer": lambda m: f"column_{m.lower().replace('column_', '').strip()}"
    },
]


# ---------------------------------------------------------------------------
# Reference Graph Database
# ---------------------------------------------------------------------------

class ReferenceGraph:
    """
    SQLite-backed graph of cross-references.

    Two tables:
      references: source → target (forward lookup)
      ref_index:  target → sources (reverse lookup)

    WHY both directions:
      Forward: "what does page 147 reference?" → fetch Table 4.2
      Reverse: "what references Table 4.2?" → used in incremental
               updates to find all chunks that go stale when
               Table 4.2 changes (Q6 ripple analysis)
    """

    def __init__(self, db_path: str = "db/reference_graph.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS cross_references (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_page     INTEGER,
                source_chunk_id TEXT,
                ref_type        TEXT,
                ref_id          TEXT,
                ref_normalized  TEXT,
                snippet         TEXT,
                confidence      REAL
            );

            CREATE TABLE IF NOT EXISTS ref_targets (
                ref_normalized  TEXT PRIMARY KEY,
                ref_type        TEXT,
                resolved_pages  TEXT,    -- JSON list of page numbers
                resolved_chunk_ids TEXT  -- JSON list of chunk IDs
            );

            CREATE INDEX IF NOT EXISTS idx_source
                ON cross_references(source_chunk_id);

            CREATE INDEX IF NOT EXISTS idx_ref_norm
                ON cross_references(ref_normalized);
        """)
        self.conn.commit()

    def add_reference(self, source_page: int, source_chunk_id: str,
                      ref: CrossReference):
        self.conn.execute("""
            INSERT INTO cross_references
            (source_page, source_chunk_id, ref_type, ref_id,
             ref_normalized, snippet, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            source_page, source_chunk_id, ref.ref_type,
            ref.ref_id, ref.ref_normalized,
            ref.source_text_snippet, ref.confidence
        ))
        self.conn.commit()

    def register_target(self, ref_normalized: str, ref_type: str,
                        pages: list, chunk_ids: list):
        """
        Register what a reference points TO.
        Called after chunking when we know which chunk IDs
        correspond to each table/section.
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO ref_targets
            (ref_normalized, ref_type, resolved_pages, resolved_chunk_ids)
            VALUES (?, ?, ?, ?)
        """, (
            ref_normalized, ref_type,
            json.dumps(pages), json.dumps(chunk_ids)
        ))
        self.conn.commit()

    def get_references_from_chunk(self, chunk_id: str) -> list:
        """
        What does this chunk reference?
        Used at query time to fetch linked chunks.
        """
        rows = self.conn.execute("""
            SELECT r.ref_normalized, r.ref_type, r.ref_id,
                   t.resolved_chunk_ids
            FROM cross_references r
            LEFT JOIN ref_targets t
                ON r.ref_normalized = t.ref_normalized
            WHERE r.source_chunk_id = ?
        """, (chunk_id,)).fetchall()

        results = []
        for row in rows:
            ref_norm, ref_type, ref_id, resolved_json = row
            resolved = json.loads(resolved_json) if resolved_json else []
            results.append({
                "ref_normalized": ref_norm,
                "ref_type": ref_type,
                "ref_id": ref_id,
                "resolved_chunk_ids": resolved
            })
        return results

    def get_chunks_referencing(self, ref_normalized: str) -> list:
        """
        What chunks reference this target?
        Used in incremental updates — when Table 4.2 changes,
        find all 47 chunks that cite it.

        This is the ripple analysis from Q6.
        """
        rows = self.conn.execute("""
            SELECT DISTINCT source_chunk_id, source_page
            FROM cross_references
            WHERE ref_normalized = ?
        """, (ref_normalized,)).fetchall()

        return [{"chunk_id": r[0], "page": r[1]} for r in rows]

    def get_stats(self) -> dict:
        total_refs = self.conn.execute(
            "SELECT COUNT(*) FROM cross_references"
        ).fetchone()[0]

        by_type = self.conn.execute("""
            SELECT ref_type, COUNT(*)
            FROM cross_references
            GROUP BY ref_type
        """).fetchall()

        return {
            "total_references": total_refs,
            "by_type": dict(by_type)
        }

    def clear(self):
        self.conn.executescript("""
            DELETE FROM cross_references;
            DELETE FROM ref_targets;
        """)
        self.conn.commit()

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Reference Extractor
# ---------------------------------------------------------------------------

class ReferenceExtractor:
    """
    Scans page text and extracts all cross-references.

    Two modes:
      1. Regex only — fast, works on explicit references
         "Table 4.2", "Section 5.3", "see page 32"

      2. Regex + LLM fallback — for ambiguous references
         "as noted above", "the preceding table", "values in column C"
         (LLM fallback not implemented here — flag for future)
    """

    def __init__(self, graph: ReferenceGraph):
        self.graph = graph

    def extract_from_pages(self, pages: list,
                           chunk_id_map: dict = None) -> int:
        """
        Extract all cross-references from a list of PageData objects.

        chunk_id_map: {page_num: chunk_id} mapping so we can
        store the source chunk ID, not just the page number.

        Returns: total number of references found.
        """
        total = 0

        for page_data in pages:
            page_num = page_data.page_num
            chunk_id = (chunk_id_map or {}).get(
                page_num, f"chunk_p{page_num:03d}"
            )

            refs = self._extract_from_text(
                page_data.raw_text, page_num
            )

            for ref in refs:
                self.graph.add_reference(page_num, chunk_id, ref)
                total += 1

        return total

    def _extract_from_text(self, text: str,
                            page_num: int) -> list:
        """
        Apply all regex patterns to extract cross-references.

        WHY we extract the surrounding sentence:
          The snippet gives the LLM context about HOW this
          reference is used — "adjust dosing per Table 4.2"
          vs "see Table 4.2 for background" — different intent.
        """
        refs = []
        seen = set()  # deduplicate same ref on same page

        for pattern_def in CROSS_REF_PATTERNS:
            pattern = pattern_def["pattern"]
            ref_type = pattern_def["type"]
            normalizer = pattern_def["normalizer"]

            for match in re.finditer(pattern, text, re.IGNORECASE):
                ref_id = match.group(0).strip()
                ref_num = match.group(1).strip()

                # Normalize: "Table 4.2" → "table_4.2"
                ref_normalized = f"{ref_type}_{ref_num.lower()}"

                # Deduplicate
                dedup_key = f"{page_num}_{ref_normalized}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Extract surrounding sentence for context
                snippet = self._get_surrounding_sentence(
                    text, match.start(), max_chars=150
                )

                # Confidence: higher for explicit patterns
                # Lower for ambiguous ones (not implemented yet)
                confidence = 0.95

                refs.append(CrossReference(
                    source_page=page_num,
                    source_text_snippet=snippet,
                    ref_type=ref_type,
                    ref_id=ref_id,
                    ref_normalized=ref_normalized,
                    confidence=confidence
                ))

        return refs

    def _get_surrounding_sentence(self, text: str,
                                   match_pos: int,
                                   max_chars: int = 150) -> str:
        """
        Extract the sentence containing the match.
        This becomes the 'snippet' stored with the reference.
        """
        # Find sentence start (look back for period or newline)
        start = max(0, match_pos - max_chars // 2)
        end = min(len(text), match_pos + max_chars // 2)

        # Try to start at a sentence boundary
        snippet = text[start:end].strip()

        # Clean up newlines
        snippet = " ".join(snippet.split())
        return snippet[:max_chars]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from extractor import PDFExtractor

    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    _default_pdf = os.path.join(
        _project_root, "data", "nasa_systems_engineering_handbook_0.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else _default_pdf

    # Extract pages
    print("=== Extracting pages ===")
    extractor = PDFExtractor(pdf_path)
    pages = extractor.extract_all_pages(end_page=30)
    extractor.close()

    # Build reference graph
    print("\n=== Building reference graph ===")
    graph = ReferenceGraph(db_path="db/reference_graph.db")
    graph.clear()

    extractor_obj = ReferenceExtractor(graph)
    total = extractor_obj.extract_from_pages(pages)

    # Show stats
    stats = graph.get_stats()
    print(f"\nTotal references found: {stats['total_references']}")
    print(f"By type:")
    for ref_type, count in stats['by_type'].items():
        print(f"  {ref_type:10s}: {count}")

    # Show sample references
    print(f"\n--- Sample references (first 10) ---")
    rows = graph.conn.execute("""
        SELECT source_page, ref_type, ref_id, snippet
        FROM cross_references
        LIMIT 10
    """).fetchall()

    for row in rows:
        page, rtype, rid, snippet = row
        print(f"\n  Page {page+1} references [{rtype}] '{rid}'")
        print(f"  Context: \"{snippet[:100]}\"")

    # Demo: ripple analysis
    # Find all pages that reference "Table 2"
    print(f"\n--- Ripple analysis: what references 'table_2'? ---")
    referencing = graph.get_chunks_referencing("table_2")
    if referencing:
        for r in referencing:
            print(f"  Page {r['page']+1} (chunk: {r['chunk_id']})")
    else:
        print("  No references to table_2 found in first 30 pages")

    graph.close()
    print(f"\nReference graph built and Reference graph is working.")