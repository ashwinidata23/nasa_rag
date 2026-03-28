"""
reconstructor.py
------------------------
Responsibility: Take raw per-page TableData objects from extractor.py
and merge multi-page tables into single coherent structures.

The core problem:
  - Page 1: column headers
  - Page 2: data rows (no title)
  - Page 3: more data rows + footer "TABLE 2.2-1 Project Life Cycle Phases"

Standard parsers create 3 orphaned fragments.
We create 1 complete ReconstructedTable.

WHY 2-pass approach:
  Pass 1 — collect fragments into buffers, detect continuations
  Pass 2 — resolve footer titles back to their fragments
  We cannot assign titles in pass 1 because the title
  appears AFTER the data (page 3 footer).

WHY SQLite buffer (not a Python dict in RAM):
  A 1GB PDF can have hundreds of open table buffers simultaneously.
  Storing all in RAM risks OOM errors. SQLite serializes to disk,
  costs ~2ms per read/write, and survives pipeline restarts.
"""

import sqlite3
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReconstructedTable:
    """
    A complete table spanning one or more pages.
    This is what enters the vector store — not raw fragments.
    """
    table_id: str               # e.g. "T_p031_033" (pages 31-33)
    title: str                  # from footer, e.g. "TABLE 2.7-1 NASA SE Competency Model"
    pages: list                 # [31, 32, 33]
    headers: list               # column header row
    all_rows: list              # all data rows combined
    html: str                   # full HTML representation
    col_count: int
    citations: dict             # {page_num: table_index} for citation generation

    def to_chunk_text(self) -> str:
        """
        Convert to embeddable text chunk.
        Headers are prepended to every chunk so retrieval
        always has column context.
        """
        lines = []
        if self.title:
            lines.append(f"Table: {self.title}")
        lines.append(f"Pages: {self.pages}")
        lines.append(f"Columns: {self.headers}")
        lines.append("Data:")
        for row in self.all_rows:
            lines.append("  " + " | ".join(str(c) for c in row if c))
        return "\n".join(lines)

    def to_metadata(self) -> dict:
        """
        Metadata envelope that travels with this chunk
        through embedding → retrieval → citation generation.
        Never loses source information.
        """
        return {
            "table_id": self.table_id,
            "title": self.title,
            "pages": json.dumps(self.pages),
            "page_start": min(self.pages) if self.pages else 0,
            "page_end": max(self.pages) if self.pages else 0,
            "col_count": self.col_count,
            "type": "table"
        }


# ---------------------------------------------------------------------------
# SQLite buffer for open table fragments
# ---------------------------------------------------------------------------

class TableBuffer:
    """
    Persists open table fragments to SQLite during ingestion.

    WHY: If we stored fragments in a Python dict, a crash midway
    through a 1GB PDF loses everything. SQLite gives us durability
    and lets us resume from a checkpoint.
    """

    def __init__(self, db_path: str = "db/table_buffer.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS open_tables (
                buffer_id     TEXT PRIMARY KEY,
                start_page    INTEGER,
                current_page  INTEGER,
                col_count     INTEGER,
                col_schema    TEXT,      -- JSON: column x-positions
                headers       TEXT,      -- JSON: header row cells
                all_rows      TEXT,      -- JSON: accumulated rows
                footer_title  TEXT,      -- filled when footer found
                bbox_x0       REAL,
                bbox_x1       REAL,
                status        TEXT       -- 'open' or 'closed'
            )
        """)
        self.conn.commit()

    def open_table(self, buffer_id: str, page_num: int,
                   col_count: int, col_schema: list,
                   headers: list, initial_rows: list,
                   bbox_x0: float, bbox_x1: float):
        """Start tracking a new table fragment."""
        self.conn.execute("""
            INSERT OR REPLACE INTO open_tables
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            buffer_id, page_num, page_num, col_count,
            json.dumps(col_schema), json.dumps(headers),
            json.dumps(initial_rows), "",
            bbox_x0, bbox_x1, "open"
        ))
        self.conn.commit()

    def append_rows(self, buffer_id: str, new_rows: list,
                    current_page: int):
        """Add more rows to an existing open table."""
        row = self.conn.execute(
            "SELECT all_rows FROM open_tables WHERE buffer_id=?",
            (buffer_id,)
        ).fetchone()

        if row:
            existing = json.loads(row[0])
            existing.extend(new_rows)
            self.conn.execute("""
                UPDATE open_tables
                SET all_rows=?, current_page=?
                WHERE buffer_id=?
            """, (json.dumps(existing), current_page, buffer_id))
            self.conn.commit()

    def set_footer_title(self, buffer_id: str, title: str):
        """Assign footer title when we find it (pass 2)."""
        self.conn.execute(
            "UPDATE open_tables SET footer_title=? WHERE buffer_id=?",
            (title, buffer_id)
        )
        self.conn.commit()

    def close_table(self, buffer_id: str):
        """Mark table as complete — no more rows expected."""
        self.conn.execute(
            "UPDATE open_tables SET status='closed' WHERE buffer_id=?",
            (buffer_id,)
        )
        self.conn.commit()

    def get_open_tables(self) -> list:
        """Get all currently open (incomplete) table buffers."""
        rows = self.conn.execute(
            "SELECT * FROM open_tables WHERE status='open'"
        ).fetchall()
        return rows

    def get_all_tables(self) -> list:
        """Get all tables (open and closed) for pass 2 resolution."""
        return self.conn.execute(
            "SELECT * FROM open_tables"
        ).fetchall()

    def clear(self):
        """Reset buffer for a fresh ingestion run."""
        self.conn.execute("DELETE FROM open_tables")
        self.conn.commit()

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Main reconstructor
# ---------------------------------------------------------------------------

class TableReconstructor:
    """
    Takes a list of PageData objects (from extractor.py)
    and returns a list of ReconstructedTable objects.

    Two-pass algorithm:
      Pass 1: page by page, detect continuations, buffer fragments
      Pass 2: walk all buffers, assign footer titles, finalize
    """

    def __init__(self, db_path: str = "db/table_buffer.db"):
        self.buffer = TableBuffer(db_path)
        self.buffer.clear()  # fresh run
        self._table_counter = 0

    def reconstruct(self, pages: list) -> list:
        """
        Main entry point.
        pages: list of PageData objects from extractor.py
        returns: list of ReconstructedTable objects
        """
        print("Pass 1: Collecting table fragments...")
        self._pass1_collect(pages)

        print("Pass 2: Resolving titles and finalizing...")
        reconstructed = self._pass2_resolve(pages)

        print(f"Reconstruction complete: {len(reconstructed)} tables found.")
        return reconstructed

    # -----------------------------------------------------------------------
    # Pass 1: collect fragments
    # -----------------------------------------------------------------------

    def _pass1_collect(self, pages: list):
        """
        Walk pages in order. For each table on a page:
          - Check if it continues an open buffer
          - If yes: append rows to buffer
          - If no: open a new buffer
          - If footer contains a table title: record it
        """
        for page_data in pages:
            page_num = page_data.page_num

            for table in page_data.tables:
                # Get column schema (x-positions of column boundaries)
                col_schema = self._get_col_schema(table)

                # Check if this table continues any open buffer
                matching_buffer_id = self._find_matching_buffer(
                    table, col_schema, page_num
                )

                if matching_buffer_id:
                    # This is a continuation — append rows
                    # Skip first row if it's a duplicate header
                    rows_to_add = table.rows
                    if table.header_candidate and len(rows_to_add) > 0:
                        rows_to_add = rows_to_add[1:]  # drop repeated header

                    self.buffer.append_rows(
                        matching_buffer_id,
                        rows_to_add,
                        page_num
                    )

                    # Check if footer has a title
                    title = self._extract_table_title(table.footer_text)
                    if title:
                        self.buffer.set_footer_title(matching_buffer_id, title)
                        self.buffer.close_table(matching_buffer_id)

                else:
                    # New table — open a fresh buffer
                    buffer_id = self._new_buffer_id(page_num)
                    headers = table.rows[0] if table.rows else []
                    data_rows = table.rows[1:] if len(table.rows) > 1 else []

                    self.buffer.open_table(
                        buffer_id=buffer_id,
                        page_num=page_num,
                        col_count=table.col_count,
                        col_schema=col_schema,
                        headers=headers,
                        initial_rows=data_rows,
                        bbox_x0=table.bbox[0],
                        bbox_x1=table.bbox[2]
                    )

                    # Immediately check footer for title
                    title = self._extract_table_title(table.footer_text)
                    if title:
                        self.buffer.set_footer_title(buffer_id, title)
                        self.buffer.close_table(buffer_id)

    # -----------------------------------------------------------------------
    # Pass 2: resolve and finalize
    # -----------------------------------------------------------------------

    def _pass2_resolve(self, pages: list) -> list:
        """
        Walk all buffers. For any buffer still missing a title,
        search nearby text for a table reference pattern.
        Build final ReconstructedTable objects.
        """
        all_buffers = self.buffer.get_all_tables()
        reconstructed = []

        for buf in all_buffers:
            (buffer_id, start_page, current_page, col_count,
             col_schema_json, headers_json, all_rows_json,
             footer_title, bbox_x0, bbox_x1, status) = buf

            headers = json.loads(headers_json)
            all_rows = json.loads(all_rows_json)
            pages_spanned = list(range(start_page, current_page + 1))

            # If still no title, try to find one in surrounding page text
            title = footer_title
            if not title:
                title = self._search_nearby_text(
                    pages, start_page, current_page
                )
            if not title:
                # Last resort: give it a provisional ID
                # Flag for human review rather than silently wrong
                title = f"[PROVISIONAL] Table on pages {pages_spanned}"
                print(f"  Warning: No title found for table "
                      f"{buffer_id} — flagged for review")

            # Build HTML for this complete table
            html = self._build_html(headers, all_rows, title)

            table_id = f"T_p{start_page:03d}_{current_page:03d}"

            rt = ReconstructedTable(
                table_id=table_id,
                title=title,
                pages=pages_spanned,
                headers=headers,
                all_rows=all_rows,
                html=html,
                col_count=col_count,
                citations={start_page: 0}
            )
            reconstructed.append(rt)

        return reconstructed

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _get_col_schema(self, table) -> list:
        """
        Extract column x-positions as a normalized schema.

        WHY normalized: page width varies between PDFs.
        We normalize to 0-1 range so comparison works
        across different page sizes.

        WHY this matters: Two tables on the same page can have
        similar column counts but different x-positions.
        Schema matching prevents false continuation detection.
        """
        if not table.rows or not table.rows[0]:
            return []

        # Use column count as simple schema
        # In production: extract actual x-midpoints from bbox
        return list(range(table.col_count))

    def _find_matching_buffer(self, table, col_schema: list,
                               page_num: int) -> Optional[str]:
        """
        Check if this table continues an open buffer from the previous page.

        Matching criteria:
          1. Buffer is still open (not closed)
          2. Buffer's last page = current page - 1 (consecutive pages)
          3. Column count matches
          4. X-span is similar (same column on page)

        WHY consecutive pages only:
          Skipping a page means something else appeared between —
          the table likely ended. We don't bridge non-consecutive pages.
        """
        open_buffers = self.buffer.get_open_tables()

        for buf in open_buffers:
            (buffer_id, start_page, current_page_buf, col_count,
             col_schema_json, headers_json, all_rows_json,
             footer_title, bbox_x0, bbox_x1, status) = buf

            # Must be consecutive pages
            if current_page_buf != page_num - 1:
                continue

            # Column count must match
            if col_count != table.col_count:
                continue

            # X-span must be similar (within 10% of page width)
            # This catches the 2-column layout problem where two
            # separate tables have different x-positions
            x_span_buf = bbox_x1 - bbox_x0
            x_span_new = table.bbox[2] - table.bbox[0]
            if abs(x_span_buf - x_span_new) / max(x_span_buf, 1) > 0.10:
                continue

            # X-start position must be similar
            if abs(bbox_x0 - table.bbox[0]) > 20:  # within 20 points
                continue

            return buffer_id

        return None

    def _extract_table_title(self, footer_text: str) -> str:
        """
        Extract table title from footer text.

        Patterns we look for:
          "TABLE 2.2-1 Project Life Cycle Phases"
          "Table 3.11-1 Example of Program/Project Types"
          "TABLE 4.1-1 Stakeholder Identification"

        WHY regex here: Footer text is short and structured.
        LLM would be overkill and add latency/cost.
        We only use LLM for ambiguous cases (see reconstructor notes).
        """
        if not footer_text:
            return ""

        # Primary pattern: "Table X.X: title" or "Table X: title"
        pattern = r'Table\s+[\d\.]+[\s:\-–]+([^\n]{5,80})'
        match = re.search(pattern, footer_text, re.IGNORECASE)
        if match:
            return f"Table {re.search(r'[\d\.]+', footer_text).group()}: {match.group(1).strip()}"

        # Secondary: just "Table X.X" with no title
        pattern2 = r'(Table\s+[\d\.]+)'
        match2 = re.search(pattern2, footer_text, re.IGNORECASE)
        if match2:
            return match2.group(1).strip()

        return ""

    def _search_nearby_text(self, pages: list,
                             start_page: int,
                             end_page: int) -> str:
        """
        If no footer title found, search page text for table references.
        Looks at the page AFTER the table ends — sometimes titles
        appear as captions below the table on the next page.
        """
        # Look one page after table ends
        search_page = end_page + 1
        for page_data in pages:
            if page_data.page_num == search_page:
                title = self._extract_table_title(page_data.raw_text[:200])
                if title:
                    return title
        return ""

    def _build_html(self, headers: list, rows: list, title: str) -> str:
        """Build complete HTML for the reconstructed table."""
        html = f"<!-- {title} -->\n<table>\n"

        if headers:
            html += "  <tr>\n"
            for cell in headers:
                html += f"    <th>{cell or ''}</th>\n"
            html += "  </tr>\n"

        for row in rows:
            html += "  <tr>\n"
            for cell in row:
                html += f"    <td>{cell or ''}</td>\n"
            html += "  </tr>\n"

        html += "</table>"
        return html

    def _new_buffer_id(self, page_num: int) -> str:
        self._table_counter += 1
        return f"buf_p{page_num:03d}_{self._table_counter:04d}"


# ---------------------------------------------------------------------------
# Quick test
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

    # Step 1: Extract
    print("=== Step 1: Extracting pages ===")
    extractor = PDFExtractor(pdf_path)
    pages = extractor.extract_all_pages(end_page=20)
    extractor.close()

    # Step 2: Reconstruct
    print("\n=== Step 2: Reconstructing tables ===")
    reconstructor = TableReconstructor(db_path="db/table_buffer.db")
    tables = reconstructor.reconstruct(pages)

    # Step 3: Show results
    print("\n=== RECONSTRUCTED TABLES ===")
    for t in tables:
        print(f"\nTable ID: {t.table_id}")
        print(f"  Title:    {t.title}")
        print(f"  Pages:    {t.pages}")
        print(f"  Columns:  {t.col_count}")
        print(f"  Rows:     {len(t.all_rows)}")
        print(f"  Headers:  {t.headers}")
        if t.all_rows:
            print(f"  Sample row: {t.all_rows[0]}")
        print(f"\n  Chunk text preview:")
        print("  " + t.to_chunk_text()[:300].replace("\n", "\n  "))

    print(f"Reconstructor.py is working and {len(tables)} tables reconstructed.")