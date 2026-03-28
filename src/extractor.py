"""
extractor.py 
--------------------
Responsibility: Open the PDF, extract raw content page by page.
Output: A list of PageData objects, each containing:
  - page number
  - raw text
  - detected tables (as dicts with bbox, rows, html)
  - images (position + bytes)

WHY PyMuPDF (fitz) over pdfplumber or PyPDF2?
  - PyMuPDF gives us bbox (bounding box) coordinates for every element
  - We need coordinates to detect table continuations across pages (Q1)
  - PyPDF2 gives text only — no layout info, useless for our problem
  - pdfplumber is good but slower; PyMuPDF is faster for large docs
"""

import fitz  # PyMuPDF
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TableData:
    """
    Represents a single table detected on a single page.
    Note: this is RAW — not yet reconstructed across pages.
    That happens in reconstructor.py
    """
    page_num: int
    table_index: int          # which table on this page (0-based)
    bbox: tuple               # (x0, y0, x1, y1) — position on page
    col_count: int            # number of columns
    rows: list                # list of lists: rows[row_idx][col_idx] = cell text
    html: str                 # HTML representation — preserves structure for RAG
    footer_text: str = ""     # text below the table bbox — may contain table title
    header_candidate: bool = False  # True if first row looks like a header


@dataclass
class PageData:
    """
    Everything we extract from a single PDF page.
    """
    page_num: int
    width: float
    height: float
    raw_text: str             # full page text (for cross-ref detection)
    tables: list              # list of TableData objects
    images: list              # list of dicts: {bbox, image_bytes, ext}
    text_blocks: list         # list of dicts: {bbox, text} — for layout analysis


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class PDFExtractor:
    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        print(f"Opened PDF: {pdf_path}")
        print(f"Total pages: {self.total_pages}")

    def extract_all_pages(self, start_page: int = 0, end_page: int = None) -> list:
        """
        Extract all pages and return list of PageData objects.
        
        start_page / end_page let us process a subset — useful for
        incremental updates (only re-process changed pages).
        """
        if end_page is None:
            end_page = self.total_pages

        pages = []
        for page_num in range(start_page, end_page):
            print(f"  Extracting page {page_num + 1}/{end_page}...", end="\r")
            page_data = self._extract_single_page(page_num)
            pages.append(page_data)
        
        print(f"\nExtracted {len(pages)} pages.")
        return pages

    def _extract_single_page(self, page_num: int) -> PageData:
        page = self.doc[page_num]
        width = page.rect.width
        height = page.rect.height

        # --- Raw text (used for cross-reference detection later) ---
        raw_text = page.get_text("text")

        # --- Text blocks with bbox (used for layout/column analysis) ---
        # Each block: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        text_blocks = []
        for b in blocks:
            if b[6] == 0:  # type 0 = text block (type 1 = image)
                text_blocks.append({
                    "bbox": (b[0], b[1], b[2], b[3]),
                    "text": b[4].strip()
                })

        # --- Tables ---
        tables = self._extract_tables(page, page_num, height)

        # --- Images ---
        images = self._extract_images(page, page_num, text_blocks=text_blocks, raw_text=raw_text)

        return PageData(
            page_num=page_num,
            width=width,
            height=height,
            raw_text=raw_text,
            tables=tables,
            images=images,
            text_blocks=text_blocks
        )

    def _row_is_nasa_handbook_footer(self, row: list) -> bool:
        """Repeated page footer: title + page number — not a data table."""
        if not row or len(row) < 2:
            return False
        parts = [str(c or "").strip() for c in row]
        joined = " ".join(parts).upper()
        return (
            "NASA" in joined
            and "SYSTEMS ENGINEERING" in joined
            and "HANDBOOK" in joined
        )

    def _is_handbook_running_footer_table(
        self, rows: list, bbox: tuple, page_height: float
    ) -> bool:
        """
        Drop the handbook's 3-column page footer PyMuPDF detects as a table.

        If kept, reconstructor merges consecutive footers into one fake
        multi-page 'table' (pages iii–xix), polluting retrieval.
        """
        if not rows or page_height <= 0:
            return False
        y0 = bbox[1]
        in_bottom_band = y0 >= page_height * 0.78
        if not in_bottom_band:
            return False
        if len(rows) <= 3 and all(self._row_is_nasa_handbook_footer(r) for r in rows):
            return True
        if len(rows) == 1 and self._row_is_nasa_handbook_footer(rows[0]):
            return True
        return False

    def _is_toc_figure_banner_table(self, rows: list, bbox: tuple,
                                     page_height: float) -> bool:
        """
        Top-of-page 2-column banner: empty | 'Table of Contents' / Figures / etc.
        Not body content — avoids false merges and junk chunks.
        """
        if len(rows) != 1:
            return False
        r = rows[0]
        if len(r) != 2:
            return False
        a, b = str(r[0] or "").strip(), str(r[1] or "").strip()
        if a:
            return False
        labels = (
            "Table of Contents",
            "Table of Figures",
            "Table of Tables",
            "Table of Boxes",
        )
        if b not in labels:
            return False
        y0 = bbox[1]
        return y0 < page_height * 0.25

    def _is_sparse_layout_false_table(self, rows: list) -> bool:
        """
        Vector figures (e.g. V-model poster) split into many high-column
        grids with mostly empty cells — not useful as tabular data.
        """
        if not rows:
            return False
        col_count = max(len(row) for row in rows)
        if col_count < 5:
            return False
        cells = [c for row in rows for c in row]
        if len(cells) < 10:
            return False
        empty = sum(1 for c in cells if c is None or str(c).strip() == "")
        return (empty / len(cells)) >= 0.62

    def _extract_tables(self, page, page_num: int, page_height: float) -> list:
        """
        Detect and extract tables from a single page.
        
        WHY we check header_candidate:
          In multi-page tables, page 2 often starts with a repeated header row.
          We need to detect and suppress these duplicates when merging.
          A header row typically has: bold font, short cells, or all-caps text.
        """
        tables = []
        
        try:
            # PyMuPDF's find_tables() returns a TableFinder object
            # It uses spatial analysis to detect table boundaries
            table_finder = page.find_tables()
            
            for idx, table in enumerate(table_finder.tables):
                bbox = table.bbox  # (x0, y0, x1, y1)
                
                # Extract cells as a 2D list
                # extract() returns list of rows, each row is list of cell strings
                rows = table.extract()
                
                if not rows:
                    continue

                if self._is_handbook_running_footer_table(rows, bbox, page_height):
                    continue
                if self._is_toc_figure_banner_table(rows, bbox, page_height):
                    continue
                if self._is_sparse_layout_false_table(rows):
                    continue
                
                col_count = max(len(row) for row in rows) if rows else 0
                
                # Generate HTML representation
                # WHY HTML: preserves cell boundaries better than plain text
                # when we later chunk and embed this table
                html = self._table_to_html(rows)
                
                # Look for footer text (table title often appears below)
                footer_text = self._get_footer_text(
                    page, bbox, page_height
                )
                
                # Detect if first row is a header
                header_candidate = self._is_header_row(rows[0]) if rows else False
                
                tables.append(TableData(
                    page_num=page_num,
                    table_index=idx,
                    bbox=bbox,
                    col_count=col_count,
                    rows=rows,
                    html=html,
                    footer_text=footer_text,
                    header_candidate=header_candidate
                ))
        
        except Exception as e:
            # PyMuPDF table detection can fail on complex layouts
            # We log and continue — don't crash the whole pipeline
            print(f"\n  Warning: Table extraction failed on page {page_num}: {e}")
        
        return tables

    def _get_footer_text(self, page, table_bbox: tuple, 
                          page_height: float, margin: float = 50) -> str:
        """
        Get text immediately below a table bbox.
        
        WHY: In the NASA handbook, table titles like "TABLE 2.2-1 ..."
        may appear in a footer BELOW the table data, not above.
        Standard parsers miss this entirely.
        
        We look for text within `margin` points below the table's bottom edge.
        """
        x0, y0, x1, y1 = table_bbox
        
        # Define a search rectangle: below the table, same width
        footer_rect = fitz.Rect(x0, y1, x1, min(y1 + margin, page_height))
        
        # Extract text in that region
        footer_text = page.get_text("text", clip=footer_rect).strip()
        return footer_text

    def _is_header_row(self, row: list) -> bool:
        """
        Heuristic: is this row likely a column header?
        
        Signs of a header row:
          - Short cell text (column names are rarely long sentences)
          - Contains short labels, acronyms, or units — common in SE tables
          - All cells non-empty (data rows often have empty cells)
        
        WHY this matters: When merging page 2 rows onto page 1's table,
        we need to skip the repeated header row that PDF generators
        often inject at the top of each continuation page.
        """
        if not row:
            return False
        
        non_empty = [c for c in row if c and str(c).strip()]
        if len(non_empty) < len(row) * 0.7:
            return False  # too many empty cells for a header
        
        # Check average cell length — headers tend to be short
        avg_len = sum(len(str(c)) for c in non_empty) / len(non_empty)
        if avg_len > 40:
            return False  # cells too long — probably data
        
        return True

    def _table_to_html(self, rows: list) -> str:
        """
        Convert rows (list of lists) to HTML table string.
        
        WHY HTML over plain text:
          - Preserves cell boundaries explicitly
          - LLMs understand HTML tables well
          - Easier to reconstruct column alignment when merging fragments
        """
        if not rows:
            return ""
        
        html = "<table>\n"
        for i, row in enumerate(rows):
            html += "  <tr>\n"
            tag = "th" if i == 0 else "td"
            for cell in row:
                cell_text = str(cell).strip() if cell else ""
                html += f"    <{tag}>{cell_text}</{tag}>\n"
            html += "  </tr>\n"
        html += "</table>"
        return html

    def _find_nearby_caption(self,
                             img_bbox: tuple,
                             text_blocks: list,
                             raw_text: str = "",
                             window: float = 80) -> str:
        """
        Try to find a caption near an image using surrounding text blocks.

        Universal (non-NASA-specific) patterns:
          - "Figure 2.1-1", "Fig. 3.2", "Table 4-1", "Diagram 5"
          - Captions are often directly below (or sometimes above) the image.
        """
        if not img_bbox:
            return ""

        x0, y0, x1, y1 = img_bbox
        caption_patterns = [
            r'^\s*(figure|fig\.?|table|chart|diagram)\s+[\w\d]+(?:[.\-][\w\d]+)*\b.*',
        ]

        def x_overlap_ratio(a0, a1, b0, b1) -> float:
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            denom = max(1.0, min(a1 - a0, b1 - b0))
            return inter / denom

        best = None  # (distance, text)
        for block in text_blocks or []:
            bb = block.get("bbox")
            txt = (block.get("text") or "").strip()
            if not bb or not txt:
                continue

            bx0, by0, bx1, by1 = bb
            # Horizontal alignment is a strong signal for caption association
            if x_overlap_ratio(x0, x1, bx0, bx1) < 0.25:
                continue

            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            for line in lines[:3]:
                if any(re.match(p, line, re.IGNORECASE) for p in caption_patterns):
                    # Prefer captions just below the image; accept above as fallback.
                    if by0 >= y1:
                        dist = by0 - y1
                    else:
                        dist = y0 - by1
                    if 0 <= dist <= window:
                        cand = (dist, line)
                        if best is None or cand[0] < best[0]:
                            best = cand

        if best:
            return best[1]

        # Fallback: scan raw text for first caption-like line (no layout guarantee)
        if raw_text:
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if any(re.match(p, line, re.IGNORECASE) for p in caption_patterns):
                    return line

        return ""

    def _extract_images(self, page, page_num: int, text_blocks: list = None, raw_text: str = "") -> list:
        """
        Extract embedded images from a page.
        
        WHY: Technical diagrams (lifecycle, flowcharts) may be embedded as images.
        We need the image bytes to later send to GPT-4o vision
        for diagram annotation extraction (Q3).
        """
        images = []
        
        # get_images() returns list of (xref, smask, w, h, bpc, cs, alt_cs, name, filter, referencer)
        image_list = page.get_images(full=True)
        
        for img_info in image_list:
            xref = img_info[0]
            
            try:
                # Extract image bytes
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]  # png, jpeg, etc.
                
                # Get image position on page
                # We need bbox to associate image with surrounding text
                img_rects = page.get_image_rects(xref)
                bbox = img_rects[0] if img_rects else None
                
                if bbox and len(image_bytes) > 1000:  # skip tiny images (bullets, logos)
                    bbox_tuple = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                    caption = self._find_nearby_caption(
                        img_bbox=bbox_tuple,
                        text_blocks=text_blocks or [],
                        raw_text=raw_text,
                        window=90
                    )
                    images.append({
                        "page_num": page_num,
                        "bbox": bbox_tuple,
                        "image_bytes": image_bytes,
                        "ext": ext,
                        "size_bytes": len(image_bytes),
                        "caption": caption
                    })
            except Exception as e:
                print(f"\n  Warning: Image extraction failed on page {page_num}: {e}")
        
        return images

    def get_page_content_hash(self, page_num: int) -> str:
        """
        Generate a stable hash of a page's TEXT CONTENT.
        
        WHY text hash NOT byte hash:
          PDF metadata changes (timestamps, rendering params) alter the
          raw byte hash even when text content is identical.
          We hash extracted text to get a content-stable fingerprint.
          Used in incremental update pipeline (Q6).
        """
        import hashlib
        page = self.doc[page_num]
        text = page.get_text("text")
        return hashlib.sha256(text.encode()).hexdigest()

    def get_all_page_hashes(self) -> dict:
        """Returns {page_num: content_hash} for entire document."""
        return {
            i: self.get_page_content_hash(i) 
            for i in range(self.total_pages)
        }

    def close(self):
        self.doc.close()


# ---------------------------------------------------------------------------
# Quick test — run this file directly to see output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    _default_pdf = os.path.join(
        _project_root, "data", "nasa_systems_engineering_handbook_0.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else _default_pdf
    
    extractor = PDFExtractor(pdf_path)
    
    # Extract first 5 pages only for quick test
    pages = extractor.extract_all_pages(end_page=20)
    
    print("\n--- EXTRACTION RESULTS ---")
    for page in pages:
        print(f"\nPage {page.page_num + 1}:")
        print(f"  Text length: {len(page.raw_text)} chars")
        print(f"  Tables found: {len(page.tables)}")
        print(f"  Images found: {len(page.images)}")
        
        for t in page.tables:
            print(f"    Table {t.table_index}: {t.col_count} cols, "
                  f"{len(t.rows)} rows, bbox={tuple(round(x,1) for x in t.bbox)}")
            print(f"      Footer text: '{t.footer_text[:80]}'")
            print(f"      Header candidate: {t.header_candidate}")
            if t.rows:
                print(f"      First row: {t.rows[0]}")
    
    # Show page hashes (used for incremental updates)
    print("\n--- PAGE HASHES (first 3 pages) ---")
    hashes = extractor.get_all_page_hashes()
    for page_num, h in list(hashes.items())[:3]:
        print(f"  Page {page_num + 1}: {h[:16]}...")
    
    extractor.close()
    print("\nextractor.py is working and Extraction is complete.")