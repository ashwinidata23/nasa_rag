"""
diagram_extractor.py
---------------------
Responsibility: Scan PDF pages for technical diagrams, send them to
GPT-4o vision, get structured descriptions, return Chunk objects
ready for embedding.

WHY page-render instead of embedded-image extraction:
  PyMuPDF's get_images() only returns raster images (JPEG/PNG bytes
  baked into the PDF). Vector drawings — lifecycle diagrams, process
  flowcharts, technical figures — are drawn as PDF path commands.
  They have no embedded image representation, so get_images() returns
  nothing for them.

  Rendering the whole page with page.get_pixmap() captures everything
  in one shot: vector paths, raster images, AND caption text that sits
  next to the diagram. GPT-4o then reads "FIGURE 2.2-1" directly from
  the rendered image and puts it in the caption field.

HOW it works:
  1. For each page: _page_has_diagram_content() checks for drawings/
     images using only PyMuPDF (no extra dependencies). Skips pure text.
  2. _render_page() renders the page at 2x zoom → PNG bytes.
  3. PNG sent to GPT-4o vision with structured JSON prompt.
  4. Response serialised → Chunk (embedded + indexed like any other).

COST GATE:
  Each rendered page is hashed before sending to GPT-4o.
  If hash exists in db/diagram_cache.db → skip API call.
  Re-running pipeline never re-pays for the same page.

INTEGRATION:
  diagram_chunks = DiagramExtractor().extract_all(pages, pdf_path=pdf_path)
  all_chunks += diagram_chunks
"""

import os
import sys
import json
import base64
import hashlib
import sqlite3
import time
from typing import Optional

import fitz  # PyMuPDF — already used by extractor.py
from openai import OpenAI
from dotenv import load_dotenv
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chunker import Chunk

load_dotenv()


# ---------------------------------------------------------------------------
# Structured prompt for GPT-4o vision (universal)
# ---------------------------------------------------------------------------

VISION_PROMPT = """You are analyzing a full rendered page from an engineering/technical handbook.
The page may contain one or more technical diagrams, flowcharts, models, or table images.

If this page is purely text (no diagram, chart, or figure), return:
{"diagram_type": "none", "title": "", "caption": "", "summary": "Text-only page.",
 "nodes": [], "connections": [], "tables": [], "key_terms": [], "refs": []}

Otherwise extract ALL visible text and structure into structured JSON.
Return ONLY valid JSON — no markdown, no explanation, no code fences.

JSON format (ONLY JSON):
{
  "diagram_type": "flowchart|process_flow|model|matrix|table_image|timeline|decision_tree|other",
  "title": "the main title text visible in the diagram",
  "caption": "the figure/table caption exactly as written, e.g. 'FIGURE 2.2-1 NASA Project Life Cycle'",
  "summary": "2 sentences: what this specific diagram shows and its purpose",
  "nodes": ["every labeled box, stage, step, or component visible"],
  "connections": ["A → B", "Phase A → Phase B", "Input → Output (label)"],
  "tables": ["if diagram is a table-image: list column headers or key rows"],
  "key_terms": ["5-10 important technical terms visible in the diagram"],
  "refs": ["any cross-references like 'See Table 4.2', 'Section 6.4', 'NPR 7123.1'"]
}

Rules:
- caption: copy figure/table label text EXACTLY (e.g. 'FIGURE 2.1-1 The SE Engine').
- nodes: be exhaustive — every labeled element.
- connections: every arrow or flow relationship.
- If a field has no content use empty string "" or empty list [].
"""


# ---------------------------------------------------------------------------
# Diagram cache (avoid re-paying for same image)
# ---------------------------------------------------------------------------

class DiagramCache:
    """
    SQLite cache keyed by image content hash.
    
    WHY: Re-running pipeline.py after adding new text chunks
    should NOT re-send all 20 images to GPT-4o vision.
    Hash the image bytes → check cache → skip if exists.
    """
    
    def __init__(self, db_path: str = "db/diagram_cache.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS diagram_descriptions (
                image_hash   TEXT PRIMARY KEY,
                page_num     INTEGER,
                description  TEXT,      -- full JSON from GPT-4o
                chunk_text   TEXT,      -- serialized text for embedding
                created_at   TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def get(self, image_hash: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT description, chunk_text FROM diagram_descriptions WHERE image_hash=?",
            (image_hash,)
        ).fetchone()
        if row:
            return {"description": json.loads(row[0]), "chunk_text": row[1]}
        return None

    def store(self, image_hash: str, page_num: int,
              description: dict, chunk_text: str):
        self.conn.execute("""
            INSERT OR REPLACE INTO diagram_descriptions
            (image_hash, page_num, description, chunk_text)
            VALUES (?, ?, ?, ?)
        """, (image_hash, page_num, json.dumps(description), chunk_text))
        self.conn.commit()

    def get_all(self) -> list:
        """Return all cached descriptions — used if re-running pipeline."""
        return self.conn.execute(
            "SELECT image_hash, page_num, description, chunk_text FROM diagram_descriptions"
        ).fetchall()

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Main diagram extractor
# ---------------------------------------------------------------------------

class DiagramExtractor:
    """
    Extracts technical diagrams from PDF pages using full-page rendering.

    Primary mode (page-render):
      Renders each page with page.get_pixmap() — captures vector diagrams
      (lifecycle flows, schematic figures) that get_images() completely misses.
      GPT-4o reads caption text ("FIGURE 2.2-1 ...") from the rendered image.

    Fallback mode (embedded-images):
      Used when pdf_path is not available. Only finds raster images
      embedded in the PDF — misses vector diagrams.

    Filter:
      _page_has_diagram_content() uses PyMuPDF's own drawing/image detection
      to skip pure-text pages before spending any API tokens.
    """

    # Max pages to send to GPT-4o per run (cost gate)
    MAX_PAGES = 60

    # Page render zoom factor (2x ≈ 150 DPI — clear enough for GPT to read text)
    RENDER_ZOOM = 2.0

    def __init__(self, cache_db: str = "db/diagram_cache.db"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache  = DiagramCache(cache_db)
        self._chunk_counter = 0

    # -----------------------------------------------------------------------
    # Page-level filter (no external dependencies — PyMuPDF only)
    # -----------------------------------------------------------------------

    def _page_has_diagram_content(self, page) -> bool:
        """
        Cheap check before rendering: does this page have anything visual?

        - Embedded raster images → almost certainly a photo/diagram.
        - Many vector paths → flowchart, lifecycle diagram, table-image.
        - Pure-text pages have zero images and ≤ a handful of border lines.

        Threshold of 20 drawing paths chosen empirically:
          - Simple bordered page: ~5 paths
          - Page with one diagram: 50-500+ paths
        """
        if page.get_images(full=False):
            return True
        return len(page.get_drawings()) > 20

    # -----------------------------------------------------------------------
    # Page renderer
    # -----------------------------------------------------------------------

    def _render_page(self, page) -> bytes:
        """
        Render a PDF page to PNG at RENDER_ZOOM.
        alpha=False keeps file size lower (no transparency channel needed).
        """
        mat = fitz.Matrix(self.RENDER_ZOOM, self.RENDER_ZOOM)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def extract_all(self, pages: list, pdf_path: str = None) -> list:
        """
        Main entry point. Returns list of Chunk objects.

        pdf_path provided → page-render mode (recommended).
        pdf_path=None     → embedded-image fallback (legacy).
        """
        if pdf_path:
            return self._extract_via_page_render(pdf_path, len(pages))
        return self._extract_via_embedded_images(pages)

    # -----------------------------------------------------------------------
    # Page-render mode (primary — captures vector diagrams)
    # -----------------------------------------------------------------------

    def _extract_via_page_render(self, pdf_path: str, total_pages: int) -> list:
        """
        Render each page as a full PNG and send diagram-containing pages to GPT-4o.
        This captures vector diagrams that get_images() completely misses.
        """
        print(f"\n[DIAGRAMS] Page-render mode: scanning {total_pages} pages in {pdf_path}...")
        doc = fitz.open(pdf_path)
        n = min(total_pages, self.MAX_PAGES, len(doc))

        # Filter to pages that actually have visual content
        candidate_pages = [
            pn for pn in range(n)
            if self._page_has_diagram_content(doc[pn])
        ]
        skipped = n - len(candidate_pages)
        print(f"[DIAGRAMS] {len(candidate_pages)} pages with diagram content "
              f"({skipped} pure-text pages skipped)")

        chunks = []
        for i, page_num in enumerate(candidate_pages):
            page = doc[page_num]
            page_display = page_num + 1
            print(f"[DIAGRAMS] Page {page_display} ({i+1}/{len(candidate_pages)})...",
                  end=" ", flush=True)

            img_bytes   = self._render_page(page)
            image_hash  = hashlib.md5(img_bytes).hexdigest()
            img_info    = {
                "page_num":    page_num,
                "image_bytes": img_bytes,
                "ext":         "png",
                "caption":     "",   # GPT reads caption from the rendered image directly
                "hash":        image_hash,
            }

            chunk = self._process_image(img_info)
            if chunk:
                chunks.append(chunk)
                cap = chunk.metadata.get("caption", "")
                print(f"✓  {cap}" if cap else "✓")
            else:
                print("skipped (text-only page)")

        doc.close()
        self.cache.close()
        print(f"[DIAGRAMS] Done. {len(chunks)} diagram chunks created.")
        return chunks

    # -----------------------------------------------------------------------
    # Embedded-image fallback (legacy — misses vector diagrams)
    # -----------------------------------------------------------------------

    def _extract_via_embedded_images(self, pages: list) -> list:
        """
        Legacy path: extract raster images already in PageData objects.
        Use this only when pdf_path is unavailable.
        """
        MIN_SIZE_BYTES = 10_000
        print(f"\n[DIAGRAMS] Embedded-image mode (fallback): scanning {len(pages)} pages...")

        all_images = []
        for page_data in pages:
            for img in page_data.images:
                size = img.get("size_bytes", 0)
                if size < MIN_SIZE_BYTES:
                    continue
                caption = img.get("caption", "") or ""
                all_images.append({
                    "page_num":    page_data.page_num,
                    "image_bytes": img["image_bytes"],
                    "ext":         img.get("ext", "png"),
                    "bbox":        img.get("bbox"),
                    "size_bytes":  size,
                    "caption":     caption,
                })

        seen_hashes: set = set()
        unique_images = []
        for img in all_images:
            h = hashlib.md5(img["image_bytes"]).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                img["hash"] = h
                unique_images.append(img)

        print(f"[DIAGRAMS] {len(unique_images)} unique images (above {MIN_SIZE_BYTES:,} bytes)")

        chunks = []
        cap_list = unique_images[:self.MAX_PAGES]
        for i, img in enumerate(cap_list):
            page_display = img["page_num"] + 1
            print(f"[DIAGRAMS] Image {i+1}/{len(cap_list)} "
                  f"(page {page_display}, {img['size_bytes']:,} bytes)...",
                  end=" ", flush=True)
            chunk = self._process_image(img)
            if chunk:
                chunks.append(chunk)
                print("✓")
            else:
                print("skipped")

        self.cache.close()
        print(f"[DIAGRAMS] Done. {len(chunks)} diagram chunks created.")
        return chunks

    def _process_image(self, img: dict) -> Optional[Chunk]:
        """
        Process a single image: cache check → GPT-4o → serialize → Chunk.
        """
        image_hash = img["hash"]
        page_num   = img["page_num"]

        # Check cache first
        cached = self.cache.get(image_hash)
        if cached:
            print("(cache hit)", end=" ", flush=True)
            return self._build_chunk(
                cached["description"],
                cached["chunk_text"],
                page_num,
                image_hash
            )

        # Not in cache — call GPT-4o vision
        description = self._call_vision_api(img)
        if not description:
            return None

        # GPT signals text-only page via diagram_type == "none"
        if description.get("diagram_type") == "none":
            return None

        # Serialize to searchable text
        chunk_text = self._serialize_description(description, page_num)

        # Store in cache
        self.cache.store(image_hash, page_num, description, chunk_text)

        return self._build_chunk(description, chunk_text, page_num, image_hash)

    def _call_vision_api(self, img: dict) -> Optional[dict]:
        """
        Send image to GPT-4o vision. Returns parsed JSON or None on failure.
        
        WHY base64: OpenAI vision API accepts either URL or base64.
        We use base64 so there's no need to host images anywhere.
        """
        try:
            # Encode image as base64
            b64 = base64.b64encode(img["image_bytes"]).decode("utf-8")
            ext = img.get("ext", "png").lower()

            # Map extension to MIME type
            mime_map = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png",  "gif": "image/gif",
                "webp": "image/webp"
            }
            mime = mime_map.get(ext, "image/png")

            response = self.client.chat.completions.create(
                model="gpt-4o",          # Must be gpt-4o for vision
                max_tokens=800,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                    "detail": "high"  # high = more tokens, better for diagrams
                                }
                            },
                            {
                                "type": "text",
                                "text": (
                                    VISION_PROMPT
                                    + f"\n\nContext:\n"
                                    + f"- Page number: {img.get('page_num', 0) + 1}\n"
                                    + f"- This is a full rendered page (vector + raster content visible).\n"
                                    + f"- Read any 'FIGURE X.X-X' or 'TABLE X' caption text from the image itself.\n"
                                )
                            }
                        ]
                    }
                ],
                temperature=0  # deterministic output for structured data
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            return parsed

        except json.JSONDecodeError as e:
            print(f"\n[DIAGRAMS] JSON parse error: {e}")
            print(f"[DIAGRAMS] Raw response: {raw[:200]}")
            return None
        except Exception as e:
            print(f"\n[DIAGRAMS] Vision API error: {e}")
            # Rate limit — wait and the pipeline continues
            if "rate" in str(e).lower():
                print("[DIAGRAMS] Rate limited — waiting 10s...")
                time.sleep(10)
            return None

    def _serialize_description(self, desc: dict, page_num: int) -> str:
        """
        Convert GPT-4o JSON output into searchable text for embedding.
        
        WHY not just embed the raw JSON:
          JSON keys and brackets hurt embedding quality.
          Plain English with labeled sections embeds much better.
          e.g. "Diagram type: life cycle flow. Shows: phases left to right..."
          matches natural questions about that figure.
          
        WHY include all fields as separate lines:
          Each concept (node, flow step, review name) gets its own
          embedding weight. A query that names a phase and a review
          can match both strings in the same chunk.
        """
        lines = []

        # Title and type
        title = desc.get("title", "").strip()
        dtype = desc.get("diagram_type", "diagram")
        caption = (desc.get("caption", "") or "").strip()
        if title:
            lines.append(f"Diagram: {title}")
        else:
            lines.append(f"Diagram type: {dtype}")

        if caption:
            lines.append(f"Caption: {caption}")
        lines.append(f"Page: {page_num + 1}")

        # Summary — most important for broad queries
        summary = desc.get("summary", "")
        if summary:
            lines.append(f"Description: {summary}")

        # Nodes — important for "what stages/steps are shown?"
        nodes = desc.get("nodes", [])
        if nodes:
            lines.append(f"Stages/components: {', '.join(nodes)}")

        # Connections — important for "what leads to what?"
        connections = desc.get("connections", []) or desc.get("flow", [])
        if connections:
            lines.append("Connections:")
            for step in connections:
                lines.append(f"  {step}")

        # Tables (if table image)
        tables = desc.get("tables", [])
        if tables:
            lines.append(f"Table content (image): {', '.join(tables)}")

        # Annotations
        annotations = desc.get("annotations", [])
        if annotations:
            lines.append(f"Annotations: {', '.join(annotations)}")

        # References
        refs = desc.get("refs", []) or desc.get("external_refs", [])
        if refs:
            lines.append(f"References: {', '.join(refs)}")

        # Key concepts — improves semantic search coverage
        concepts = desc.get("key_terms", []) or desc.get("key_concepts", [])
        if concepts:
            lines.append(f"Key terms: {', '.join(concepts)}")

        return "\n".join(lines)

    def _build_chunk(self, description: dict, chunk_text: str,
                     page_num: int, image_hash: str) -> Chunk:
        """Build a Chunk object from the serialized diagram description."""
        self._chunk_counter += 1
        caption = (description.get("caption") or "").strip()
        if caption:
            m = re.search(r"\b(?:figure|fig\.?|table|chart|diagram)\s+([\w\d]+(?:[.\-][\w\d]+)*)",
                          caption, re.IGNORECASE)
            if m:
                fig_num = m.group(1).lower().replace(".", "-")
                chunk_id = f"fig_{fig_num}_p{page_num+1:03d}"
            else:
                chunk_id = f"diag_p{page_num:03d}_{self._chunk_counter:03d}"
        else:
            chunk_id = f"diag_p{page_num:03d}_{self._chunk_counter:03d}"

        title = (
            description.get("title", "")
            or caption
            or description.get("diagram_type", "diagram")
        )

        return Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            chunk_type="diagram",
            metadata={
                "chunk_id":    chunk_id,
                "page_start":  page_num,
                "page_end":    page_num,
                "section":     None,
                "type":        "diagram",
                "diagram_type": description.get("diagram_type", "unknown"),
                "title":       title,
                "caption":     caption,
                "table_id":    None,
                "is_parent":   False,
                "amendment_version": None,
                "superseded_by":     None,
                "token_count": len(chunk_text.split()),  # approximate
                "image_hash":  image_hash,
                # Store refs for cross-ref graph wiring
                "external_refs": (description.get("refs", []) or description.get("external_refs", [])),
                "reviews_mentioned": description.get("reviews", [])
            }
        )


# ---------------------------------------------------------------------------
# Helper: print a description nicely (for debugging)
# ---------------------------------------------------------------------------

def print_description(desc: dict, page_num: int):
    print(f"\n{'='*50}")
    print(f"Page {page_num + 1} | Type: {desc.get('diagram_type')}")
    print(f"Title: {desc.get('title', '(no title)')}")
    print(f"Summary: {desc.get('summary', '')}")
    print(f"Nodes ({len(desc.get('nodes',[]))}): {desc.get('nodes', [])}")
    print(f"Flow ({len(desc.get('flow',[]))}): {desc.get('flow', [])}")
    print(f"Reviews: {desc.get('reviews', [])}")
    print(f"Refs: {desc.get('external_refs', [])}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Test — run standalone on just the first 60 pages (where most diagrams are)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from extractor import PDFExtractor

    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    _default_pdf = os.path.join(
        _project_root, "data", "nasa_systems_engineering_handbook_0.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else _default_pdf

    MAX_TEST_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 80

    print(f"=== DIAGRAM EXTRACTION TEST (page-render mode) ===")
    print(f"PDF:        {pdf_path}")
    print(f"Max pages:  {MAX_TEST_PAGES}")

    # Minimal page extraction (just need page count for extract_all)
    extractor = PDFExtractor(pdf_path)
    pages = extractor.extract_all_pages(end_page=MAX_TEST_PAGES)
    extractor.close()

    # Run diagram extraction — page-render mode captures vector diagrams
    diagram_extractor = DiagramExtractor(
        cache_db=os.path.join(_project_root, "db", "diagram_cache.db")
    )
    chunks = diagram_extractor.extract_all(pages, pdf_path=pdf_path)

    print(f"\n=== RESULTS ===")
    print(f"Diagram chunks created: {len(chunks)}")

    for chunk in chunks:
        print(f"\nChunk: {chunk.chunk_id}")
        print(f"  Page:     {chunk.metadata['page_start'] + 1}")
        print(f"  Type:     {chunk.metadata['diagram_type']}")
        print(f"  Caption:  {chunk.metadata.get('caption', '')}")
        print(f"  Title:    {chunk.metadata['title']}")
        print(f"  Text preview:")
        for line in chunk.text.split("\n")[:8]:
            print(f"    {line}")
        if chunk.metadata.get("external_refs"):
            print(f"  Refs:     {chunk.metadata['external_refs']}")

    print("\nDone. These chunks exist only in memory — the API searches db/faiss.index.")
    print("To make diagram text retrievable, from project root run:")
    print("  python src/pipeline.py --reindex")
    print("(Diagram cache is reused — vision calls only happen for new/changed pages.)")
    print("Cost note: ~$0.05-0.10 per *new* diagram page (rendered PNG is large).")