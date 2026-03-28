"""
validator.py — Table quality filter
------------------------------------
Filters out false positive tables that PyMuPDF incorrectly
detects from redacted sections, paragraphs, and noise.

Called after reconstruction, before chunking.
"""


class TableValidator:
    """
    Filters reconstructed tables before they enter the vector store.

    WHY this exists:
      PyMuPDF detects tables by spatial alignment of text blocks.
      Some regulatory PDFs use redacted placeholders like "(b)(4)" in 2-column layouts
      and get flagged as tables. We need to catch these before embedding
      because a false table in the vector store will be retrieved
      and confuse the LLM.
    """

    # Headers that indicate redacted/non-table content
    REDACTION_MARKERS = ["(b) (4)", "(b)(4)", "(b) (6)", "(b)(6)"]

    def validate_all(self, tables: list) -> tuple:
        """
        Returns (valid_tables, rejected_tables)
        So we can log what was rejected and why.
        """
        valid = []
        rejected = []

        for table in tables:
            passed, reason = self._validate_single(table)
            if passed:
                valid.append(table)
            else:
                rejected.append((table, reason))
                print(f"  Rejected table {table.table_id}: {reason}")

        return valid, rejected

    def _validate_single(self, table) -> tuple:
        """
        Returns (True, "") if valid, (False, reason) if rejected.

        Checks in order of cheapest to most expensive.
        """

        # Check 1: Minimum rows
        # A real data table has at least 1 header + 1 data row
        if len(table.all_rows) < 1:
            return False, "No data rows"

        # Check 2: Redaction markers in headers
        # Some regulatory submissions use (b)(4) for confidential content
        # These get misdetected as table headers
        for header_cell in table.headers:
            cell_text = str(header_cell).strip()
            if cell_text in self.REDACTION_MARKERS:
                return False, f"Redaction marker in header: {cell_text}"

        # Check 3: Too many empty cells
        # Real tables have data. False tables are mostly empty.
        all_cells = []
        for row in table.all_rows:
            all_cells.extend(row)

        if all_cells:
            empty_count = sum(
                1 for c in all_cells
                if c is None or str(c).strip() == ""
            )
            empty_ratio = empty_count / len(all_cells)
            if empty_ratio > 0.7:
                return False, f"Too many empty cells ({empty_ratio:.0%})"

        # Check 4: Single column tables are usually paragraphs
        if table.col_count < 2:
            return False, "Single column — likely a paragraph"

        # Check 5: Headers look like sentences (not column names)
        # Real column headers are short (< 40 chars average)
        if table.headers:
            non_empty_headers = [
                h for h in table.headers
                if h and str(h).strip()
            ]
            if non_empty_headers:
                avg_header_len = sum(
                    len(str(h)) for h in non_empty_headers
                ) / len(non_empty_headers)

                if avg_header_len > 60:
                    return False, f"Headers too long — likely paragraph text"

        # Check 6: Every row is the NASA handbook page-footer strip (extractor miss)
        if table.all_rows and all(
            self._row_looks_like_handbook_footer(r) for r in table.all_rows
        ):
            return False, "Handbook page-footer only — not tabular content"

        return True, ""

    @staticmethod
    def _row_looks_like_handbook_footer(row: list) -> bool:
        parts = [str(c or "").strip() for c in (row or [])]
        j = " ".join(parts).upper()
        return (
            "NASA" in j
            and "SYSTEMS ENGINEERING" in j
            and "HANDBOOK" in j
        )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from extractor import PDFExtractor
    from reconstructor import TableReconstructor

    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    _default_pdf = os.path.join(
        _project_root, "data", "nasa_systems_engineering_handbook_0.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else _default_pdf

    print("=== Extracting ===")
    extractor = PDFExtractor(pdf_path)
    pages = extractor.extract_all_pages(end_page=20)
    extractor.close()

    print("\n=== Reconstructing ===")
    reconstructor = TableReconstructor(db_path="db/table_buffer.db")
    tables = reconstructor.reconstruct(pages)

    print(f"\n=== Validating {len(tables)} tables ===")
    validator = TableValidator()
    valid_tables, rejected_tables = validator.validate_all(tables)

    print(f"\nResults:")
    print(f"  Valid tables:    {len(valid_tables)}")
    print(f"  Rejected tables: {len(rejected_tables)}")

    print(f"\n--- Valid Tables ---")
    for t in valid_tables:
        print(f"  {t.table_id}: '{t.title}' | "
              f"pages={t.pages} | rows={len(t.all_rows)}")

    print(f"\n--- Rejected Tables ---")
    for t, reason in rejected_tables:
        print(f"  {t.table_id}: REJECTED — {reason}")