"""
embedder.py — Day 5 (part 2)
-----------------------------
Responsibility: Take Chunk objects from chunker.py,
embed them using OpenAI text-embedding-3-small,
and store vectors + metadata in FAISS index.

WHY FAISS over ChromaDB on Windows:
  ChromaDB requires C++ build tools (chroma-hnswlib compilation).
  FAISS ships pre-built Windows wheels — installs in 30 seconds.
  Same concept: vector similarity search. Different install path.

WHY text-embedding-3-small over ada-002:
  3-small: $0.02/1M tokens, better quality than ada-002
  ada-002: $0.10/1M tokens, older model
  For 434 chunks at avg 200 tokens = 86,800 tokens
  Cost: $0.0017 — essentially free for our document size.

WHY we store metadata separately (not in FAISS):
  FAISS stores vectors only — no metadata.
  We store metadata in a parallel SQLite DB keyed by vector index.
  At retrieval: FAISS gives us vector indices → SQLite gives metadata.
  This pattern works for any vector store (Pinecone, Weaviate too).
"""

import os
import json
import pickle
import sqlite3
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


# ---------------------------------------------------------------------------
# Metadata store (parallel SQLite for FAISS)
# ---------------------------------------------------------------------------

class MetadataStore:
    """
    Stores chunk metadata keyed by FAISS vector index position.

    WHY keyed by position (integer) not chunk_id (string):
      FAISS returns integer indices (0, 1, 2...) not string IDs.
      We need the integer → metadata mapping for retrieval.
      We also store chunk_id → integer for reverse lookup.
    """

    def __init__(self, db_path: str = "db/metadata_store.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunk_metadata (
                faiss_idx   INTEGER PRIMARY KEY,
                chunk_id    TEXT UNIQUE,
                chunk_type  TEXT,
                text        TEXT,
                metadata    TEXT    -- full JSON metadata envelope
            );
            CREATE INDEX IF NOT EXISTS idx_chunk_id
                ON chunk_metadata(chunk_id);
        """)
        self.conn.commit()

    def store(self, faiss_idx: int, chunk_id: str,
              chunk_type: str, text: str, metadata: dict):
        self.conn.execute("""
            INSERT OR REPLACE INTO chunk_metadata
            (faiss_idx, chunk_id, chunk_type, text, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (faiss_idx, chunk_id, chunk_type, text,
               json.dumps(metadata)))
        self.conn.commit()

    def get_by_faiss_idx(self, faiss_idx: int) -> dict:
        """Get full chunk data by FAISS index position."""
        row = self.conn.execute("""
            SELECT chunk_id, chunk_type, text, metadata
            FROM chunk_metadata WHERE faiss_idx = ?
        """, (faiss_idx,)).fetchone()

        if not row:
            return None
        return {
            "chunk_id": row[0],
            "chunk_type": row[1],
            "text": row[2],
            "metadata": json.loads(row[3])
        }

    def get_by_chunk_id(self, chunk_id: str) -> dict:
        """Get full chunk data by chunk ID string."""
        row = self.conn.execute("""
            SELECT faiss_idx, chunk_type, text, metadata
            FROM chunk_metadata WHERE chunk_id = ?
        """, (chunk_id,)).fetchone()

        if not row:
            return None
        return {
            "faiss_idx": row[0],
            "chunk_type": row[1],
            "text": row[2],
            "metadata": json.loads(row[3])
        }

    def get_total_chunks(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM chunk_metadata"
        ).fetchone()[0]

    def clear(self):
        self.conn.execute("DELETE FROM chunk_metadata")
        self.conn.commit()

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """
    Embeds chunks and builds FAISS index.

    Index type: IndexFlatIP (inner product = cosine similarity
    when vectors are normalized).

    WHY IndexFlatIP over IndexIVFFlat:
      IndexFlatIP: exact search, no training needed, perfect for
      <100k vectors. Our 434 chunks fit easily.
      IndexIVFFlat: approximate search, needs training, better
      for 1M+ vectors. Overkill here.
    """

    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536        # dimension for text-embedding-3-small
    BATCH_SIZE = 50             # embed 50 chunks per API call
                                # WHY: OpenAI allows up to 2048 per call
                                # but smaller batches are safer on rate limits

    def __init__(self,
                 index_path: str = "db/faiss.index",
                 metadata_db: str = "db/metadata_store.db"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.index_path = index_path
        self.metadata_store = MetadataStore(metadata_db)

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self._vector_count = 0

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _embed_batch(self, texts: list) -> np.ndarray:
        """
        Call OpenAI embedding API for a batch of texts.

        WHY @retry with exponential backoff:
          OpenAI rate limits at ~3000 RPM on free tier.
          On a 434-chunk document this rarely triggers,
          but on a 1GB doc with 5000+ chunks it will.
          Tenacity retries automatically with increasing delays.
        """
        response = self.client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=texts
        )
        vectors = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32
        )
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)
        return vectors

    def embed_chunks(self, chunks: list) -> int:
        """
        Embed all chunks and add to FAISS index.
        Returns total chunks embedded.
        """
        print(f"Embedding {len(chunks)} chunks...")
        self.metadata_store.clear()

        total_embedded = 0

        # Process in batches
        for batch_start in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[batch_start: batch_start + self.BATCH_SIZE]
            texts = [c.text for c in batch]

            print(f"  Batch {batch_start//self.BATCH_SIZE + 1}/"
                  f"{(len(chunks)-1)//self.BATCH_SIZE + 1} "
                  f"({len(batch)} chunks)...", end="\r")

            # Get embeddings
            vectors = self._embed_batch(texts)

            # Add to FAISS
            self.index.add(vectors)

            # Store metadata for each chunk in this batch
            for i, chunk in enumerate(batch):
                faiss_idx = self._vector_count + i
                self.metadata_store.store(
                    faiss_idx=faiss_idx,
                    chunk_id=chunk.chunk_id,
                    chunk_type=chunk.chunk_type,
                    text=chunk.text,
                    metadata=chunk.metadata
                )

            self._vector_count += len(batch)
            total_embedded += len(batch)

        print(f"\nEmbedded {total_embedded} chunks total.")
        return total_embedded

    def save(self):
        """
        Persist FAISS index to disk.
        WHY: FAISS index lives in RAM during ingestion.
        We must save to disk for retrieval in a separate process.
        """
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved: {self.index_path}")
        print(f"Total vectors: {self.index.ntotal}")

    def load(self):
        """Load existing FAISS index from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"No index found at {self.index_path}. "
                f"Run ingestion pipeline first."
            )
        self.index = faiss.read_index(self.index_path)
        self._vector_count = self.index.ntotal
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def search(self, query_text: str, top_k: int = 5) -> list:
        """
        Search for most similar chunks to a query.
        Returns list of {chunk_id, score, text, metadata}.
        """
        # Embed the query
        query_vec = self._embed_batch([query_text])

        # Search FAISS
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for empty slots
                continue
            chunk_data = self.metadata_store.get_by_faiss_idx(int(idx))
            if chunk_data:
                chunk_data["score"] = float(score)
                results.append(chunk_data)

        return results

    def close(self):
        self.metadata_store.close()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from extractor import PDFExtractor
    from reconstructor import TableReconstructor
    from validator import TableValidator
    from chunker import Chunker
    from diagram_extractor import DiagramExtractor

    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    _default_pdf = os.path.join(
        _project_root, "data", "nasa_systems_engineering_handbook_0.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else _default_pdf

    # Full pipeline: extract → reconstruct → validate → chunk → diagram → embed
    print("=== Step 1: Extracting ===")
    extractor = PDFExtractor(pdf_path)
    pages = extractor.extract_all_pages(end_page=None)
    extractor.close()

    print("\n=== Step 2: Reconstructing tables ===")
    reconstructor = TableReconstructor(
        db_path=os.path.join(_project_root, "db", "table_buffer.db")
    )
    raw_tables = reconstructor.reconstruct(pages)

    print("\n=== Step 3: Validating ===")
    validator = TableValidator()
    valid_tables, _ = validator.validate_all(raw_tables)

    print("\n=== Step 4: Chunking ===")
    chunker = Chunker(max_tokens=400, overlap_tokens=80)
    table_chunks = chunker.chunk_tables(valid_tables)
    narrative_chunks = chunker.chunk_narrative(pages)

    print("\n=== Step 4.5: Extracting diagrams (cached — no re-charge for unchanged pages) ===")
    diagram_ext = DiagramExtractor(
        cache_db=os.path.join(_project_root, "db", "diagram_cache.db")
    )
    diagram_chunks = diagram_ext.extract_all(pages, pdf_path=pdf_path)
    print(f"Diagram chunks: {len(diagram_chunks)}")

    all_chunks = table_chunks + narrative_chunks + diagram_chunks
    print(f"\nTotal chunks: {len(all_chunks)}"
          f" (tables={len(table_chunks)}, narrative={len(narrative_chunks)}, diagrams={len(diagram_chunks)})")

    # Step 5: Embed
    print("\n=== Step 5: Embedding ===")
    embedder = Embedder(
        index_path="db/faiss.index",
        metadata_db="db/metadata_store.db"
    )
    embedder.embed_chunks(all_chunks)
    embedder.save()

    # Step 6: Test search
    print("\n=== Step 6: Test search ===")
    test_queries = [
        "What is the Systems Engineering Engine in NPR 7123.1?",
        "NASA project life cycle phases and KDPs",
        "difference between verification and validation",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = embedder.search(query, top_k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r['score']:.3f} | "
                  f"type={r['chunk_type']} | "
                  f"page={r['metadata'].get('page_start', '?')} | "
                  f"id={r['chunk_id']}")
            print(f"       {r['text'][:100]}...")

    embedder.close()
    print("\nEmbedder.py is working and FAISS index ready.")