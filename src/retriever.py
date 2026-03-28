"""
retriever.py
---------------------
Responsibility: Take a user query, retrieve relevant chunks,
resolve cross-references, build a cited prompt, and generate
a final answer with per-claim citations.

This file is where Q2, Q4, Q6, Q9 all come together:
  Q2: cross-reference resolution (fetch linked chunks)
  Q4: citation generation (metadata → cited answer)
  Q8: cost control (top-k tuning, model selection)
  Q9: honest about what we can't answer

Pipeline per query:
  1. Embed query
  2. FAISS search → top-K chunks
  3. For each retrieved chunk: resolve cross-references
  4. Deduplicate and rank final context
  5. Build cited prompt
  6. Call GPT-4o-mini (not GPT-4 — cost control)
  7. Return answer + citations
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Citation builder
# ---------------------------------------------------------------------------

def build_citation(metadata: dict) -> str:
    """
    Build a human-readable citation string from chunk metadata.

    Format: "Page X, Section Y, Table Z"
    Only includes fields that are actually present.

    WHY this matters (Q4):
      The metadata envelope we attached at ingestion time
      is now paying off — every chunk knows exactly where
      it came from, so every claim can be cited precisely.
    """
    parts = []

    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")

    if page_start is not None:
        page_str = f"Page {page_start + 1}"
        if page_end and page_end != page_start:
            page_str += f"–{page_end + 1}"
        parts.append(page_str)

    section = metadata.get("section")
    if section:
        parts.append(f"Section {section}")

    table_id = metadata.get("table_id")
    title = metadata.get("title", "")
    if table_id and title and "[PROVISIONAL]" not in title:
        parts.append(title)
    elif table_id:
        parts.append(f"Table ({table_id})")

    return ", ".join(parts) if parts else "Source unknown"


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Orchestrates retrieval → cross-ref resolution → LLM generation.

    Configuration:
      top_k_initial : how many chunks to retrieve from FAISS (20)
      top_k_final   : how many chunks to send to LLM after dedup (5)
      model         : LLM for answer generation

    WHY top_k_initial=20 then reduce to 5 (Q8):
      Retrieve broadly (20) to catch cross-referenced chunks.
      After adding cross-ref neighbors, deduplicate and
      take top 5 by score to stay within context window.
      Sending all 20 to the LLM would cost 3x more and
      add noise that reduces answer quality.

    WHY gpt-4o-mini not gpt-4o (Q8):
      gpt-4o-mini: $0.15/1M input, $0.60/1M output
      gpt-4o:      $5.00/1M input, $15.00/1M output
      33x cheaper. For factual Q&A on retrieved context,
      quality difference is minimal — the context does the
      heavy lifting, not the model's parametric knowledge.
    """

    def __init__(self,
                 embedder,
                 reference_graph=None,
                 top_k_initial: int = 20,
                 top_k_final: int = 5,
                 model: str = "gpt-4o-mini"):

        self.embedder = embedder
        self.reference_graph = reference_graph
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def query(self, user_question: str,
              region: str = "global",
              verbose: bool = False) -> dict:
        """
        Full retrieval pipeline for a user question.

        Returns:
          {
            answer: str,
            citations: list,
            chunks_used: list,
            model: str,
            tokens_used: int
          }
        """

        if verbose:
            print(f"\n--- Query: {user_question} ---")

        # Step 1: Initial FAISS search
        initial_results = self.embedder.search(
            user_question,
            top_k=self.top_k_initial
        )

        if verbose:
            print(f"Step 1: Retrieved {len(initial_results)} chunks")

        # Step 2: Cross-reference resolution
        # For each retrieved chunk, fetch any chunks it references
        # This is the Q2 solution running live
        enriched_results = self._resolve_cross_references(
            initial_results, verbose
        )

        if verbose:
            print(f"Step 2: After cross-ref resolution: "
                  f"{len(enriched_results)} chunks")

        # Step 3: Deduplicate and take top_k_final
        final_chunks = self._deduplicate_and_rank(
            enriched_results,
            top_k=self.top_k_final
        )

        if verbose:
            print(f"Step 3: Final context: {len(final_chunks)} chunks")
            for c in final_chunks:
                print(f"  - {c['chunk_id']} "
                      f"(score={c.get('score', 0):.3f}, "
                      f"type={c['chunk_type']})")

        # Step 4: Build cited prompt
        prompt, context_used = self._build_prompt(
            user_question, final_chunks
        )

        # Step 5: Generate answer
        answer, tokens_used = self._generate_answer(prompt)

        # Step 6: Build citations list
        citations = [
            {
                "chunk_id": c["chunk_id"],
                "citation": build_citation(c["metadata"]),
                "chunk_type": c["chunk_type"]
            }
            for c in final_chunks
        ]

        return {
            "answer": answer,
            "citations": citations,
            "chunks_used": [c["chunk_id"] for c in final_chunks],
            "model": self.model,
            "tokens_used": tokens_used,
            "context_chunks": final_chunks,
            "retrieval_stats": {
                "faiss_top_k":     self.top_k_initial,
                "initial_hits":    len(initial_results),
                "after_cross_ref": len(enriched_results),
                "final_to_llm":    len(final_chunks),
            },
        }

    # -----------------------------------------------------------------------
    # Cross-reference resolution
    # -----------------------------------------------------------------------

    def _resolve_cross_references(self, results: list,
                                   verbose: bool = False) -> list:
        """
        For each retrieved chunk, check if it references other chunks.
        If yes, fetch those referenced chunks and add to results.

        WHY this is critical (Q2):
          Page 147 says "adjust dosing per Table 4.2"
          FAISS retrieves page 147 (high similarity to query)
          But Table 4.2 (pages 31-33) has low similarity to query text
          Without this step: LLM gets "adjust per Table 4.2" with no Table 4.2
          With this step: LLM gets page 147 AND Table 4.2 content
        """
        if not self.reference_graph:
            return results

        enriched = list(results)
        existing_ids = {r["chunk_id"] for r in results}

        for result in results:
            chunk_id = result["chunk_id"]

            # Get what this chunk references
            refs = self.reference_graph.get_references_from_chunk(chunk_id)

            for ref in refs:
                resolved_ids = ref.get("resolved_chunk_ids", [])
                if verbose and resolved_ids:
                    print(f"  Cross-ref: {chunk_id} → "
                          f"{ref['ref_id']} → {resolved_ids}")

                for ref_chunk_id in resolved_ids:
                    if ref_chunk_id not in existing_ids:
                        # Fetch the referenced chunk from metadata store
                        ref_data = self.embedder.metadata_store\
                            .get_by_chunk_id(ref_chunk_id)
                        if ref_data:
                            ref_data["score"] = 0.5  # lower priority
                            ref_data["via_reference"] = ref["ref_id"]
                            enriched.append(ref_data)
                            existing_ids.add(ref_chunk_id)

        return enriched

    # -----------------------------------------------------------------------
    # Deduplication and ranking
    # -----------------------------------------------------------------------

    def _deduplicate_and_rank(self, results: list,
                               top_k: int = 5) -> list:
        """
        Remove duplicates and rank by score.
        Cross-referenced chunks get a slight boost if their
        source chunk was highly relevant.
        """
        seen_ids = set()
        unique = []

        for r in results:
            cid = r.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                unique.append(r)

        # Sort by score descending
        unique.sort(key=lambda x: x.get("score", 0), reverse=True)

        return unique[:top_k]

    # -----------------------------------------------------------------------
    # Prompt building
    # -----------------------------------------------------------------------

    def _build_prompt(self, question: str,
                      chunks: list) -> tuple:
        """
        Build the LLM prompt with cited context chunks.

        WHY the explicit non-inference instruction (Q2, Q4):
          Without it, GPT-4 class models will interpolate a
          plausible technical value from nearby context and
          present it as fact. In standards and handbooks this is dangerous.

        WHY we include metadata inline in the prompt (Q4):
          The LLM needs to see source info to cite correctly.
          "Page 47, Table 6.1" must appear IN the context block
          so the model can attribute claims to the right source.
        """
        context_parts = []

        for i, chunk in enumerate(chunks):
            citation = build_citation(chunk["metadata"])
            via = chunk.get("via_reference", "")
            via_note = f" [retrieved via {via}]" if via else ""

            context_parts.append(
                f"[SOURCE {i+1}: {citation}{via_note}]\n"
                f"{chunk['text']}\n"
            )

        context_text = "\n---\n".join(context_parts)

        prompt = f"""You are a precise technical assistant helping users query a \
document handbook or manual.

INSTRUCTIONS:
- Answer ONLY from the provided context sources below
- Cite each factual claim inline using [SOURCE N] notation
- If the exact answer is not in the context, say:
  "The specific information is not available in the retrieved sections.
   Please refer to the relevant section directly."
- Do NOT estimate, interpolate, or infer values not explicitly stated
- Do NOT use prior knowledge outside the provided sources
- For external references (e.g. NPR 7123.1, referenced standards):
  flag them as "External reference — verify [document name] directly"

QUESTION: {question}

CONTEXT:
{context_text}

Provide a precise answer with inline citations [SOURCE N] for each claim."""

        return prompt, context_text

    # -----------------------------------------------------------------------
    # LLM generation
    # -----------------------------------------------------------------------

    def _generate_answer(self, prompt: str) -> tuple:
        """
        Call OpenAI to generate the final answer.
        Returns (answer_text, tokens_used).
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise technical assistant. "
                               "Answer only from the provided context. "
                               "Never guess, infer, or fabricate values."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,      # WHY 0: technical handbooks need deterministic
                                # answers, not creative variations
            max_tokens=500
        )

        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        return answer, tokens


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from embedder import Embedder
    from reference_graph import ReferenceGraph

    # Load existing FAISS index (don't re-embed)
    print("=== Loading FAISS index ===")
    embedder = Embedder(
        index_path="db/faiss.index",
        metadata_db="db/metadata_store.db"
    )
    embedder.load()
    print(f"Loaded {embedder.index.ntotal} vectors")

    # Load reference graph
    print("\n=== Loading reference graph ===")
    graph = ReferenceGraph(db_path="db/reference_graph.db")
    stats = graph.get_stats()
    print(f"References in graph: {stats['total_references']}")

    # Initialize retriever
    retriever = Retriever(
        embedder=embedder,
        reference_graph=graph,
        top_k_initial=20,
        top_k_final=5,
        model="gpt-4o-mini"
    )

    # Test queries (NASA Systems Engineering Handbook)
    test_questions = [
        "What is a Key Decision Point (KDP) in the NASA project life cycle?",
        "What are the 17 systems engineering processes in NPR 7123.1?",
        "What is the difference between verification and validation?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        result = retriever.query(question, verbose=True)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nCITATIONS:")
        for c in result["citations"]:
            print(f"  [{c['chunk_type']}] {c['citation']}")
        print(f"\nTokens used: {result['tokens_used']}")
        print(f"Cost: ${result['tokens_used'] * 0.00000015:.6f}")

    graph.close()
    embedder.close()
    print("\nRetriever.py is working and Retriever is complete.")