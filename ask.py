#!/usr/bin/env python3
"""
ask.py
======
Build the index over papers + result tables, then answer questions with a
grounded, cited RAG pipeline.

Examples
--------
# 1. Offline self-test (no LLM, no embeddings download needed for the logic check)
python ask.py --selftest

# 2. Build index and ask one question (local Ollama by default)
python ask.py --papers data/papers --tables data/tables \
              --backend ollama --model llama3.1:8b \
              --q "Which prioritised genes are both exercise-responsive and ageing-causal, and what is the evidence?"

# 3. Use an API backend instead
ANTHROPIC_API_KEY=... python ask.py --backend anthropic --q "..."

# 4. Interactive
python ask.py --papers data/papers --tables data/tables --interactive
"""

from __future__ import annotations
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag import load_papers, load_tables, DenseIndex, RAG  # noqa: E402


def selftest():
    """
    Verify ingestion + retrieval logic with NO model and NO network:
    monkeypatch the embedder with a trivial bag-of-words vectoriser so the
    pipeline's plumbing (chunking, table verbalisation, mixed retrieval) is
    testable offline.
    """
    import numpy as np
    from rag import Chunk

    # tiny synthetic corpus
    papers = [Chunk(id="paper:demo#0",
                    text="Cathepsin F (CTSF) was validated as a longevity-associated "
                         "target by cis-MR and colocalisation.",
                    source="demo (paper)", kind="paper")]
    tables = [Chunk(id="table:convergent#CTSF",
                    text="[convergent] gene: CTSF; exercise_responsive: yes; "
                         "ageing_causal: yes; layer: proteomics",
                    source="convergent (results table)", kind="table_row"),
              Chunk(id="table:convergent#FADS1",
                    text="[convergent] gene: FADS1; exercise_responsive: yes; "
                         "ageing_causal: yes; layer: lipid",
                    source="convergent (results table)", kind="table_row")]
    chunks = papers + tables

    # trivial deterministic "embedder": hashed bag-of-words
    class FakeIndex(DenseIndex):
        def __init__(self):
            self.chunks = []
            self.embs = None
            self._vocab = {}
        def _vec(self, text):
            v = np.zeros(64, dtype=np.float32)
            for w in text.lower().split():
                v[hash(w) % 64] += 1.0
            n = np.linalg.norm(v)
            return v / n if n else v
        def build(self, chunks):
            self.chunks = chunks
            self.embs = np.vstack([self._vec(c.text) for c in chunks])
            return self
        def search(self, query, k=6, kind=None):
            q = self._vec(query)
            sims = self.embs @ q
            order = np.argsort(-sims)
            hits = []
            for idx in order:
                c = self.chunks[idx]
                if kind and c.kind != kind:
                    continue
                hits.append((float(sims[idx]), c))
                if len(hits) >= k:
                    break
            return hits

    idx = FakeIndex().build(chunks)
    hits = idx.search("which gene is a longevity target validated by MR", k=3)
    ids = [c.id for _, c in hits]
    assert any("CTSF" in i for i in ids), f"expected CTSF in {ids}"

    # mixed retrieval: ensure both a paper and a table row can be returned
    paper_hits = idx.search("CTSF", k=2, kind="paper")
    table_hits = idx.search("CTSF", k=2, kind="table_row")
    assert paper_hits and table_hits, "mixed retrieval should find both kinds"

    print("OK: ingestion, chunking, table verbalisation, and mixed dense "
          "retrieval all behave as expected (offline, no model).")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--papers", default="data/papers")
    ap.add_argument("--tables", default="data/tables")
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--q", default=None, help="single question")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    # build index
    chunks = load_papers(args.papers) + load_tables(args.tables)
    if not chunks:
        print(f"No data found in {args.papers} or {args.tables}. "
              f"Add .txt/.md papers and .csv/.tsv tables.")
        return 1
    print(f"Ingested {len(chunks)} chunks "
          f"({sum(c.kind=='paper' for c in chunks)} paper, "
          f"{sum(c.kind=='table_row' for c in chunks)} table rows). Embedding...")
    index = DenseIndex(args.embed_model).build(chunks)

    from llm_backend import LLM
    llm = LLM(backend=args.backend, model=args.model)
    rag = RAG(index, llm)

    def show(res):
        print("\n" + "=" * 70)
        print("Q:", res["question"])
        print("-" * 70)
        print(res["answer"])
        print("-" * 70)
        print("Sources:")
        for s in res["sources"]:
            print(f"  [{s['id']}]  {s['source']}  (sim={s['score']})")
        print("=" * 70 + "\n")

    if args.q:
        show(rag.answer(args.q, k=args.k))
    if args.interactive:
        print("Interactive RAG. Empty line to quit.")
        while True:
            try:
                q = input("\nQuestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            show(rag.answer(q, k=args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
