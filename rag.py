#!/usr/bin/env python3
"""
rag.py
======
A retrieval-augmented generation pipeline over heterogeneous biomedical
evidence: unstructured paper text AND structured model-output tables
(gene rankings, convergence results). Answers are grounded in retrieved
chunks and returned with explicit source citations.

Design
------
1. INGEST    : load .txt/.md papers and .csv/.tsv result tables; turn each into
               text "chunks", each tagged with a source id and a type
               ("paper" | "table_row").
2. EMBED     : sentence-transformers embeddings (all-MiniLM-L6-v2 by default;
               small, fast, runs on CPU).
3. INDEX     : an in-memory vector store (cosine similarity via numpy). No
               external DB needed; swap in FAISS/Chroma for scale.
4. RETRIEVE  : top-k chunks for a query.
5. GENERATE  : build a grounded prompt that instructs the LLM to answer ONLY
               from the retrieved context and to cite the chunk ids it used;
               refuse if the context is insufficient.

Why heterogeneous retrieval: a question like "which prioritised genes are
ageing-causal and what is the evidence?" is best answered by combining the
model's *structured* output (which genes, what scores) with the *paper text*
(why, with what caveats). Mixing both in one index lets one query draw on both.

Honest scope: this is a research prototype. Retrieval is dense-only (no
re-ranking), the store is in-memory, and groundedness is enforced by prompt
plus citation-checking, not by a separate verifier model.
"""

from __future__ import annotations
import os, glob, re, json
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    id: str            # e.g. "paper:mr_gat#3" or "table:convergent#CTSF"
    text: str
    source: str        # human-readable source name
    kind: str          # "paper" | "table_row"
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def _chunk_paragraphs(text: str, target_chars: int = 800):
    """Split on blank lines, then greedily pack paragraphs to ~target_chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= target_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def load_papers(papers_dir: str) -> list[Chunk]:
    """Load .txt/.md files as paper chunks."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(papers_dir, "*"))):
        if not path.lower().endswith((".txt", ".md")):
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for i, ch in enumerate(_chunk_paragraphs(text)):
            chunks.append(Chunk(
                id=f"paper:{name}#{i}",
                text=ch,
                source=f"{name} (paper)",
                kind="paper",
                meta={"file": os.path.basename(path), "chunk": i},
            ))
    return chunks


def load_tables(tables_dir: str) -> list[Chunk]:
    """
    Load .csv/.tsv result tables. Each ROW becomes a chunk verbalised as
    "col1: val1; col2: val2; ...", so structured rows are retrievable by the
    same dense index as the paper text. Expects a header row.
    """
    import csv
    chunks = []
    for path in sorted(glob.glob(os.path.join(tables_dir, "*"))):
        if not path.lower().endswith((".csv", ".tsv")):
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        delim = "\t" if path.lower().endswith(".tsv") else ","
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delim)
            # use the first column as a row key if it looks like a gene/id
            key_col = reader.fieldnames[0] if reader.fieldnames else "row"
            for i, row in enumerate(reader):
                key = (row.get(key_col) or str(i)).strip()
                verbal = "; ".join(f"{k}: {v}" for k, v in row.items() if v not in (None, ""))
                text = f"[{name}] {verbal}"
                chunks.append(Chunk(
                    id=f"table:{name}#{key}",
                    text=text,
                    source=f"{name} (results table)",
                    kind="table_row",
                    meta={"file": os.path.basename(path), "row": i, "key": key},
                ))
    return chunks


# ---------------------------------------------------------------------------
# Embedding + index
# ---------------------------------------------------------------------------
class DenseIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self.embs: Optional[np.ndarray] = None

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embs = self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=True, batch_size=64)
        self.embs = np.asarray(embs, dtype=np.float32)
        return self

    def search(self, query: str, k: int = 6, kind: Optional[str] = None):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.embs @ q  # cosine (vectors are normalized)
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

    # simple persistence so you don't re-embed every run
    def save(self, path: str):
        np.savez(path + ".npz", embs=self.embs)
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump([c.__dict__ for c in self.chunks], f)

    def load(self, path: str):
        self.embs = np.load(path + ".npz")["embs"]
        with open(path + ".json", encoding="utf-8") as f:
            self.chunks = [Chunk(**d) for d in json.load(f)]
        return self


# ---------------------------------------------------------------------------
# Grounded generation
# ---------------------------------------------------------------------------
GROUNDED_SYSTEM = (
    "You are a careful biomedical research assistant. Answer ONLY using the "
    "provided context passages. Each passage has an id in [brackets]. Cite the "
    "ids you use, in square brackets, after the sentences they support. If the "
    "context does not contain enough information to answer, say so plainly "
    "rather than guessing. Do not introduce facts not present in the context."
)


def build_prompt(question: str, hits) -> str:
    ctx = "\n\n".join(f"[{c.id}] {c.text}" for _, c in hits)
    return (
        f"Context passages:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Answer (grounded in and citing the passages above):"
    )


class RAG:
    def __init__(self, index: DenseIndex, llm):
        self.index = index
        self.llm = llm

    def answer(self, question: str, k: int = 6, mix: bool = True):
        """
        Retrieve top-k and generate a grounded, cited answer.
        If mix=True, ensure both a paper chunk and a table row are represented
        when available, so structured + unstructured evidence are combined.
        """
        hits = self.index.search(question, k=k)
        if mix:
            kinds = {c.kind for _, c in hits}
            for need in ("paper", "table_row"):
                if need not in kinds:
                    extra = self.index.search(question, k=2, kind=need)
                    hits += extra[:1]
        prompt = build_prompt(question, hits)
        answer = self.llm.complete(prompt, system=GROUNDED_SYSTEM)
        return {
            "question": question,
            "answer": answer,
            "sources": [{"id": c.id, "source": c.source, "score": round(s, 3)}
                        for s, c in hits],
        }
