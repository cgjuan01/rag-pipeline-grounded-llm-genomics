#!/usr/bin/env python3
"""
mcp_server.py
=============
Exposes the Ask-my-research pipeline over the Model Context Protocol (MCP), so
any MCP-capable client -- Claude Desktop, an agent framework, an IDE -- can
ground its own reasoning in this research instead of answering from memory.

Why this exists
---------------
`ask.py` answers questions itself: retrieve, then generate. An agent usually
wants something different -- the *evidence*, so it can reason over it, compare
it against other sources, and check that a claim is actually supported. This
server therefore exposes retrieval and citation-checking as first-class tools,
with generation as an optional extra rather than the main event.

Design notes
------------
* `search_evidence`, `get_passage` and `describe_index` need no LLM and no
  network. The same property that makes `ask.py --selftest` possible makes
  these tools cheap, deterministic and testable. The calling agent supplies
  the reasoning; this server supplies grounded, cited evidence.

* The pipeline's refusal behaviour is carried into the tool layer. When the
  best match falls below MIN_SCORE, `search_evidence` says so explicitly and
  returns nothing, rather than handing back weak passages that a model would
  likely treat as support. Refusing at the tool boundary is stricter than
  refusing in a prompt, because the caller never sees the weak context.

* `verify_citation` exists because grounded generation is only as good as its
  citations. It lets an agent check that a quoted id exists and that the
  passage really contains what was attributed to it -- the retrieval-side
  equivalent of the offline self-test.

* `answer_question` is the only tool that loads a generator, and it is
  deliberately last. It reuses the existing RAG class and backend selection
  unchanged.

Usage
-----
    pip install "mcp[cli]"

    # register with an MCP client (e.g. Claude Desktop config):
    #   "ask-my-research": {
    #     "command": "python",
    #     "args": ["/abs/path/to/mcp_server.py"],
    #     "env": {"PAPERS_DIR": "data/papers", "TABLES_DIR": "data/tables"}
    #   }

    python mcp_server.py            # stdio transport
    python mcp_server.py --selftest # exercise every tool, no model, no network

Environment
-----------
    PAPERS_DIR   default data/papers
    TABLES_DIR   default data/tables
    INDEX_PATH   optional; cache embeddings here to avoid re-embedding
    EMBED_MODEL  default all-MiniLM-L6-v2
    MIN_SCORE    default 0.25; below this, search refuses
    LLM_BACKEND  ollama | anthropic | openai | transformers  (answer_question only)
    LLM_MODEL    model name for the chosen backend
"""

from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

from rag import RAG, DenseIndex, load_papers, load_tables

PAPERS_DIR = os.environ.get("PAPERS_DIR", "data/papers")
TABLES_DIR = os.environ.get("TABLES_DIR", "data/tables")
INDEX_PATH = os.environ.get("INDEX_PATH") or None
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.25"))

mcp = FastMCP(
    "ask-my-research",
    instructions=(
        "Grounded retrieval over a set of research papers and "
        "structured model-output tables. Prefer search_evidence and cite the "
        "passage ids you use. If search_evidence reports grounded=false, say "
        "these documents do not support an answer rather than filling the gap "
        "from your own knowledge."
    ),
)

_index: DenseIndex | None = None


def _get_index() -> DenseIndex:
    """Build or load the index once, lazily, and reuse it for the process."""
    global _index
    if _index is not None:
        return _index

    idx = DenseIndex(EMBED_MODEL)
    if INDEX_PATH and os.path.exists(INDEX_PATH + ".npz"):
        idx.load(INDEX_PATH)
    else:
        chunks = load_papers(PAPERS_DIR) + load_tables(TABLES_DIR)
        if not chunks:
            raise RuntimeError(
                f"No documents found. Looked in PAPERS_DIR={PAPERS_DIR!r} for "
                f"*.txt/*.md and TABLES_DIR={TABLES_DIR!r} for *.csv/*.tsv."
            )
        idx.build(chunks)
        if INDEX_PATH:
            idx.save(INDEX_PATH)
    _index = idx
    return _index


def _fmt(score: float, chunk) -> dict:
    return {
        "id": chunk.id,
        "source": chunk.source,
        "kind": chunk.kind,
        "score": round(float(score), 3),
        "text": chunk.text,
    }


@mcp.tool()
def search_evidence(query: str, k: int = 6, kind: str | None = None) -> dict:
    """Retrieve passages supporting a query, with citable ids and no generation.

    Searches paper text and verbalised result-table rows in one index, so a
    single query draws on both prose and structured results.

    Args:
        query: What to look for, phrased as a question or topic.
        k: Number of passages to return (1-20).
        kind: Restrict to "paper" or "table_row"; omit for both.

    Returns:
        grounded=false with a reason when nothing clears the relevance floor,
        otherwise the matching passages with ids to cite.
    """
    if not query or not query.strip():
        return {"grounded": False, "reason": "Empty query.", "passages": []}
    if kind not in (None, "paper", "table_row"):
        return {"grounded": False,
                "reason": f"kind must be 'paper', 'table_row' or omitted; got {kind!r}.",
                "passages": []}

    k = max(1, min(int(k), 20))
    hits = _get_index().search(query, k=k, kind=kind)

    if not hits:
        return {"grounded": False, "reason": "Nothing has been indexed.",
                "passages": []}

    best = hits[0][0]
    if best < MIN_SCORE:
        return {
            "grounded": False,
            "reason": (
                f"Best match scored {best:.3f}, below the {MIN_SCORE} relevance "
                "floor. These documents do not appear to cover the query. Say so "
                "rather than answering from prior knowledge."
            ),
            "best_score": round(float(best), 3),
            "passages": [],
        }

    return {
        "grounded": True,
        "query": query,
        "passages": [_fmt(s, c) for s, c in hits],
        "note": "Cite passage ids in square brackets after the claims they support.",
    }


@mcp.tool()
def get_passage(chunk_id: str) -> dict:
    """Fetch one passage verbatim by its id, for checking a citation in full.

    Args:
        chunk_id: An id from search_evidence, e.g.
            "paper:exercise_ageing_gat#3" or "table:convergent_genes#CTSF".
    """
    for c in _get_index().chunks:
        if c.id == chunk_id:
            return {"found": True, "id": c.id, "source": c.source,
                    "kind": c.kind, "text": c.text, "meta": c.meta}
    return {"found": False, "id": chunk_id,
            "reason": "No passage with that id. Ids come from search_evidence."}


@mcp.tool()
def verify_citation(chunk_id: str, claim: str) -> dict:
    """Check whether a passage plausibly supports a claim attributed to it.

    Lexical overlap only -- this flags citations with no textual basis, it does
    not confirm that a well-overlapping claim is a correct reading. Treat a low
    score as a reason to re-read the passage, not as proof of fabrication.

    Args:
        chunk_id: The passage id cited.
        claim: The sentence attributed to it.
    """
    passage = get_passage(chunk_id)
    if not passage.get("found"):
        return {"verdict": "unsupported", "reason": "Cited id does not exist.",
                "id": chunk_id}

    import re
    stop = {"the", "a", "an", "and", "or", "of", "in", "to", "is", "are", "was",
            "were", "for", "on", "with", "that", "this", "it", "as", "by", "at",
            "from", "be", "has", "have", "had", "not", "but", "than", "then"}
    tok = lambda s: {w for w in re.findall(r"[a-z0-9#:_.-]+", s.lower()) if w not in stop and len(w) > 2}
    c_tokens, p_tokens = tok(claim), tok(passage["text"])
    if not c_tokens:
        return {"verdict": "unsupported", "reason": "Claim had no comparable terms.",
                "id": chunk_id}

    overlap = c_tokens & p_tokens
    ratio = len(overlap) / len(c_tokens)
    verdict = "supported" if ratio >= 0.5 else "partial" if ratio >= 0.25 else "unsupported"
    return {
        "verdict": verdict,
        "id": chunk_id,
        "overlap_ratio": round(ratio, 3),
        "shared_terms": sorted(overlap)[:20],
        "missing_terms": sorted(c_tokens - p_tokens)[:20],
        "passage": passage["text"],
        "note": "Lexical check only; re-read the passage before relying on this.",
    }


@mcp.tool()
def describe_index() -> dict:
    """Report what is indexed, so a caller knows the scope before asking."""
    chunks = _get_index().chunks
    papers = sorted({c.meta.get("file", c.source) for c in chunks if c.kind == "paper"})
    tables = sorted({c.meta.get("file", c.source) for c in chunks if c.kind == "table_row"})
    return {
        "total_passages": len(chunks),
        "paper_passages": sum(1 for c in chunks if c.kind == "paper"),
        "table_row_passages": sum(1 for c in chunks if c.kind == "table_row"),
        "papers": papers,
        "tables": tables,
        "embedding_model": EMBED_MODEL,
        "relevance_floor": MIN_SCORE,
        "scope": ("Ageing and exercise genomics: the author's own papers and "
                  "structured model outputs. Not a general biomedical library."),
    }


@mcp.tool()
def answer_question(question: str, k: int = 6, mix: bool = True) -> dict:
    """Generate a grounded, cited answer using the configured LLM backend.

    Prefer search_evidence when you can reason over the passages yourself --
    it needs no model and leaves the reasoning with you. Use this when you
    want the pipeline's own grounded answer.

    Args:
        question: The question to answer.
        k: Passages to retrieve (1-20).
        mix: Ensure both a paper chunk and a table row are represented.
    """
    backend = os.environ.get("LLM_BACKEND")
    if not backend:
        return {"error": "LLM_BACKEND is not set; use search_evidence instead, "
                         "which needs no model."}
    try:
        from llm_backend import LLM
    except ImportError as e:
        return {"error": f"Could not import the LLM backend: {e}"}

    llm = LLM(backend=backend, model=os.environ.get("LLM_MODEL"))
    k = max(1, min(int(k), 20))
    return RAG(_get_index(), llm).answer(question, k=k, mix=mix)


def _selftest() -> int:
    """Exercise every no-model tool against the configured documents."""
    print(f"sources: PAPERS_DIR={PAPERS_DIR} TABLES_DIR={TABLES_DIR}")
    stats = describe_index()
    print(f"  indexed {stats['total_passages']} passages "
          f"({stats['paper_passages']} paper, {stats['table_row_passages']} table)")
    assert stats["total_passages"] > 0, "nothing indexed"

    hits = search_evidence("which genes are ageing-causal", k=3)
    print(f"  search_evidence -> grounded={hits['grounded']}")
    assert hits["grounded"], hits.get("reason")
    first = hits["passages"][0]["id"]

    got = get_passage(first)
    assert got["found"] and got["id"] == first, "get_passage round-trip failed"
    print(f"  get_passage      -> {first}")

    missing = get_passage("paper:does-not-exist#0")
    assert not missing["found"], "unknown id should not be found"

    good = verify_citation(first, got["text"][:120])
    bad = verify_citation(first, "Ridley Scott directed Alien in 1979.")
    print(f"  verify_citation  -> quoted={good['verdict']} unrelated={bad['verdict']}")
    assert good["verdict"] == "supported", good
    assert bad["verdict"] == "unsupported", bad

    empty = search_evidence("   ")
    assert not empty["grounded"], "empty query should refuse"
    badkind = search_evidence("genes", kind="nonsense")
    assert not badkind["grounded"], "bad kind should refuse"

    print("selftest OK - every tool exercised with no model and no network")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    mcp.run()
