# Ask-my-research

**A retrieval-augmented generation (RAG) pipeline that answers questions about my research by retrieving from both the papers *and* the structured model outputs — and grounding every answer in cited sources.**

Most RAG demos retrieve from text only. Real research questions often need *both* the prose (the reasoning, the caveats) and the structured results (which genes, what scores). This pipeline indexes paper chunks and verbalised result-table rows in one store, so a single query draws on both — and every answer carries the passages it used, or refuses when the context is too thin.

![architecture](assets/architecture.png)

---

## It works — on real research

![example](assets/example.png)

The answer above is generated, not hard-coded: the pipeline retrieved the relevant paragraphs from the paper and a row from the results table, then grounded the model on them and cited each source.

---

## Why it's built this way

| Choice | Reason |
|---|---|
| **Heterogeneous retrieval** (papers + tables in one index) | Research answers need prose *and* structured results; rows are verbalised to text so one dense index serves both. |
| **Grounded generation with citations** | Every claim points back to a retrieved passage; the model is told to refuse when context is insufficient. |
| **Backend-agnostic LLM** | Same pipeline runs on a self-hosted/open-weight model (Ollama or HF Transformers) **or** a frontier API model — switchable with one flag. |
| **Offline self-test** | Ingestion + retrieval logic is verifiable with no model and no network. |

## Quickstart

```bash
pip install -r requirements.txt

# verify the pipeline logic with no model, no network
python ask.py --selftest

# ask a question (free, local, open-weight via HF Transformers)
python ask.py --papers data/papers --tables data/tables \
              --backend transformers --model "Qwen/Qwen2.5-7B-Instruct" --k 12 \
              --q "What did the model add beyond the raw per-gene signal?"
```

Drop your own `.txt`/`.md` papers into `data/papers/` and `.csv`/`.tsv` result tables into `data/tables/`. A Colab notebook (`ask_my_research_colab.ipynb`) runs the whole thing on a free GPU. Example data is included so it runs out of the box.

## How it works

1. **Ingest** — papers become ~800-char chunks; table rows are verbalised to text (`gene: CTSF; ageing_causal: yes; ...`). Each chunk keeps a source id.
2. **Embed + index** — `sentence-transformers` (MiniLM); cosine similarity.
3. **Retrieve** — top-k, with an optional step that guarantees both a paper chunk and a table row when available.
4. **Generate** — a grounded prompt instructs the model to answer only from context, cite the ids it used, and decline when context is thin.

## Honest scope

This is a **research prototype**, and its limits are part of the design, not hidden:

- **Answer accuracy scales with model size.** A small (1.5B) model can misread a nuanced caveat that a larger (7B) model gets right. The retrieval is the same; the generator is the variable. Use a 7B+ model for faithful answers.
- **Dense retrieval only** — no learned re-ranker. Nuanced single-sentence caveats are sometimes retrieved weakly; a re-ranker or stronger embedder would help.
- **The LLM and embedder are existing open-weight/API models, used as-is.** The pipeline, retrieval design, grounding, and citation logic are the contribution — not the models.
- **In-memory index** — fine for a personal corpus; swap in FAISS/Chroma for scale.

## Files

| File | What it is |
|---|---|
| `ask.py` | CLI: build index, ask questions, offline `--selftest`. |
| `src/rag.py` | Ingestion (papers + tables), dense index, grounded-generation engine. |
| `src/llm_backend.py` | Backend-agnostic LLM interface: Transformers / Ollama / Anthropic / OpenAI. |
| `ask_my_research_colab.ipynb` | One-click Colab notebook (free GPU). |
| `data/` | Your corpus — example files included. |
