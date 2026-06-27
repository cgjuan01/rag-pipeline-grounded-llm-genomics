#!/usr/bin/env python3
"""
llm_backend.py
==============
A thin, backend-agnostic interface to a chat LLM, so the RAG pipeline runs
against either a self-hosted/open-weight model (via Ollama) or a frontier
API-based model (Anthropic / OpenAI), switchable by one config flag.

This mirrors a real deployment concern: the retrieval and prompt-construction
logic should not care which model generates the final answer.

Backends
--------
- "ollama"    : local, self-hosted/open-weight (default). Free, offline.
                Requires Ollama running locally (https://ollama.com) and a
                pulled model, e.g.:  ollama pull llama3.1:8b
- "anthropic" : Claude via API. Requires ANTHROPIC_API_KEY in the environment.
- "openai"    : GPT via API. Requires OPENAI_API_KEY in the environment.

Usage
-----
    from llm_backend import LLM
    llm = LLM(backend="ollama", model="llama3.1:8b")
    print(llm.complete("Say hello in one word."))
"""

from __future__ import annotations
import os
from typing import Optional


class LLM:
    def __init__(
        self,
        backend: str = "ollama",
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.backend = backend.lower()
        self.temperature = temperature
        # sensible per-backend default models
        self.model = model or {
            "ollama": "llama3.1:8b",
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4o-mini",
        }.get(self.backend, "llama3.1:8b")

        if self.backend not in {"ollama", "anthropic", "openai", "transformers"}:
            raise ValueError(f"Unknown backend: {self.backend}")
        if self.backend == "transformers":
            # default to a small, ungated instruct model that runs on Colab's
            # free GPU with no API key and no access request.
            self.model = model or "Qwen/Qwen2.5-1.5B-Instruct"
            self._pipe = None  # lazy-loaded on first call

    # --- public API -------------------------------------------------------
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Return a single completion string for a prompt."""
        if self.backend == "ollama":
            return self._ollama(prompt, system)
        if self.backend == "anthropic":
            return self._anthropic(prompt, system)
        if self.backend == "openai":
            return self._openai(prompt, system)
        if self.backend == "transformers":
            return self._transformers(prompt, system)
        raise RuntimeError("unreachable")

    # --- backends ---------------------------------------------------------
    def _ollama(self, prompt: str, system: Optional[str]) -> str:
        # Local, self-hosted/open-weight. Uses the official `ollama` client if
        # present, else falls back to the REST endpoint via `requests`.
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            import ollama  # type: ignore
            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature},
            )
            return resp["message"]["content"]
        except ImportError:
            import requests
            r = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

    def _anthropic(self, prompt: str, system: Optional[str]) -> str:
        import anthropic  # pip install anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        # concatenate text blocks
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def _openai(self, prompt: str, system: Optional[str]) -> str:
        from openai import OpenAI  # pip install openai
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

    def _transformers(self, prompt: str, system: Optional[str]) -> str:
        # Free, open-weight, no API key. Runs in-process via Hugging Face
        # transformers — ideal for Google Colab's free GPU. The model is
        # lazy-loaded once and reused.
        if self._pipe is None:
            import torch
            from transformers import pipeline
            self._pipe = pipeline(
                "text-generation",
                model=self.model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        out = self._pipe(
            messages,
            max_new_tokens=512,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 0.01),
            return_full_text=False,
        )
        # transformers returns the generated assistant turn
        gen = out[0]["generated_text"]
        if isinstance(gen, list):  # chat format returns list of messages
            return gen[-1]["content"]
        return gen


if __name__ == "__main__":
    # quick smoke test (requires a running backend)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    llm = LLM(backend=args.backend, model=args.model)
    print(llm.complete("Reply with exactly one word: ready"))
