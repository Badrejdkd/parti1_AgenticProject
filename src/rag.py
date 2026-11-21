"""Étape 4 : recherche multimodale + génération de la réponse avec Qwen (Ollama)."""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from . import config
from .embeddings import get_clip_embedder, get_text_embedder
from .ingestion import load_elements
from .vectorstore import Hit, open_store

SYSTEM_PROMPT = """Tu es MaintAssist, un assistant de maintenance industrielle.
Tu réponds en français, UNIQUEMENT à partir du CONTEXTE fourni (extraits de manuels techniques : passages de texte, tableaux et descriptions d'images).
Règles :
- Cite tes sources avec leur numéro [S1], [S2]… et précise la page.
- Pour une procédure, donne des étapes numérotées.
- Rappelle les consignes de sécurité pertinentes quand une intervention est décrite.
- Si une image du contexte illustre la réponse, mentionne-la (ex. « voir le schéma page 9 »).
- Si le contexte ne contient pas la réponse, dis-le clairement au lieu d'inventer."""


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_HOST, temperature=0.1,
                      num_ctx=config.LLM_NUM_CTX, reasoning=config.LLM_THINK)


def rrf_fuse(result_lists: list[list[Hit]], k: int, c: int = 60) -> list[tuple[Hit, dict]]:
    """Reciprocal Rank Fusion : combine plusieurs classements."""
    table: dict[str, dict] = {}
    for li, hits in enumerate(result_lists):
        for rank, h in enumerate(hits):
            entry = table.setdefault(h.element["id"], {"hit": h, "rrf": 0.0, "scores": {}})
            entry["rrf"] += 1.0 / (c + rank + 1)
            entry["scores"][li] = h.score
    ranked = sorted(table.values(), key=lambda x: x["rrf"], reverse=True)[:k]
    return [(x["hit"], x["scores"]) for x in ranked]


class MultimodalRetriever:
    def __init__(self, backend: str | None = None):
        self.backend = backend or config.DEFAULT_BACKEND
        self.text_emb = get_text_embedder()
        self.clip_emb = get_clip_embedder()
        self.store = open_store(self.backend, self.text_emb.dim, self.clip_emb.dim)

    def search(self, query: str, type: str, k: int = 5, doc: str | None = None) -> list[Hit]:
        return self.store.search(self.text_emb.embed_query(query), "text", type, k, doc)

    def search_images(self, query: str, k: int = 3, doc: str | None = None, filter_scores: bool = True) -> list[Hit]:
        """Images retrouvées par leur description (bge-m3) ET par CLIP texte→image, fusionnées."""
        by_desc = self.store.search(self.text_emb.embed_query(query), "text", "image", k * 2, doc)
        by_clip = self.store.search(self.clip_emb.embed_text(query), "clip", "image", k * 2, doc)
        out = []
        for hit, scores in rrf_fuse([by_desc, by_clip], k):
            d, c = scores.get(0, 0.0), scores.get(1, 0.0)
            if filter_scores and d < config.IMG_DESC_MIN_SCORE and c < config.IMG_CLIP_MIN_SCORE:
                continue
            hit.element.setdefault("metadata", {})
            hit.element["metadata"] = {**(hit.element["metadata"] or {}), "desc_score": round(d, 3),
                                       "clip_score": round(c, 3)}
            out.append(Hit(hit.element, max(d, c)))
        return out

    def search_by_image(self, image, k: int = 3, doc: str | None = None) -> list[Hit]:
        """Recherche image→image avec CLIP (ex. photo prise par le technicien)."""
        return self.store.search(self.clip_emb.embed_pil([image])[0], "clip", "image", k, doc)

    def retrieve(self, query: str, k_text: int = 5, k_table: int = 3, k_image: int = 3,
                 doc: str | None = None, query_image=None) -> dict[str, list[Hit]]:
        res = {"text": self.search(query, "text", k_text, doc),
               "table": self.search(query, "table", k_table, doc),
               "image": self.search_images(query, k_image, doc)}
        if query_image is not None:
            seen = {h.element["id"] for h in res["image"]}
            res["image"] = [h for h in self.search_by_image(query_image, k_image, doc)
                            if h.element["id"] not in seen] + res["image"]
        return res


# ----------------------------------------------------------------------------- recherche exacte
def _focus_table(content: str, pattern: re.Pattern) -> str:
    """Pour un tableau Markdown, garde le titre, l'en-tête et seulement les lignes contenant le code."""
    lines = content.splitlines()
    rows = [i for i, l in enumerate(lines) if l.startswith("|")]
    if len(rows) < 3:
        return content
    header = lines[:rows[1] + 1]
    matched = [lines[i] for i in rows[2:] if pattern.search(lines[i])]
    return "\n".join(header + matched) if matched else content


def lookup_code(code: str, doc: str | None = None, max_hits: int = 6) -> list[dict]:
    """Recherche EXACTE d'un code de défaut/alarme/paramètre (ex. 2310, A2B1, 99.13)."""
    code = code.strip().upper()
    pattern = re.compile(rf"(?<![0-9A-Za-z.]){re.escape(code)}(?![0-9A-Za-z])", re.I)
    hits = [dict(e) for e in load_elements(doc) if e["type"] in ("table", "text") and pattern.search(e["content"])]
    hits.sort(key=lambda e: (0 if e["type"] == "table" else 1, e["page"]))
    for h in hits:
        if h["type"] == "table":
            h["content"] = _focus_table(h["content"], pattern)
    return hits[:max_hits]


# ----------------------------------------------------------------------------- génération
def format_context(results: dict[str, list[Hit]]) -> tuple[str, list[dict]]:
    """Construit le contexte numéroté [S1]… envoyé au LLM."""
    labels = {"text": "texte", "table": "tableau", "image": "image"}
    parts, sources = [], []
    for kind in ("text", "table", "image"):
        for h in results.get(kind, []):
            n = len(sources) + 1
            e = h.element
            body = e["content"] if kind != "image" else f"Description de l'image : {e['content']}"
            parts.append(f"[S{n}] ({labels[kind]}, page {e['page']}, document {e['doc']})\n{body}")
            sources.append({"n": n, "score": round(h.score, 3), **e})
    return "\n\n---\n\n".join(parts), sources


class RAGPipeline:
    """RAG « classique » (non agentique) : recherche fixe puis génération."""

    def __init__(self, backend: str | None = None):
        self.retriever = MultimodalRetriever(backend)
        self.llm = get_llm()

    def ask(self, question: str, query_image=None, k_text: int = 5, k_table: int = 3, k_image: int = 3,
            doc: str | None = None) -> dict:
        results = self.retriever.retrieve(question, k_text, k_table, k_image, doc, query_image)
        context, sources = format_context(results)
        extra = ""
        if query_image is not None:
            extra = "\n(L'utilisateur a joint une photo ; les images similaires du manuel sont dans le contexte.)"
        messages = [SystemMessage(SYSTEM_PROMPT),
                    HumanMessage(f"CONTEXTE :\n{context}\n\nQUESTION : {question}{extra}")]
        answer = self.llm.invoke(messages).content
        return {"answer": answer, "sources": sources,
                "images": [s for s in sources if s["type"] == "image"],
                "tables": [s for s in sources if s["type"] == "table"],
                "context": context}
