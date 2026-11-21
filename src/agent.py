"""Étape 6 : RAG agentique avec LangChain.

L'agent (Qwen via Ollama) choisit lui-même quels outils de recherche appeler :
texte, tableaux, images (CLIP + descriptions) ou recherche exacte d'un code de défaut.
`create_agent` de LangChain >= 1.0 s'appuie sur LangGraph pour la boucle agentique.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from .rag import MultimodalRetriever, get_llm, lookup_code

AGENT_PROMPT = """Tu es MaintAssist, un agent expert en maintenance industrielle (variateurs, machines, équipements).
Tu disposes d'outils pour interroger les manuels techniques indexés. Stratégie :
1. Si la question contient un code de défaut, d'alarme ou de paramètre (ex. 2310, 3220, A2B1, FA81, 99.13), appelle d'abord `rechercher_code_defaut`.
2. Pour une procédure, une explication, une consigne : `rechercher_texte`.
3. Pour des valeurs, caractéristiques, intervalles, listes de paramètres : `rechercher_tableaux`.
4. Si l'utilisateur demande un schéma, une figure, un câblage, un panneau, ou si une illustration aide : `rechercher_images`.
Tu peux appeler plusieurs outils et reformuler ta requête si les résultats sont insuffisants.
Réponds en français, de façon structurée, UNIQUEMENT à partir des résultats des outils. Cite les pages (ex. « page 12 »).
Ajoute les consignes de sécurité utiles pour toute intervention. Si rien de pertinent n'est trouvé, dis-le."""


class Collector:
    """Garde la trace des éléments retrouvés par les outils (pour l'affichage Streamlit)."""

    def __init__(self):
        self.items: dict[str, dict] = {}

    def reset(self):
        self.items = {}

    def add(self, element: dict, score: float | None = None):
        if element["id"] not in self.items:
            self.items[element["id"]] = {**element, "score": None if score is None else round(score, 3)}

    def by_type(self, t: str) -> list[dict]:
        return [e for e in self.items.values() if e["type"] == t]


def _fmt(elements: list[dict], max_chars: int = 1500) -> str:
    if not elements:
        return "Aucun résultat pertinent."
    out = []
    for e in elements:
        score = f" | score {e['score']:.2f}" if e.get("score") is not None else ""
        out.append(f"[page {e['page']} | {e['type']} | {e['doc']}{score}]\n{e['content'][:max_chars]}")
    return "\n---\n".join(out)


def build_tools(retriever: MultimodalRetriever, collector: Collector):
    def _run(hits):
        els = []
        for h in hits:
            collector.add(h.element, h.score)
            els.append({**h.element, "score": h.score})
        return _fmt(els)

    @tool
    def rechercher_texte(requete: str) -> str:
        """Recherche des passages de texte (procédures, explications, consignes de sécurité) dans les manuels."""
        return _run(retriever.search(requete, "text", k=5))

    @tool
    def rechercher_tableaux(requete: str) -> str:
        """Recherche dans les tableaux des manuels (codes, caractéristiques, paramètres, intervalles de maintenance)."""
        return _run(retriever.search(requete, "table", k=4))

    @tool
    def rechercher_images(requete: str) -> str:
        """Recherche des images, schémas, figures ou logigrammes (via CLIP et les descriptions d'images).
        Renvoie la description de chaque image trouvée."""
        return _run(retriever.search_images(requete, k=3))

    @tool
    def rechercher_code_defaut(code: str) -> str:
        """Recherche EXACTE d'un code de défaut, d'alarme ou de paramètre (ex. '2310', 'A2B1', 'FA81', '99.13')."""
        els = lookup_code(code)
        for e in els:
            collector.add(e)
        return _fmt(els)

    return [rechercher_code_defaut, rechercher_texte, rechercher_tableaux, rechercher_images]


class MaintenanceAgent:
    def __init__(self, backend: str | None = None):
        self.retriever = MultimodalRetriever(backend)
        self.collector = Collector()
        self.tools = build_tools(self.retriever, self.collector)
        self.agent = create_agent(model=get_llm(), tools=self.tools, system_prompt=AGENT_PROMPT)

    def ask(self, question: str, history: list[dict] | None = None, query_image=None) -> dict:
        self.collector.reset()
        content = question
        if query_image is not None:  # photo du technicien -> recherche CLIP image→image
            hits = self.retriever.search_by_image(query_image, k=3)
            for h in hits:
                self.collector.add(h.element, h.score)
            desc = "\n".join(f"- page {h.element['page']} (similarité {h.score:.2f}) : {h.element['content'][:400]}"
                             for h in hits)
            content = (f"{question}\n\n[Photo jointe par l'utilisateur. Images les plus proches dans les manuels "
                       f"(CLIP) :\n{desc}]")
        messages = list(history or []) + [{"role": "user", "content": content}]
        result = self.agent.invoke({"messages": messages}, config={"recursion_limit": 14})

        steps, by_call = [], {}
        for m in result["messages"]:
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    step = {"tool": tc["name"], "args": tc["args"], "result": ""}
                    by_call[tc["id"]] = step
                    steps.append(step)
            elif isinstance(m, ToolMessage) and m.tool_call_id in by_call:
                by_call[m.tool_call_id]["result"] = str(m.content)
        return {"answer": result["messages"][-1].content, "steps": steps,
                "sources": list(self.collector.items.values()),
                "images": self.collector.by_type("image"), "tables": self.collector.by_type("table")}
