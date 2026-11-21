"""Pipeline d'indexation complet : PDF -> éléments -> descriptions -> embeddings -> bases vectorielles.

Utilisation en ligne de commande :
    python -m src.indexer pdfs/ABB_ACS580_Guide_Variateur.pdf --backends pgvector qdrant chroma
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import config
from .embeddings import get_clip_embedder, get_text_embedder
from .ingestion import Element, parse_pdf, save_elements, slugify
from .vectorstore import open_store


def _say(progress):
    def p(msg, frac):
        if progress:
            progress(msg, min(max(frac, 0.0), 1.0))
        else:
            print(f"[{frac:5.0%}] {msg}")
    return p


def text_for_embedding(e: Element) -> str:
    if e.type == "image":
        return f"Figure / image (page {e.page}) : {e.content}"
    return e.content


def describe_images(elements: list[Element], progress=None) -> None:
    """Remplit e.content des images avec une description générée par le modèle vision."""
    from .images import ImageDescriber
    p = _say(progress)
    images = [e for e in elements if e.type == "image"]
    if not images:
        return
    describer = ImageDescriber()
    for i, e in enumerate(images):
        p(f"Description de l'image {i + 1}/{len(images)} (page {e.page}) avec {config.VLM_MODEL}", i / len(images))
        try:
            e.content = describer.describe(e.image_path, e.metadata.get("source", ""), e.page,
                                           e.metadata.get("page_context", ""))
        except Exception as ex:  # Ollama arrêté, modèle absent…
            e.metadata["description_error"] = str(ex)
            e.content = fallback_description(e)


def fallback_description(e: Element) -> str:
    ctx = (e.metadata.get("page_context") or "").replace("\n", " ")[:400]
    return f"Image extraite de la page {e.page}. Contexte de la page : {ctx}"


def build_records(elements: list[Element]) -> list[dict]:
    """Calcule les embeddings bge-m3 (tous les éléments) et CLIP (images)."""
    text_emb, clip_emb = get_text_embedder(), get_clip_embedder()
    vecs = text_emb.embed([text_for_embedding(e) for e in elements])
    img_idx = [i for i, e in enumerate(elements) if e.type == "image"]
    clips = clip_emb.embed_images([elements[i].image_path for i in img_idx]) if img_idx else []
    clip_by_idx = dict(zip(img_idx, clips))
    records = []
    for i, e in enumerate(elements):
        r = e.to_dict()
        r["embedding"] = vecs[i]
        r["clip"] = clip_by_idx.get(i)
        records.append(r)
    return records


def index_pdf(pdf_path, backends=None, describe: bool = True, render_vector_pages: bool = False,
              progress=None) -> dict:
    backends = backends or [config.DEFAULT_BACKEND]
    p = _say(progress)
    t0 = time.time()
    pdf_path = Path(pdf_path)
    doc_id = slugify(pdf_path.name)

    p("Analyse du PDF : texte, tableaux, images…", 0.0)
    elements = parse_pdf(pdf_path, render_vector_pages=render_vector_pages,
                         progress=lambda m, f: p(m, 0.30 * f))

    if describe:
        describe_images(elements, progress=lambda m, f: p(m, 0.30 + 0.40 * f))
    for e in elements:
        if e.type == "image" and not e.content:
            e.content = fallback_description(e)

    p("Calcul des embeddings (bge-m3 + CLIP)…", 0.72)
    records = build_records(elements)
    text_dim, clip_dim = len(records[0]["embedding"]), get_clip_embedder().dim

    for j, name in enumerate(backends):
        p(f"Stockage dans {name}…", 0.85 + 0.15 * j / len(backends))
        store = open_store(name, text_dim, clip_dim)
        store.delete_doc(doc_id)
        store.upsert(records)

    save_elements(elements, doc_id)
    stats = {"doc": doc_id, "source": pdf_path.name, "backends": backends,
             "text": sum(e.type == "text" for e in elements),
             "table": sum(e.type == "table" for e in elements),
             "image": sum(e.type == "image" for e in elements),
             "seconds": round(time.time() - t0, 1)}
    p(f"Indexation terminée : {stats}", 1.0)
    return stats


def index_all(backends=None, describe: bool = True, render_vector_pages: bool = False,
              exclude: list[str] | None = None, progress=None) -> list[dict]:
    """Indexe tous les PDF du dossier pdfs/ (sauf ceux dont le nom contient un motif de `exclude`)."""
    exclude = [x.lower() for x in (exclude or [])]
    pdfs = [p for p in sorted(config.PDF_DIR.glob("*.pdf"))
            if not any(x in p.name.lower() for x in exclude)]
    results = []
    for i, pdf in enumerate(pdfs):
        def p(msg, f, i=i, name=pdf.name):
            frac = (i + f) / len(pdfs)
            if progress:
                progress(f"[{i + 1}/{len(pdfs)}] {name} – {msg}", frac)
            else:
                print(f"[{frac:5.0%}] [{i + 1}/{len(pdfs)}] {name} – {msg}")
        try:
            results.append(index_pdf(pdf, backends, describe, render_vector_pages, progress=p))
        except Exception as ex:
            results.append({"source": pdf.name, "error": str(ex)})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Indexer un PDF (ou tous les PDF de pdfs/) dans les bases vectorielles")
    ap.add_argument("pdf", nargs="?", help="chemin du PDF (inutile avec --all)")
    ap.add_argument("--all", action="store_true", help="indexer tous les PDF du dossier pdfs/")
    ap.add_argument("--exclude", nargs="*", default=[], help="motifs de noms de fichiers à ignorer avec --all")
    ap.add_argument("--backends", nargs="+", default=[config.DEFAULT_BACKEND], choices=config.BACKENDS)
    ap.add_argument("--no-describe", action="store_true", help="ne pas décrire les images (plus rapide)")
    ap.add_argument("--vector-pages", action="store_true", help="rendre en image les pages de schémas vectoriels")
    a = ap.parse_args()
    if a.all:
        for r in index_all(a.backends, not a.no_describe, a.vector_pages, a.exclude):
            print(r)
    elif a.pdf:
        print(index_pdf(a.pdf, a.backends, describe=not a.no_describe, render_vector_pages=a.vector_pages))
    else:
        ap.error("indiquer un PDF ou --all")
