"""Étapes 1-2 : analyse du PDF et extraction du texte, des tableaux et des images.

- PyMuPDF  : texte par blocs + images raster (+ rendu des pages contenant des schémas vectoriels)
- pdfplumber : détection et extraction des tableaux
- LlamaIndex : découpage du texte en passages (chunks) avec SentenceSplitter
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
import pdfplumber
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from PIL import Image

from . import config

TYPE_ORDER = {"text": 0, "table": 1, "image": 2}


@dataclass
class Element:
    """Unité indexée : un passage de texte, un morceau de tableau ou une image."""
    id: str
    doc: str
    type: str                 # "text" | "table" | "image"
    page: int                 # page du PDF (commence à 1)
    content: str              # texte, tableau Markdown ou description de l'image
    image_path: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", Path(name).stem).strip("_").lower()
    return s[:60] or "document"


# ----------------------------------------------------------------------------- tableaux
def _clean_cell(c) -> str:
    if c is None:
        return ""
    return re.sub(r"\s+", " ", str(c)).strip().replace("|", "/")


def table_to_markdown_chunks(rows: list[list], rows_per_chunk: int) -> list[str]:
    """Convertit un tableau en Markdown, découpé en morceaux qui répètent l'en-tête."""
    rows = [[_clean_cell(c) for c in r] for r in rows if r and any(c for c in r)]
    if len(rows) < 2:
        return []
    header, body = rows[0], rows[1:]
    ncol = len(header)
    chunks = []
    for i in range(0, len(body), rows_per_chunk):
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * ncol]
        for r in body[i:i + rows_per_chunk]:
            r = (r + [""] * ncol)[:ncol]
            lines.append("| " + " | ".join(r) + " |")
        chunks.append("\n".join(lines))
    return chunks


def _center_inside(bbox, rect, margin: float = 2.0) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return rect[0] - margin <= cx <= rect[2] + margin and rect[1] - margin <= cy <= rect[3] + margin


def _find_caption(table_bbox, text_blocks, max_gap: float = 30.0) -> str | None:
    """Cherche une légende « Tableau … / Table … » juste au-dessus ou au-dessous du tableau."""
    x0, top, x1, bottom = table_bbox
    for b in text_blocks:
        txt = b[4].strip().replace("\n", " ")
        if not re.match(r"^(Tableau|Table|Tab\.)\s", txt, re.I):
            continue
        below = 0 <= b[1] - bottom <= max_gap
        above = 0 <= top - b[3] <= max_gap
        if below or above:
            return txt
    return None


# ----------------------------------------------------------------------------- parsing
def parse_pdf(pdf_path, render_vector_pages: bool = False, progress=None) -> list[Element]:
    """Analyse un PDF et renvoie la liste des éléments (texte, tableaux, images).

    render_vector_pages : si True, une page contenant beaucoup de dessins vectoriels
    (schémas de câblage des manuels officiels) est rendue en image pour être indexée.
    """
    pdf_path = Path(pdf_path)
    doc_id = slugify(pdf_path.name)
    img_dir = config.IMAGES_DIR / doc_id
    img_dir.mkdir(parents=True, exist_ok=True)

    elements: list[Element] = []
    page_docs: list[Document] = []
    seen_images: set[str] = set()

    pdf = fitz.open(pdf_path)
    with pdfplumber.open(pdf_path) as plumber:
        n_pages = len(pdf)
        for pno in range(n_pages):
            page, ppage, page_num = pdf[pno], plumber.pages[pno], pno + 1
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, texte, n°bloc, type)
            text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

            # --- 1. Tableaux (pdfplumber)
            table_bboxes = []
            try:
                tables = ppage.find_tables()
            except Exception:
                tables = []
            for ti, t in enumerate(tables):
                try:
                    chunks = table_to_markdown_chunks(t.extract(), config.TABLE_ROWS_PER_CHUNK)
                except Exception:
                    chunks = []
                if not chunks:
                    continue
                table_bboxes.append(t.bbox)
                caption = _find_caption(t.bbox, text_blocks)
                title = caption or f"Tableau extrait de la page {page_num}"
                for ci, md in enumerate(chunks):
                    elements.append(Element(
                        id=f"{doc_id}-p{page_num}-table{ti}-{ci}", doc=doc_id, type="table", page=page_num,
                        content=f"{title}\n\n{md}",
                        metadata={"source": pdf_path.name, "caption": caption or "", "table_index": ti, "part": ci},
                    ))

            # --- 2. Texte hors tableaux (sera découpé par LlamaIndex)
            page_text = "\n".join(
                b[4].strip() for b in text_blocks
                if not any(_center_inside(b[:4], tb) for tb in table_bboxes)
            )
            if page_text.strip():
                page_docs.append(Document(text=page_text,
                                          metadata={"page": page_num, "doc": doc_id, "source": pdf_path.name}))

            # --- 3. Images raster
            for ii, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    data = pdf.extract_image(xref)["image"]
                    pil = Image.open(io.BytesIO(data))
                    pil.load()
                except Exception:
                    continue
                if min(pil.size) < config.MIN_IMAGE_SIZE:
                    continue  # icônes, logos, puces…
                digest = hashlib.md5(data).hexdigest()
                if digest in seen_images:
                    continue  # même image répétée sur plusieurs pages
                seen_images.add(digest)
                out = img_dir / f"p{page_num:03d}_img{ii}.png"
                pil.convert("RGB").save(out)
                elements.append(Element(
                    id=f"{doc_id}-p{page_num}-img{ii}", doc=doc_id, type="image", page=page_num, content="",
                    image_path=config.rel_path(out),
                    metadata={"source": pdf_path.name, "kind": "raster", "width": pil.width, "height": pil.height,
                              "page_context": page_text[:800]},
                ))

            # --- 4. Schémas vectoriels (optionnel)
            if render_vector_pages:
                drawings = page.get_drawings()
                outside = [d for d in drawings if not any(_center_inside(tuple(d["rect"]), tb) for tb in table_bboxes)]
                if len(outside) >= 40:
                    out = img_dir / f"p{page_num:03d}_page.png"
                    page.get_pixmap(dpi=110).save(out)
                    elements.append(Element(
                        id=f"{doc_id}-p{page_num}-page", doc=doc_id, type="image", page=page_num, content="",
                        image_path=config.rel_path(out),
                        metadata={"source": pdf_path.name, "kind": "page_render", "page_context": page_text[:800]},
                    ))

            if progress:
                progress(f"Page {page_num}/{n_pages} analysée", (pno + 1) / n_pages)
    pdf.close()

    # --- 5. Découpage du texte en passages avec LlamaIndex
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(page_docs)
    counters: dict[int, int] = {}
    for node in nodes:
        p = node.metadata["page"]
        k = counters.get(p, 0)
        counters[p] = k + 1
        elements.append(Element(id=f"{doc_id}-p{p}-text{k}", doc=doc_id, type="text", page=p,
                                content=node.get_content(), metadata={"source": pdf_path.name}))

    elements.sort(key=lambda e: (e.page, TYPE_ORDER[e.type], e.id))
    return elements


# ----------------------------------------------------------------------------- sauvegarde
def save_elements(elements: list[Element], doc_id: str) -> Path:
    path = config.ELEMENTS_DIR / f"{doc_id}.json"
    path.write_text(json.dumps([e.to_dict() for e in elements], ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_elements(doc_id: str | None = None) -> list[dict]:
    """Charge les éléments d'un document (ou de tous les documents indexés)."""
    files = [config.ELEMENTS_DIR / f"{doc_id}.json"] if doc_id else sorted(config.ELEMENTS_DIR.glob("*.json"))
    out = []
    for f in files:
        if f.exists():
            out.extend(json.loads(f.read_text(encoding="utf-8")))
    return out


def indexed_documents() -> dict[str, dict]:
    """Résumé des documents indexés : {doc_id: {source, text, table, image}}."""
    res = {}
    for f in sorted(config.ELEMENTS_DIR.glob("*.json")):
        els = json.loads(f.read_text(encoding="utf-8"))
        counts = {"text": 0, "table": 0, "image": 0}
        for e in els:
            counts[e["type"]] += 1
        res[f.stem] = {"source": els[0]["metadata"].get("source", f.stem) if els else f.stem, **counts}
    return res
