"""Étape 3 : génération de descriptions textuelles des images avec un modèle vision (Ollama).

Les descriptions sont indexées comme du texte (bge-m3), ce qui rend les images
interrogeables en langage naturel. CLIP (voir embeddings.py) complète avec une
recherche directe dans l'espace image.
"""
from __future__ import annotations

import hashlib
import io
import json

import ollama
from PIL import Image

from . import config

PROMPT = """Tu es un expert en maintenance industrielle. Décris précisément cette image extraite d'un document technique ({source}, page {page}) afin qu'elle puisse être retrouvée par un moteur de recherche.
Indique :
1) le type d'image (schéma électrique, logigramme, photo, graphique, panneau de commande, tableau, chronologie…) ;
2) les composants, libellés, codes, paramètres, valeurs et unités lisibles ;
3) ce que l'image explique ou permet de faire en maintenance.
Réponds en français, en un seul paragraphe de 5 à 8 phrases, sans inventer ce qui n'est pas visible.

Contexte textuel de la page : {context}"""


class ImageDescriber:
    def __init__(self, model: str = config.VLM_MODEL, host: str = config.OLLAMA_HOST):
        self.model = model
        self.client = ollama.Client(host=host)
        self.cache_path = config.CACHE_DIR / "descriptions.json"
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}

    @staticmethod
    def _image_bytes(path, max_side: int = 1024) -> bytes:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_side, max_side))  # limite la VRAM et le temps de calcul
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def describe(self, image_path, source: str = "", page: int = 0, context: str = "") -> str:
        path = config.abs_path(image_path)
        raw = path.read_bytes()
        key = f"{self.model}:{hashlib.md5(raw).hexdigest()}"
        if key in self.cache:  # évite de recalculer à chaque indexation
            return self.cache[key]
        prompt = PROMPT.format(source=source, page=page, context=(context or "")[:600])
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [self._image_bytes(path)]}],
            options={"temperature": 0.1, "num_ctx": 4096},
            think=False,  # pas de mode réflexion pour une simple description
        )
        text = resp["message"]["content"].strip()
        self.cache[key] = text
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return text
