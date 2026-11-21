"""Embeddings open source.

- TextEmbedder : BAAI/bge-m3 (multilingue FR/EN, 1024 dimensions) pour le texte,
  les tableaux et les descriptions d'images.
- ClipEmbedder : CLIP ViT-B/32 (512 dimensions) pour les images, avec un encodeur
  de texte CLIP multilingue pour pouvoir interroger les images en français.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from . import config


def _device() -> str:
    if config.EMB_DEVICE != "auto":
        return config.EMB_DEVICE
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class TextEmbedder:
    def __init__(self, model_name: str = config.TEXT_EMB_MODEL):
        self.device = _device()
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda":
            self.model.half()  # divise la mémoire GPU par deux
        self.model.max_seq_length = 1024
        self.dim = len(self.embed(["test"])[0])

    def embed(self, texts, batch_size: int = 16) -> np.ndarray:
        texts = list(texts)
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 64)
        return vecs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class ClipEmbedder:
    def __init__(self, image_model: str = config.CLIP_IMAGE_MODEL, text_model: str = config.CLIP_TEXT_MODEL):
        self.device = _device()
        self.image_model = SentenceTransformer(image_model, device=self.device)
        self.text_model = SentenceTransformer(text_model, device=self.device)
        self.dim = len(self.embed_text("test"))

    def embed_pil(self, images: list[Image.Image]) -> np.ndarray:
        imgs = [im.convert("RGB") for im in images]
        return self.image_model.encode(imgs, batch_size=16, normalize_embeddings=True,
                                       convert_to_numpy=True).astype(np.float32)

    def embed_images(self, paths) -> np.ndarray:
        return self.embed_pil([Image.open(config.abs_path(p)) for p in paths])

    def embed_text(self, text: str) -> np.ndarray:
        return self.text_model.encode([text], normalize_embeddings=True,
                                      convert_to_numpy=True)[0].astype(np.float32)


@lru_cache(maxsize=1)
def get_text_embedder() -> TextEmbedder:
    return TextEmbedder()


@lru_cache(maxsize=1)
def get_clip_embedder() -> ClipEmbedder:
    return ClipEmbedder()
