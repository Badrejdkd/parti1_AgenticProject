"""Configuration centrale du projet (lue depuis le fichier .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def abs_path(p) -> Path:
    """Chemin absolu à partir d'un chemin relatif à la racine du projet."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def rel_path(p) -> str:
    """Chemin relatif à la racine (portable entre machines)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


# --- Dossiers ---
PDF_DIR = ROOT / "pdfs"
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
ELEMENTS_DIR = DATA_DIR / "elements"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = abs_path(_env("CHROMA_DIR", "data/chroma"))
for _d in (PDF_DIR, DATA_DIR, IMAGES_DIR, ELEMENTS_DIR, CACHE_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Bases vectorielles ---
PG_DSN = _env("PG_DSN", "postgresql://rag:rag@localhost:5433/maintenance")
QDRANT_URL = _env("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "maintenance")
BACKENDS = ["pgvector", "qdrant", "chroma"]
DEFAULT_BACKEND = _env("DEFAULT_BACKEND", "pgvector")

# --- Ollama ---
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
# Qwen3.5 9B est multimodal (texte + images) et gère les appels d'outils :
# un seul modèle sert à la fois de LLM (génération, agent) et de VLM (description des images).
LLM_MODEL = _env("LLM_MODEL", "qwen3.5:9b")
VLM_MODEL = _env("VLM_MODEL", "qwen3.5:9b")
LLM_NUM_CTX = int(_env("LLM_NUM_CTX", "8192"))
# Mode « réflexion » de Qwen3.5 : désactivé par défaut (réponses beaucoup plus rapides)
LLM_THINK = _env("LLM_THINK", "false").lower() in ("1", "true", "yes")

# --- Embeddings ---
TEXT_EMB_MODEL = _env("TEXT_EMB_MODEL", "BAAI/bge-m3")
CLIP_IMAGE_MODEL = _env("CLIP_IMAGE_MODEL", "clip-ViT-B-32")
CLIP_TEXT_MODEL = _env("CLIP_TEXT_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1")
EMB_DEVICE = _env("EMB_DEVICE", "auto")

# --- Découpage ---
CHUNK_SIZE = int(_env("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "64"))
TABLE_ROWS_PER_CHUNK = int(_env("TABLE_ROWS_PER_CHUNK", "8"))
MIN_IMAGE_SIZE = int(_env("MIN_IMAGE_SIZE", "120"))

# --- Seuils de pertinence des images ---
IMG_DESC_MIN_SCORE = float(_env("IMG_DESC_MIN_SCORE", "0.40"))
IMG_CLIP_MIN_SCORE = float(_env("IMG_CLIP_MIN_SCORE", "0.25"))
