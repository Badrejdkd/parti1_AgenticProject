"""Bases vectorielles : PGVector (PostgreSQL), Qdrant et Chroma DB derrière une même interface.

Chaque élément est stocké avec :
- un vecteur « text » (bge-m3) : contenu du texte / tableau / description d'image ;
- un vecteur « clip » (CLIP, uniquement pour les images).
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Hit:
    element: dict      # id, doc, type, page, content, image_path, metadata
    score: float       # similarité cosinus (plus c'est haut, plus c'est proche)


def _scalar_meta(r: dict) -> dict:
    """Métadonnées simples (str/int) compatibles avec toutes les bases."""
    m = r.get("metadata") or {}
    return {"source": str(m.get("source", "")), "caption": str(m.get("caption", "") or ""),
            "kind": str(m.get("kind", ""))}


class VectorStore(ABC):
    name = "base"

    @abstractmethod
    def upsert(self, records: list[dict]) -> None:
        """records : dicts avec id, doc, type, page, content, image_path, metadata, embedding, clip."""

    @abstractmethod
    def search(self, vector: np.ndarray, space: str = "text", type: str | None = None,
               k: int = 5, doc: str | None = None) -> list[Hit]:
        """space = "text" (bge-m3) ou "clip"."""

    @abstractmethod
    def delete_doc(self, doc: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...


# ============================================================================ PGVector
class PGVectorStore(VectorStore):
    name = "pgvector"

    def __init__(self, text_dim: int, clip_dim: int, dsn: str = config.PG_DSN):
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS elements (
                id         TEXT PRIMARY KEY,
                doc        TEXT NOT NULL,
                type       TEXT NOT NULL,
                page       INT,
                content    TEXT,
                image_path TEXT,
                metadata   JSONB,
                embedding  vector({text_dim}),
                clip       vector({clip_dim})
            )""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS elements_type_idx ON elements (type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS elements_emb_idx ON elements USING hnsw (embedding vector_cosine_ops)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS elements_clip_idx ON elements USING hnsw (clip vector_cosine_ops)")

    def upsert(self, records):
        from psycopg.types.json import Jsonb
        rows = [(r["id"], r["doc"], r["type"], r["page"], r["content"], r.get("image_path"),
                 Jsonb({k: v for k, v in (r.get("metadata") or {}).items() if k != "page_context"}),
                 r["embedding"], r.get("clip")) for r in records]
        with self.conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO elements (id, doc, type, page, content, image_path, metadata, embedding, clip)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET doc = EXCLUDED.doc, type = EXCLUDED.type, page = EXCLUDED.page,
                    content = EXCLUDED.content, image_path = EXCLUDED.image_path, metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding, clip = EXCLUDED.clip""", rows)

    def search(self, vector, space="text", type=None, k=5, doc=None):
        col = "embedding" if space == "text" else "clip"
        where, params = [f"{col} IS NOT NULL"], []
        if type:
            where.append("type = %s"); params.append(type)
        if doc:
            where.append("doc = %s"); params.append(doc)
        sql = (f"SELECT id, doc, type, page, content, image_path, metadata, 1 - ({col} <=> %s) AS score "
               f"FROM elements WHERE {' AND '.join(where)} ORDER BY {col} <=> %s LIMIT %s")
        rows = self.conn.execute(sql, [vector, *params, vector, k]).fetchall()
        keys = ["id", "doc", "type", "page", "content", "image_path", "metadata"]
        return [Hit(element=dict(zip(keys, r[:7])), score=float(r[7])) for r in rows]

    def delete_doc(self, doc):
        self.conn.execute("DELETE FROM elements WHERE doc = %s", [doc])

    def count(self):
        return self.conn.execute("SELECT count(*) FROM elements").fetchone()[0]


# ============================================================================ Qdrant
class QdrantStore(VectorStore):
    name = "qdrant"

    def __init__(self, text_dim: int, clip_dim: int, url: str = config.QDRANT_URL,
                 collection: str = config.QDRANT_COLLECTION):
        from qdrant_client import QdrantClient, models
        self.m = models
        self.collection = collection
        # URL http(s) -> serveur Docker ; sinon chemin local (mode embarqué, pratique pour tester)
        self.client = QdrantClient(url=url) if url.startswith("http") else QdrantClient(path=str(config.abs_path(url)))
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config={
                "text": models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
                "clip": models.VectorParams(size=clip_dim, distance=models.Distance.COSINE),
            })
            for field_name in ("type", "doc"):
                try:
                    self.client.create_payload_index(collection, field_name, models.PayloadSchemaType.KEYWORD)
                except Exception:
                    pass

    @staticmethod
    def _pid(element_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, element_id))

    def upsert(self, records):
        points = []
        for r in records:
            vec = {"text": r["embedding"].tolist()}
            if r.get("clip") is not None:
                vec["clip"] = r["clip"].tolist()
            payload = {"element_id": r["id"], "doc": r["doc"], "type": r["type"], "page": r["page"],
                       "content": r["content"], "image_path": r.get("image_path"), "metadata": _scalar_meta(r)}
            points.append(self.m.PointStruct(id=self._pid(r["id"]), vector=vec, payload=payload))
        for i in range(0, len(points), 64):
            self.client.upsert(self.collection, points=points[i:i + 64])

    def _filter(self, type=None, doc=None):
        conds = [self.m.FieldCondition(key=k, match=self.m.MatchValue(value=v))
                 for k, v in (("type", type), ("doc", doc)) if v]
        return self.m.Filter(must=conds) if conds else None

    def search(self, vector, space="text", type=None, k=5, doc=None):
        res = self.client.query_points(self.collection, query=vector.tolist(), using=space,
                                       query_filter=self._filter(type, doc), limit=k, with_payload=True)
        hits = []
        for p in res.points:
            pl = dict(p.payload)
            pl["id"] = pl.pop("element_id")
            hits.append(Hit(element=pl, score=float(p.score)))
        return hits

    def delete_doc(self, doc):
        self.client.delete(self.collection, points_selector=self.m.FilterSelector(filter=self._filter(doc=doc)))

    def count(self):
        return self.client.count(self.collection).count


# ============================================================================ Chroma
class ChromaStore(VectorStore):
    name = "chroma"

    def __init__(self, text_dim: int, clip_dim: int, path=config.CHROMA_DIR):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(path))
        self.text_col = self.client.get_or_create_collection("elements_text", metadata={"hnsw:space": "cosine"})
        self.clip_col = self.client.get_or_create_collection("elements_clip", metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _meta(r):
        return {"doc": r["doc"], "type": r["type"], "page": int(r["page"]),
                "image_path": r.get("image_path") or "", **_scalar_meta(r)}

    def upsert(self, records):
        for i in range(0, len(records), 128):
            batch = records[i:i + 128]
            self.text_col.upsert(ids=[r["id"] for r in batch],
                                 embeddings=[r["embedding"].tolist() for r in batch],
                                 documents=[r["content"] for r in batch],
                                 metadatas=[self._meta(r) for r in batch])
        imgs = [r for r in records if r.get("clip") is not None]
        if imgs:
            self.clip_col.upsert(ids=[r["id"] for r in imgs],
                                 embeddings=[r["clip"].tolist() for r in imgs],
                                 documents=[r["content"] for r in imgs],
                                 metadatas=[self._meta(r) for r in imgs])

    @staticmethod
    def _where(type=None, doc=None):
        conds = [{k: v} for k, v in (("type", type), ("doc", doc)) if v]
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def search(self, vector, space="text", type=None, k=5, doc=None):
        col = self.text_col if space == "text" else self.clip_col
        if col.count() == 0:
            return []
        res = col.query(query_embeddings=[vector.tolist()], n_results=k, where=self._where(type, doc),
                        include=["documents", "metadatas", "distances"])
        hits = []
        for eid, txt, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            el = {"id": eid, "doc": meta["doc"], "type": meta["type"], "page": meta["page"], "content": txt,
                  "image_path": meta.get("image_path") or None,
                  "metadata": {"source": meta.get("source", ""), "caption": meta.get("caption", ""),
                               "kind": meta.get("kind", "")}}
            hits.append(Hit(element=el, score=1.0 - float(dist)))
        return hits

    def delete_doc(self, doc):
        for col in (self.text_col, self.clip_col):
            col.delete(where={"doc": doc})

    def count(self):
        return self.text_col.count()


STORES = {"pgvector": PGVectorStore, "qdrant": QdrantStore, "chroma": ChromaStore}


def open_store(name: str, text_dim: int, clip_dim: int) -> VectorStore:
    if name not in STORES:
        raise ValueError(f"Base inconnue : {name}. Choix possibles : {list(STORES)}")
    return STORES[name](text_dim, clip_dim)
