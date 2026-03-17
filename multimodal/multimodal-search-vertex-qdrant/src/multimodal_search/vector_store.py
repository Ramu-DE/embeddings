"""Qdrant vector database interface.

This module provides :class:`VectorStore`, which wraps the Qdrant Python
client (v1.x) to store and retrieve embedding vectors with rich metadata.
It supports both single-vector and named-vector (multi-dimension) storage
patterns required for two-stage retrieval.
"""

from typing import Any, Dict, List, Optional, Tuple
import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    HasIdCondition,
    PayloadSchemaType,
    NamedVector,
    QueryRequest,
)

from multimodal_search.models import EmbeddingMetadata, SearchResult, SearchFilters
from multimodal_search.exceptions import StorageError


class VectorStore:
    """Manages storage and retrieval of embeddings in Qdrant (client v1.x).

    Example:
        >>> store = VectorStore(qdrant_url="http://localhost:6333")
        >>> store.initialize_collection(dimension=756)
    """

    def __init__(self, qdrant_url: str, api_key: Optional[str] = None):
        self.qdrant_url = qdrant_url
        self.api_key = api_key
        self.collection_name = "multimodal_embeddings"

        try:
            self.client = QdrantClient(url=qdrant_url, api_key=api_key)
            self.client.get_collections()
        except Exception as e:
            raise StorageError(
                f"Failed to connect to Qdrant at {qdrant_url}: {str(e)}",
                error_type="CONNECTION_FAILED",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception as e:
            raise StorageError(
                f"Failed to check collection existence: {str(e)}",
                error_type="CONNECTION_FAILED",
            )

    def _validate_vector_dimension(self, vector: List[float], expected_dim: int) -> None:
        actual_dim = len(vector)
        if actual_dim != expected_dim:
            raise StorageError(
                f"Vector dimension mismatch: expected {expected_dim}, got {actual_dim}",
                error_type="INVALID_VECTOR",
            )

    def _get_collection_vector_names(self) -> List[str]:
        """Return the named-vector keys configured on the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            cfg = info.config.params.vectors
            if isinstance(cfg, dict):
                return list(cfg.keys())
            return []
        except Exception:
            return []

    def _ensure_named_vector_exists(self, name: str, size: int) -> None:
        """No-op: all standard dimension vectors are pre-created at collection init."""
        # All supported Matryoshka dimensions are created upfront in
        # initialize_collection(), so there is nothing to do here.
        # This method is kept for API compatibility.
        pass

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    # All supported Matryoshka dimensions — pre-created so _ensure_named_vector_exists
    # never needs to patch the collection schema at runtime.
    MATRYOSHKA_DIMENSIONS = [128, 256, 512, 756, 1024, 1536, 2048, 3072]

    def initialize_collection(
        self, dimension: int = 756, enable_named_vectors: bool = True
    ) -> None:
        """Create the Qdrant collection if it doesn't already exist.

        When *enable_named_vectors* is True, ALL supported Matryoshka
        dimensions are registered upfront so that any dimension can be
        stored or queried without schema changes later.

        Args:
            dimension: Default vector dimension (used for single-vector mode).
            enable_named_vectors: Use named-vector config for two-stage retrieval.
        """
        try:
            if self._check_collection_exists():
                return

            if enable_named_vectors:
                vectors_config = {
                    f"dim_{d}": VectorParams(size=d, distance=Distance.COSINE)
                    for d in self.MATRYOSHKA_DIMENSIONS
                }
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                )
            else:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                )

            # Payload indexes for efficient filtering
            for field, schema in [
                ("content_type", PayloadSchemaType.KEYWORD),
                ("source_id", PayloadSchemaType.KEYWORD),
                ("timestamp", PayloadSchemaType.DATETIME),
                ("language", PayloadSchemaType.KEYWORD),
            ]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Failed to initialize collection '{self.collection_name}': {str(e)}",
                error_type="STORAGE_ERROR",
            )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_embedding(
        self,
        vector: List[float],
        metadata: EmbeddingMetadata,
        point_id: Optional[str] = None,
    ) -> str:
        """Store a single embedding vector with metadata.

        The vector is stored under the named key ``dim_<dimension>`` so it
        is compatible with collections created with named-vector config.
        """
        try:
            if not self._check_collection_exists():
                raise StorageError(
                    f"Collection '{self.collection_name}' does not exist.",
                    error_type="COLLECTION_NOT_FOUND",
                )

            self._validate_vector_dimension(vector, metadata.dimension)

            if point_id is None:
                point_id = str(uuid.uuid4())

            vector_name = f"dim_{metadata.dimension}"
            self._ensure_named_vector_exists(vector_name, metadata.dimension)

            payload = self._build_payload(metadata)

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={vector_name: vector},
                        payload=payload,
                    )
                ],
            )
            return point_id

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Failed to store embedding: {str(e)}",
                error_type="STORAGE_ERROR",
            )

    def store_embedding_with_named_vectors(
        self,
        vectors: Dict[str, List[float]],
        metadata: EmbeddingMetadata,
        point_id: Optional[str] = None,
    ) -> str:
        """Store multiple dimension embeddings for the same content."""
        try:
            if point_id is None:
                point_id = str(uuid.uuid4())

            # Ensure every named vector exists in the collection schema
            for name, vec in vectors.items():
                self._ensure_named_vector_exists(name, len(vec))

            payload = self._build_payload(metadata)
            payload["dimensions"] = list(vectors.keys())

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(id=point_id, vector=vectors, payload=payload)
                ],
            )
            return point_id

        except Exception as e:
            raise StorageError(
                f"Failed to store embedding with named vectors: {str(e)}",
                error_type="STORAGE_ERROR",
            )

    def store_batch(
        self, embeddings: List[Tuple[List[float], EmbeddingMetadata]]
    ) -> List[str]:
        """Store multiple embeddings efficiently in a single upsert."""
        try:
            points = []
            point_ids = []

            for vector, metadata in embeddings:
                pid = str(uuid.uuid4())
                point_ids.append(pid)
                vector_name = f"dim_{metadata.dimension}"
                self._ensure_named_vector_exists(vector_name, metadata.dimension)
                payload = self._build_payload(metadata)
                points.append(
                    PointStruct(id=pid, vector={vector_name: vector}, payload=payload)
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            return point_ids

        except Exception as e:
            raise StorageError(
                f"Failed to store batch embeddings: {str(e)}",
                error_type="STORAGE_ERROR",
            )

    # ------------------------------------------------------------------
    # Search  (qdrant-client v1.x uses query_points)
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[SearchFilters] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors using ``query_points`` (client v1.x)."""
        try:
            qdrant_filter = self._build_filter(filters) if filters else None
            dim = len(query_vector)
            vector_name = f"dim_{dim}"

            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
            return self._parse_results(result.points)

        except Exception as e:
            raise StorageError(
                f"Failed to search vectors: {str(e)}",
                error_type="STORAGE_ERROR",
            )

    def search_with_named_vector(
        self,
        query_vector: List[float],
        vector_name: str,
        limit: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Search using a specific named vector."""
        try:
            qdrant_filter = self._build_filter(filters) if filters else None

            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            return self._parse_results(result.points)

        except Exception as e:
            raise StorageError(
                f"Failed to search with named vector '{vector_name}': {str(e)}",
                error_type="STORAGE_ERROR",
            )

    # ------------------------------------------------------------------
    # Point retrieval / deletion
    # ------------------------------------------------------------------

    def get_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                return None
            p = points[0]
            return {"id": str(p.id), "vector": p.vector, "payload": p.payload}
        except Exception as e:
            raise StorageError(
                f"Failed to retrieve point '{point_id}': {str(e)}",
                error_type="POINT_NOT_FOUND",
            )

    def delete_by_id(self, point_id: str) -> bool:
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id],
            )
            return True
        except Exception as e:
            raise StorageError(
                f"Failed to delete point '{point_id}': {str(e)}",
                error_type="STORAGE_ERROR",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_payload(metadata: EmbeddingMetadata) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "content_type": metadata.content_type,
            "source_id": metadata.source_id,
            "timestamp": metadata.timestamp.isoformat(),
            "dimension": metadata.dimension,
            "model_version": metadata.model_version,
        }
        if metadata.language is not None:
            payload["language"] = metadata.language
        if metadata.duration is not None:
            payload["duration"] = metadata.duration
        if metadata.page_count is not None:
            payload["page_count"] = metadata.page_count
        if metadata.custom_metadata is not None:
            payload["custom_metadata"] = metadata.custom_metadata
        return payload

    @staticmethod
    def _parse_results(points) -> List[SearchResult]:
        results = []
        for p in points:
            payload = p.payload
            timestamp = datetime.fromisoformat(payload["timestamp"])
            metadata = EmbeddingMetadata(
                content_type=payload["content_type"],
                source_id=payload["source_id"],
                timestamp=timestamp,
                dimension=payload["dimension"],
                model_version=payload["model_version"],
                language=payload.get("language"),
                duration=payload.get("duration"),
                page_count=payload.get("page_count"),
                custom_metadata=payload.get("custom_metadata"),
            )
            results.append(
                SearchResult(
                    point_id=str(p.id),
                    score=p.score,
                    content_type=payload["content_type"],
                    source_id=payload["source_id"],
                    timestamp=timestamp,
                    metadata=metadata,
                )
            )
        return results

    def _build_filter(self, filters: SearchFilters) -> Optional[Filter]:
        """Build a Qdrant Filter from SearchFilters (client v1.x compatible)."""
        conditions = []

        if filters.content_types:
            conditions.append(
                FieldCondition(
                    key="content_type",
                    match=MatchAny(any=filters.content_types),
                )
            )

        if filters.source_ids:
            conditions.append(
                FieldCondition(
                    key="source_id",
                    match=MatchAny(any=filters.source_ids),
                )
            )

        if filters.timestamp_from or filters.timestamp_to:
            range_kwargs: Dict[str, Any] = {}
            if filters.timestamp_from:
                range_kwargs["gte"] = filters.timestamp_from.isoformat()
            if filters.timestamp_to:
                range_kwargs["lte"] = filters.timestamp_to.isoformat()
            conditions.append(
                FieldCondition(key="timestamp", range=Range(**range_kwargs))
            )

        if filters.languages:
            conditions.append(
                FieldCondition(
                    key="language",
                    match=MatchAny(any=filters.languages),
                )
            )

        if filters.custom_filters:
            for key, value in filters.custom_filters.items():
                if key == "_point_ids":
                    conditions.append(HasIdCondition(has_id=value))
                elif isinstance(value, list):
                    conditions.append(
                        FieldCondition(
                            key=f"custom_metadata.{key}",
                            match=MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=f"custom_metadata.{key}",
                            match=MatchValue(value=value),
                        )
                    )

        return Filter(must=conditions) if conditions else None
