"""High-level API for the multimodal search system.

This module provides :class:`MultimodalSearchAPI`, the primary entry point
for embedding content and performing semantic search.  It wires together
:class:`~multimodal_search.content_processor.ContentProcessor`,
:class:`~multimodal_search.embedding_service.EmbeddingService`,
:class:`~multimodal_search.vector_store.VectorStore`, and
:class:`~multimodal_search.search_engine.SearchEngine` into a single
cohesive interface.

Typical usage::

    # From environment variables
    api = MultimodalSearchAPI.from_env()
    api.initialize_system()

    # Embed and store content
    item = ContentItem(content_type="text", data="Hello, world!")
    response = api.embed_content(item, store=True)

    # Search
    query = ContentItem(content_type="text", data="greeting")
    results = api.search(query, limit=5)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from multimodal_search.content_processor import ContentProcessor
from multimodal_search.embedding_service import EmbeddingService
from multimodal_search.vector_store import VectorStore
from multimodal_search.search_engine import SearchEngine
from multimodal_search.models import (
    ContentItem,
    SearchFilters,
    SearchResponse,
    StageConfig,
    VertexAIConfig,
    QdrantConfig,
)
from multimodal_search.exceptions import (
    ValidationError,
    EmbeddingError,
    StorageError,
    SearchError,
    MultimodalSearchError,
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResponse:
    """Response from :meth:`MultimodalSearchAPI.embed_content`.

    Attributes:
        vector: The embedding vector produced by Vertex AI.
        dimension: Length of the vector (Matryoshka dimension used).
        content_type: Modality of the embedded content.
        model_version: Name of the model that produced the embedding.
        point_id: Qdrant point ID assigned when ``store=True``.
            ``None`` when the embedding was not stored.
        metadata: Optional additional information from the embedding
            operation.

    Example:
        >>> resp = api.embed_content(item, store=True)
        >>> resp.point_id is not None
        True
    """

    vector: List[float]
    dimension: int
    content_type: str
    model_version: str
    point_id: Optional[str] = None  # Set when store=True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchEmbeddingResponse:
    """Response from :meth:`MultimodalSearchAPI.embed_batch`.

    Attributes:
        results: Per-item :class:`EmbeddingResponse` objects in the same
            order as the input ``content_items`` list.
        total: Total number of items processed.
        stored: Number of items successfully stored in Qdrant.

    Example:
        >>> resp = api.embed_batch(items, store=True)
        >>> resp.total == len(items)
        True
    """

    results: List[EmbeddingResponse]
    total: int
    stored: int  # Number of items actually stored in Qdrant


@dataclass
class SystemStatus:
    """Status returned by :meth:`MultimodalSearchAPI.initialize_system`.

    Attributes:
        initialized: ``True`` only when all sub-systems are ready.
        vertex_ai_connected: ``True`` if the Vertex AI client is
            reachable.
        qdrant_connected: ``True`` if the Qdrant instance is reachable.
        collection_ready: ``True`` if the Qdrant collection was created
            or already existed.
        default_dimension: The active default Matryoshka dimension.
        two_stage_enabled: Whether two-stage retrieval is enabled.
        message: Human-readable summary of the initialisation outcome.
        errors: List of error strings for any sub-system that failed.

    Example:
        >>> status = api.initialize_system()
        >>> if not status.initialized:
        ...     for err in status.errors:
        ...         print(err)
    """

    initialized: bool
    vertex_ai_connected: bool
    qdrant_connected: bool
    collection_ready: bool
    default_dimension: int
    two_stage_enabled: bool
    message: str
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _vertex_ai_config_from_env() -> VertexAIConfig:
    """Build a :class:`~multimodal_search.models.VertexAIConfig` from environment variables.

    Reads the following environment variables:

    * ``VERTEX_AI_PROJECT_ID`` (required) – GCP project ID.
    * ``VERTEX_AI_LOCATION`` (optional, default: ``"us-central1"``) – GCP
      region.
    * ``GOOGLE_APPLICATION_CREDENTIALS`` (optional) – path to a service
      account JSON key file.

    Returns:
        A populated :class:`~multimodal_search.models.VertexAIConfig`.
        ``project_id`` will be an empty string if the environment variable
        is not set; call :func:`_validate_vertex_ai_config` to detect this.
    """
    project_id = os.environ.get("VERTEX_AI_PROJECT_ID", "")
    location = os.environ.get("VERTEX_AI_LOCATION", "global")
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    api_key = os.environ.get("VERTEX_AI_API_KEY")
    return VertexAIConfig(
        project_id=project_id,
        location=location,
        credentials_path=credentials_path,
        api_key=api_key,
    )


def _qdrant_config_from_env() -> QdrantConfig:
    """Build a :class:`~multimodal_search.models.QdrantConfig` from environment variables.

    Reads the following environment variables:

    * ``QDRANT_URL`` (required) – base URL of the Qdrant instance.
    * ``QDRANT_API_KEY`` (optional) – API key for Qdrant Cloud.
    * ``QDRANT_COLLECTION_NAME`` (optional, default:
      ``"multimodal_embeddings"``) – collection name.

    Returns:
        A populated :class:`~multimodal_search.models.QdrantConfig`.
        ``url`` will be an empty string if the environment variable is not
        set; call :func:`_validate_qdrant_config` to detect this.
    """
    url = os.environ.get("QDRANT_URL", "")
    api_key = os.environ.get("QDRANT_API_KEY")
    collection_name = os.environ.get("QDRANT_COLLECTION_NAME", "multimodal_embeddings")
    return QdrantConfig(
        url=url,
        api_key=api_key,
        collection_name=collection_name,
    )


def _validate_vertex_ai_config(config: VertexAIConfig) -> List[str]:
    """Validate a :class:`~multimodal_search.models.VertexAIConfig` object.

    Args:
        config: The configuration to validate.

    Returns:
        A list of human-readable error strings.  An empty list means the
        configuration is valid.
    """
    errors: List[str] = []
    if not config.project_id:
        errors.append(
            "VertexAIConfig.project_id is required. "
            "Set VERTEX_AI_PROJECT_ID environment variable or pass it explicitly."
        )
    if not config.location:
        errors.append("VertexAIConfig.location must not be empty.")
    if not config.model:
        errors.append("VertexAIConfig.model must not be empty.")
    return errors


def _validate_qdrant_config(config: QdrantConfig) -> List[str]:
    """Validate a :class:`~multimodal_search.models.QdrantConfig` object.

    Args:
        config: The configuration to validate.

    Returns:
        A list of human-readable error strings.  An empty list means the
        configuration is valid.
    """
    errors: List[str] = []
    if not config.url:
        errors.append(
            "QdrantConfig.url is required. "
            "Set QDRANT_URL environment variable or pass it explicitly."
        )
    if not config.collection_name:
        errors.append("QdrantConfig.collection_name must not be empty.")
    if config.distance_metric not in ("cosine", "dot", "euclid"):
        errors.append(
            f"QdrantConfig.distance_metric '{config.distance_metric}' is not valid. "
            "Use 'cosine', 'dot', or 'euclid'."
        )
    return errors


# ---------------------------------------------------------------------------
# Main API class
# ---------------------------------------------------------------------------

class MultimodalSearchAPI:
    """High-level API that wires together all system components.

    Usage (explicit config)::

        api = MultimodalSearchAPI.from_config(
            vertex_ai_config=VertexAIConfig(project_id="my-project"),
            qdrant_config=QdrantConfig(url="http://localhost:6333"),
        )

    Usage (environment variables)::

        # Set VERTEX_AI_PROJECT_ID and QDRANT_URL first
        api = MultimodalSearchAPI.from_env()

    After construction call :meth:`initialize_system` to create the Qdrant
    collection and confirm connectivity before processing requests.
    """

    DEFAULT_DIMENSION = 756

    def __init__(
        self,
        content_processor: ContentProcessor,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        search_engine: SearchEngine,
        default_dimension: int = DEFAULT_DIMENSION,
        enable_two_stage: bool = True,
    ) -> None:
        """Initialise the API with pre-constructed component instances.

        Prefer the factory methods :meth:`from_config` or
        :meth:`from_env` over calling this constructor directly.

        Args:
            content_processor: Validates content before embedding.
            embedding_service: Generates embeddings via Vertex AI.
            vector_store: Stores and retrieves vectors in Qdrant.
            search_engine: Orchestrates search operations.
            default_dimension: Default Matryoshka dimension used when
                ``dimension`` is not specified in API calls.
            enable_two_stage: When ``True``, the Qdrant collection is
                configured with named vectors to support two-stage
                retrieval.
        """
        self._content_processor = content_processor
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._search_engine = search_engine
        self.default_dimension = default_dimension
        self.enable_two_stage = enable_two_stage

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        vertex_ai_config: VertexAIConfig,
        qdrant_config: QdrantConfig,
        default_dimension: int = DEFAULT_DIMENSION,
        enable_two_stage: bool = True,
    ) -> "MultimodalSearchAPI":
        """Create an API instance from explicit config objects.

        Raises:
            ValueError: If configuration is invalid.
            EmbeddingError: If Vertex AI initialisation fails.
            StorageError: If Qdrant connection fails.
        """
        # Validate configs
        errors = _validate_vertex_ai_config(vertex_ai_config) + _validate_qdrant_config(qdrant_config)
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

        # Set credentials env var if provided
        if vertex_ai_config.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = vertex_ai_config.credentials_path

        content_processor = ContentProcessor()
        embedding_service = EmbeddingService(
            project_id=vertex_ai_config.project_id,
            location=vertex_ai_config.location,
            api_key=getattr(vertex_ai_config, "api_key", None),
        )
        vector_store = VectorStore(
            qdrant_url=qdrant_config.url,
            api_key=qdrant_config.api_key,
        )
        vector_store.collection_name = qdrant_config.collection_name

        search_engine = SearchEngine(
            embedding_service=embedding_service,
            vector_store=vector_store,
            default_dimension=default_dimension,
        )

        return cls(
            content_processor=content_processor,
            embedding_service=embedding_service,
            vector_store=vector_store,
            search_engine=search_engine,
            default_dimension=default_dimension,
            enable_two_stage=enable_two_stage,
        )

    @classmethod
    def from_env(
        cls,
        default_dimension: int = DEFAULT_DIMENSION,
        enable_two_stage: bool = True,
    ) -> "MultimodalSearchAPI":
        """Create an API instance from environment variables.

        Required env vars: VERTEX_AI_PROJECT_ID, QDRANT_URL
        Optional env vars: VERTEX_AI_LOCATION, GOOGLE_APPLICATION_CREDENTIALS,
                           QDRANT_API_KEY, QDRANT_COLLECTION_NAME

        Raises:
            ValueError: If required environment variables are missing.
            EmbeddingError: If Vertex AI initialisation fails.
            StorageError: If Qdrant connection fails.
        """
        vertex_ai_config = _vertex_ai_config_from_env()
        qdrant_config = _qdrant_config_from_env()
        return cls.from_config(
            vertex_ai_config=vertex_ai_config,
            qdrant_config=qdrant_config,
            default_dimension=default_dimension,
            enable_two_stage=enable_two_stage,
        )

    # ------------------------------------------------------------------
    # System initialisation
    # ------------------------------------------------------------------

    def initialize_system(
        self,
        vertex_ai_config: Optional[VertexAIConfig] = None,
        qdrant_config: Optional[QdrantConfig] = None,
        default_dimension: Optional[int] = None,
        enable_two_stage: Optional[bool] = None,
    ) -> SystemStatus:
        """Initialise the system: validate connectivity and create the Qdrant collection.

        Can optionally accept new config objects to reconfigure the instance.

        Args:
            vertex_ai_config: Optional new Vertex AI config (ignored if None).
            qdrant_config: Optional new Qdrant config (ignored if None).
            default_dimension: Override the default embedding dimension.
            enable_two_stage: Override the two-stage retrieval flag.

        Returns:
            SystemStatus describing the initialisation outcome.
        """
        errors: List[str] = []
        vertex_ok = False
        qdrant_ok = False
        collection_ok = False

        if default_dimension is not None:
            self.default_dimension = default_dimension
        if enable_two_stage is not None:
            self.enable_two_stage = enable_two_stage

        # Validate Vertex AI connectivity (lightweight check)
        try:
            # Attempt a trivial operation to confirm the client is alive
            _ = self._embedding_service.project_id
            vertex_ok = True
        except Exception as exc:
            errors.append(f"Vertex AI connectivity check failed: {exc}")

        # Validate Qdrant connectivity and create collection
        try:
            self._vector_store.initialize_collection(
                dimension=self.default_dimension,
                enable_named_vectors=self.enable_two_stage,
            )
            qdrant_ok = True
            collection_ok = True
        except StorageError as exc:
            errors.append(f"Qdrant initialisation failed: {exc}")
        except Exception as exc:
            errors.append(f"Qdrant initialisation failed: {exc}")

        initialized = vertex_ok and qdrant_ok and collection_ok
        message = "System initialised successfully." if initialized else "System initialisation completed with errors."

        return SystemStatus(
            initialized=initialized,
            vertex_ai_connected=vertex_ok,
            qdrant_connected=qdrant_ok,
            collection_ready=collection_ok,
            default_dimension=self.default_dimension,
            two_stage_enabled=self.enable_two_stage,
            message=message,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Embedding endpoints
    # ------------------------------------------------------------------

    def embed_content(
        self,
        content: ContentItem,
        dimension: Optional[int] = None,
        store: bool = True,
        named_vectors: Optional[List[int]] = None,
    ) -> EmbeddingResponse:
        """Embed a single content item and optionally store it in Qdrant.

        Args:
            content: Content to embed (text, image, audio, video, pdf, or interleaved).
            dimension: Matryoshka dimension (128-3072). Defaults to instance default.
            store: Whether to persist the embedding in Qdrant.
            named_vectors: Additional dimensions to store as named vectors for
                           two-stage retrieval (e.g., [256, 1024]).

        Returns:
            EmbeddingResponse with the vector and optional point_id.

        Raises:
            ValidationError: Content failed validation.
            EmbeddingError: Vertex AI API error.
            StorageError: Qdrant storage error.
        """
        dim = dimension if dimension is not None else self.default_dimension

        # Validate content
        self._validate_content(content)

        # Generate embedding
        try:
            embedding_result = self._embed_single(content, dim)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}", error_type="API_ERROR")

        point_id: Optional[str] = None

        if store:
            try:
                from datetime import datetime, timezone
                from multimodal_search.models import EmbeddingMetadata

                metadata = EmbeddingMetadata(
                    content_type=content.content_type,
                    source_id=content.source_id or "",
                    timestamp=datetime.now(timezone.utc),
                    dimension=dim,
                    model_version=embedding_result.model_version,
                    custom_metadata=content.metadata,
                )

                if named_vectors:
                    # Store multiple dimensions as named vectors
                    vectors: Dict[str, List[float]] = {f"dim_{dim}": embedding_result.vector}
                    for extra_dim in named_vectors:
                        if extra_dim != dim:
                            extra_result = self._embed_single(content, extra_dim)
                            vectors[f"dim_{extra_dim}"] = extra_result.vector
                    point_id = self._vector_store.store_embedding_with_named_vectors(
                        vectors=vectors,
                        metadata=metadata,
                    )
                else:
                    point_id = self._vector_store.store_embedding(
                        vector=embedding_result.vector,
                        metadata=metadata,
                    )
            except StorageError:
                raise
            except Exception as exc:
                raise StorageError(f"Failed to store embedding: {exc}", error_type="STORAGE_ERROR")

        return EmbeddingResponse(
            vector=embedding_result.vector,
            dimension=dim,
            content_type=embedding_result.content_type,
            model_version=embedding_result.model_version,
            point_id=point_id,
        )

    def embed_batch(
        self,
        content_items: List[ContentItem],
        dimension: Optional[int] = None,
        store: bool = True,
    ) -> BatchEmbeddingResponse:
        """Embed multiple content items in a single call.

        Args:
            content_items: List of content items to embed.
            dimension: Matryoshka dimension. Defaults to instance default.
            store: Whether to persist embeddings in Qdrant.

        Returns:
            BatchEmbeddingResponse with per-item results.

        Raises:
            ValidationError: One or more items failed validation.
            EmbeddingError: Vertex AI API error.
            StorageError: Qdrant storage error.
        """
        dim = dimension if dimension is not None else self.default_dimension

        if not content_items:
            return BatchEmbeddingResponse(results=[], total=0, stored=0)

        # Validate all items first (fail fast)
        validation_results = self._content_processor.validate_batch(content_items)
        failed = [
            (i, r) for i, r in enumerate(validation_results) if not r.valid
        ]
        if failed:
            idx, first_failure = failed[0]
            raise ValidationError(
                f"Batch validation failed at item {idx}: {first_failure.error_message}",
                error_type=first_failure.error_type or "VALIDATION_ERROR",
            )

        # Generate embeddings
        try:
            embedding_results = self._embedding_service.embed_batch(content_items, dim)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Batch embedding failed: {exc}", error_type="API_ERROR")

        responses: List[EmbeddingResponse] = []
        stored_count = 0

        for i, (item, emb) in enumerate(zip(content_items, embedding_results)):
            point_id: Optional[str] = None

            if store:
                try:
                    from datetime import datetime, timezone
                    from multimodal_search.models import EmbeddingMetadata

                    metadata = EmbeddingMetadata(
                        content_type=item.content_type,
                        source_id=item.source_id or "",
                        timestamp=datetime.now(timezone.utc),
                        dimension=dim,
                        model_version=emb.model_version,
                        custom_metadata=item.metadata,
                    )
                    point_id = self._vector_store.store_embedding(
                        vector=emb.vector,
                        metadata=metadata,
                    )
                    stored_count += 1
                except StorageError:
                    raise
                except Exception as exc:
                    raise StorageError(
                        f"Failed to store embedding for item {i}: {exc}",
                        error_type="STORAGE_ERROR",
                    )

            responses.append(
                EmbeddingResponse(
                    vector=emb.vector,
                    dimension=dim,
                    content_type=emb.content_type,
                    model_version=emb.model_version,
                    point_id=point_id,
                )
            )

        return BatchEmbeddingResponse(
            results=responses,
            total=len(responses),
            stored=stored_count,
        )

    # ------------------------------------------------------------------
    # Search endpoints
    # ------------------------------------------------------------------

    def search(
        self,
        query: ContentItem,
        limit: int = 10,
        filters: Optional[SearchFilters] = None,
        dimension: Optional[int] = None,
        score_threshold: Optional[float] = None,
        include_vectors: bool = False,
    ) -> SearchResponse:
        """Perform single-stage semantic search.

        Args:
            query: Query content (any modality).
            limit: Maximum number of results.
            filters: Optional search filters (modality, time range, etc.).
            dimension: Embedding dimension. Defaults to instance default.
            score_threshold: Minimum similarity score (0-1).
            include_vectors: Include raw vectors in results (not yet surfaced by
                             the underlying SearchEngine but accepted for API
                             compatibility).

        Returns:
            SearchResponse with ranked results.

        Raises:
            ValidationError: Query validation failed.
            SearchError: Search operation failed.
        """
        dim = dimension if dimension is not None else self.default_dimension

        # Validate query
        self._validate_content(query)

        try:
            modality_filter = filters.content_types if filters else None
            response = self._search_engine.search(
                query=query,
                limit=limit,
                modality_filter=modality_filter,
                dimension=dim,
                score_threshold=score_threshold,
            )
            # Merge remaining filters (source_ids, timestamp, languages, custom)
            if filters and self._has_extra_filters(filters):
                response = self._apply_extra_filters(response, filters)
            return response
        except (ValidationError, SearchError):
            raise
        except Exception as exc:
            raise SearchError(f"Search failed: {exc}", error_type="SEARCH_ERROR")

    def search_two_stage(
        self,
        query: ContentItem,
        first_stage_config: StageConfig,
        second_stage_config: StageConfig,
        filters: Optional[SearchFilters] = None,
    ) -> SearchResponse:
        """Perform two-stage retrieval for speed-accuracy optimisation.

        Args:
            query: Query content (any modality).
            first_stage_config: Config for initial fast retrieval (dimension, limit).
            second_stage_config: Config for accurate re-ranking (dimension, limit).
            filters: Optional search filters.

        Returns:
            SearchResponse with re-ranked results.

        Raises:
            ValidationError: Query validation failed.
            SearchError: Search operation failed.
        """
        # Validate query
        self._validate_content(query)

        try:
            modality_filter = filters.content_types if filters else None
            return self._search_engine.search_two_stage(
                query=query,
                first_stage_dimension=first_stage_config.dimension,
                second_stage_dimension=second_stage_config.dimension,
                first_stage_limit=first_stage_config.limit,
                final_limit=second_stage_config.limit,
                modality_filter=modality_filter,
            )
        except (ValidationError, SearchError):
            raise
        except Exception as exc:
            raise SearchError(f"Two-stage search failed: {exc}", error_type="SEARCH_ERROR")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_content(self, content: ContentItem) -> None:
        """Validate content using the appropriate modality validator.

        Dispatches to the correct
        :class:`~multimodal_search.content_processor.ContentProcessor`
        method based on ``content.content_type``.

        Args:
            content: The content item to validate.

        Raises:
            ValidationError: If the content fails validation or has an
                unknown ``content_type``.
        """
        ct = content.content_type
        if ct == "text":
            result = self._content_processor.validate_text(str(content.data or ""))
        elif ct == "image":
            result = self._content_processor.validate_image(
                bytes(content.data or b""), content.mime_type or ""
            )
        elif ct == "audio":
            result = self._content_processor.validate_audio(
                bytes(content.data or b""), content.mime_type or ""
            )
        elif ct == "video":
            result = self._content_processor.validate_video(
                bytes(content.data or b""), content.mime_type or ""
            )
        elif ct == "pdf":
            result = self._content_processor.validate_pdf(bytes(content.data or b""))
        elif ct == "interleaved":
            result = self._content_processor.validate_interleaved(content)
        else:
            raise ValidationError(
                f"Unknown content type: '{ct}'",
                error_type="INVALID_FORMAT",
            )

        if not result.valid:
            raise ValidationError(
                result.error_message or "Content validation failed",
                error_type=result.error_type or "VALIDATION_ERROR",
            )

    def _embed_single(self, content: ContentItem, dimension: int):
        """Dispatch to the correct embedding method based on content type.

        Args:
            content: The content item to embed.
            dimension: Matryoshka dimension to use.

        Returns:
            :class:`~multimodal_search.models.EmbeddingResult` from the
            appropriate embedding method.

        Raises:
            EmbeddingError: If the content type is unknown or the
                embedding call fails.
        """
        ct = content.content_type
        if ct == "text":
            return self._embedding_service.embed_text(str(content.data), dimension)
        elif ct == "image":
            return self._embedding_service.embed_image(bytes(content.data), dimension)
        elif ct == "audio":
            return self._embedding_service.embed_audio(bytes(content.data), dimension)
        elif ct == "video":
            return self._embedding_service.embed_video(bytes(content.data), dimension)
        elif ct == "pdf":
            return self._embedding_service.embed_pdf(bytes(content.data), dimension)
        elif ct == "interleaved":
            return self._embedding_service.embed_interleaved(content, dimension)
        else:
            raise EmbeddingError(
                f"Unknown content type: '{ct}'",
                error_type="API_ERROR",
            )

    @staticmethod
    def _has_extra_filters(filters: SearchFilters) -> bool:
        """Return ``True`` if *filters* contains fields beyond ``content_types``.

        The :class:`~multimodal_search.search_engine.SearchEngine` only
        handles ``content_types`` natively.  All other filter fields
        require post-processing via :meth:`_apply_extra_filters`.

        Args:
            filters: The search filters to inspect.

        Returns:
            ``True`` if any of ``source_ids``, ``timestamp_from``,
            ``timestamp_to``, ``languages``, or ``custom_filters`` are
            set.
        """
        return bool(
            filters.source_ids
            or filters.timestamp_from
            or filters.timestamp_to
            or filters.languages
            or filters.custom_filters
        )

    @staticmethod
    def _apply_extra_filters(response: SearchResponse, filters: SearchFilters) -> SearchResponse:
        """Post-filter search results for fields not handled by the search engine.

        Applies in-memory filtering for ``source_ids``, timestamp range,
        ``languages``, and ``custom_filters`` after the vector search has
        already been performed.

        Args:
            response: The raw :class:`~multimodal_search.models.SearchResponse`
                from the search engine.
            filters: The full set of filters to apply.

        Returns:
            A new :class:`~multimodal_search.models.SearchResponse` with
            only the results that pass all extra filters.  ``total_results``
            is updated accordingly.
        """
        results = response.results

        if filters.source_ids:
            sid_set = set(filters.source_ids)
            results = [r for r in results if r.source_id in sid_set]

        if filters.timestamp_from:
            results = [r for r in results if r.timestamp >= filters.timestamp_from]

        if filters.timestamp_to:
            results = [r for r in results if r.timestamp <= filters.timestamp_to]

        if filters.languages:
            lang_set = set(filters.languages)
            results = [
                r for r in results
                if r.metadata.language and r.metadata.language in lang_set
            ]

        return SearchResponse(
            results=results,
            query_metadata=response.query_metadata,
            total_results=len(results),
            search_time_ms=response.search_time_ms,
            two_stage=response.two_stage,
        )
