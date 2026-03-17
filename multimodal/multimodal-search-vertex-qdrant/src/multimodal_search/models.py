"""Data models for the multimodal search system.

This module defines all shared dataclasses used across the system:
content representation, embedding results, search responses, validation
results, filters, and configuration objects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class InterleavedPart:
    """A single modality component within an interleaved multimodal content item.

    Used to build a list of parts for a :class:`ContentItem` whose
    ``content_type`` is ``"interleaved"``.  Each part carries its own
    modality type, raw data, and optional MIME type.

    Attributes:
        content_type: Modality of this part.  One of ``"text"``,
            ``"image"``, ``"audio"``, ``"video"``, or ``"pdf"``.
        data: Raw content — a plain string for text, or ``bytes`` for
            binary modalities.
        mime_type: MIME type of the binary data (e.g. ``"image/jpeg"``,
            ``"audio/wav"``).  Required for image, audio, and video parts.

    Example:
        >>> from multimodal_search.models import InterleavedPart
        >>> text_part = InterleavedPart(content_type="text", data="A cat sitting on a mat.")
        >>> image_part = InterleavedPart(
        ...     content_type="image", data=b"<jpeg bytes>", mime_type="image/jpeg"
        ... )
    """

    content_type: str  # "text", "image", "audio", "video", "pdf"
    data: Union[str, bytes]  # Text string or binary data
    mime_type: Optional[str] = None  # e.g., "image/jpeg", "audio/mp3"


@dataclass
class ContentItem:
    """Represents content to be embedded or used as a search query.

    Supports two operating modes:

    * **Single-modality** – set ``content_type`` to one of ``"text"``,
      ``"image"``, ``"audio"``, ``"video"``, or ``"pdf"`` and populate
      ``data`` (and ``mime_type`` for binary modalities).
    * **Interleaved multimodal** – set ``content_type="interleaved"`` and
      populate ``interleaved_parts`` with a list of
      :class:`InterleavedPart` objects.  ``data`` should be ``None`` in
      this mode.

    Attributes:
        content_type: Modality identifier.  One of ``"text"``,
            ``"image"``, ``"audio"``, ``"video"``, ``"pdf"``, or
            ``"interleaved"``.
        data: Raw content.  A plain string for text; ``bytes`` for binary
            modalities; ``None`` when ``content_type="interleaved"``.
        mime_type: MIME type for binary content (e.g. ``"image/jpeg"``).
        source_id: Caller-supplied identifier for the original source
            (e.g. a file path, database key, or URL).
        metadata: Arbitrary key-value pairs stored alongside the
            embedding in Qdrant.
        interleaved_parts: Ordered list of :class:`InterleavedPart`
            objects.  Only used when ``content_type="interleaved"``.

    Example:
        >>> # Single-modality text item
        >>> item = ContentItem(
        ...     content_type="text",
        ...     data="Hello, world!",
        ...     source_id="doc-001",
        ... )
        >>> # Interleaved image + caption
        >>> from multimodal_search.models import InterleavedPart
        >>> item = ContentItem(
        ...     content_type="interleaved",
        ...     source_id="product-001",
        ...     interleaved_parts=[
        ...         InterleavedPart(content_type="image", data=b"<jpeg>", mime_type="image/jpeg"),
        ...         InterleavedPart(content_type="text", data="Red running shoes, size 10."),
        ...     ],
        ... )
    """

    content_type: str  # "text", "image", "audio", "video", "pdf", or "interleaved"
    data: Union[str, bytes, None] = None  # Text string or binary data (None for interleaved)
    mime_type: Optional[str] = None  # e.g., "image/jpeg", "audio/mp3"
    source_id: Optional[str] = None  # Original source identifier
    metadata: Optional[Dict[str, Any]] = None  # Custom metadata
    interleaved_parts: Optional[List[InterleavedPart]] = None  # Parts for interleaved content


@dataclass
class EmbeddingMetadata:
    """Metadata stored alongside an embedding vector in Qdrant.

    Every point stored in the Qdrant collection carries this metadata as
    its payload, enabling rich filtering during search.

    Attributes:
        content_type: Modality of the embedded content (``"text"``,
            ``"image"``, ``"audio"``, ``"video"``, or ``"pdf"``).
        source_id: Unique identifier for the original source content.
        timestamp: UTC datetime when the content was indexed.
        dimension: Matryoshka dimension used to generate the embedding.
        model_version: Name of the embedding model (e.g.
            ``"gemini-embedding-2-preview"``).
        language: BCP-47 language code for text content (e.g. ``"en"``).
        duration: Duration in seconds for audio or video content.
        page_count: Number of pages for PDF content.
        custom_metadata: Arbitrary user-defined key-value pairs.

    Example:
        >>> from datetime import datetime, timezone
        >>> meta = EmbeddingMetadata(
        ...     content_type="text",
        ...     source_id="doc-001",
        ...     timestamp=datetime.now(timezone.utc),
        ...     dimension=756,
        ...     model_version="gemini-embedding-2-preview",
        ...     language="en",
        ... )
    """

    content_type: str  # "text", "image", "audio", "video", "pdf"
    source_id: str  # Unique identifier for source content
    timestamp: datetime  # When content was indexed
    dimension: int  # Embedding dimension used
    model_version: str  # e.g., "gemini-embedding-2-preview"
    language: Optional[str] = None  # For text content
    duration: Optional[float] = None  # For audio/video (seconds)
    page_count: Optional[int] = None  # For PDFs
    custom_metadata: Optional[Dict[str, Any]] = None  # User-defined fields


@dataclass
class EmbeddingResult:
    """Result returned by an embedding operation.

    Attributes:
        vector: The embedding vector as a list of floats.
        dimension: Length of the vector (Matryoshka dimension used).
        content_type: Modality of the embedded content.
        model_version: Name of the model that produced the embedding.
        metadata: Optional additional information from the API response.

    Example:
        >>> result = EmbeddingResult(
        ...     vector=[0.1, -0.2, 0.3],
        ...     dimension=3,
        ...     content_type="text",
        ...     model_version="gemini-embedding-2-preview",
        ... )
        >>> len(result.vector) == result.dimension
        True
    """

    vector: List[float]  # Embedding vector
    dimension: int  # Vector dimension
    content_type: str  # Modality
    model_version: str  # Model used
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """A single result returned by a vector search operation.

    Attributes:
        point_id: Qdrant point identifier for the stored embedding.
        score: Cosine similarity score in the range [0, 1].  Higher is
            more similar.
        content_type: Modality of the matched content.
        source_id: Caller-supplied identifier of the original source.
        timestamp: UTC datetime when the content was indexed.
        metadata: Full :class:`EmbeddingMetadata` payload for the point.
        vector: Raw embedding vector, included only when explicitly
            requested.

    Example:
        >>> from datetime import datetime, timezone
        >>> result = SearchResult(
        ...     point_id="abc-123",
        ...     score=0.92,
        ...     content_type="image",
        ...     source_id="img-001",
        ...     timestamp=datetime.now(timezone.utc),
        ...     metadata=EmbeddingMetadata(...),
        ... )
    """

    point_id: str  # Qdrant point ID
    score: float  # Similarity score (0-1)
    content_type: str  # Modality
    source_id: str  # Original content identifier
    timestamp: datetime  # When indexed
    metadata: EmbeddingMetadata  # Full metadata
    vector: Optional[List[float]] = None  # Include if requested


@dataclass
class SearchResponse:
    """Complete response from a search operation.

    Attributes:
        results: List of :class:`SearchResult` objects ranked in
            descending order of similarity score.
        query_metadata: Dictionary describing the query parameters used
            (e.g. ``query_type``, ``dimension``, ``modality_filter``).
        total_results: Number of results in ``results``.
        search_time_ms: Wall-clock time for the search in milliseconds.
        two_stage: ``True`` when two-stage retrieval was used.

    Example:
        >>> response = SearchResponse(
        ...     results=[],
        ...     query_metadata={"query_type": "text", "dimension": 756},
        ...     total_results=0,
        ...     search_time_ms=12.5,
        ... )
    """

    results: List[SearchResult]  # Ranked results
    query_metadata: Dict[str, Any]  # Query information
    total_results: int  # Number of results returned
    search_time_ms: float  # Search duration
    two_stage: bool = False  # Whether two-stage retrieval was used


@dataclass
class ValidationResult:
    """Result from a content validation check.

    Attributes:
        valid: ``True`` if the content passed all validation rules.
        error_type: Short error code when ``valid=False`` (e.g.
            ``"INVALID_FORMAT"``, ``"SIZE_EXCEEDED"``).
        error_message: Human-readable description of the validation
            failure.
        warnings: Non-fatal warnings that do not prevent embedding (e.g.
            batch image count approaching the limit).

    Example:
        >>> result = ValidationResult(valid=True)
        >>> result.valid
        True
        >>> bad = ValidationResult(
        ...     valid=False,
        ...     error_type="INVALID_FORMAT",
        ...     error_message="Image format 'bmp' not supported.",
        ... )
    """

    valid: bool
    error_type: Optional[str] = None  # e.g., "INVALID_FORMAT", "SIZE_EXCEEDED"
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class SearchFilters:
    """Filters applied to a vector search query.

    All fields are optional and combined with AND logic.  Omitting a
    field means no restriction is applied for that dimension.

    Attributes:
        content_types: Restrict results to these modalities (e.g.
            ``["image", "video"]``).
        source_ids: Restrict results to these source identifiers.
        timestamp_from: Inclusive lower bound for the indexing timestamp.
        timestamp_to: Inclusive upper bound for the indexing timestamp.
        languages: Restrict text results to these BCP-47 language codes.
        custom_filters: Arbitrary key-value pairs matched against
            ``custom_metadata`` fields in the Qdrant payload.  The
            special key ``"_point_ids"`` filters by a list of Qdrant
            point IDs (used internally for two-stage re-ranking).

    Example:
        >>> from datetime import datetime, timezone
        >>> filters = SearchFilters(
        ...     content_types=["image", "pdf"],
        ...     languages=["en", "fr"],
        ...     timestamp_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ... )
    """

    content_types: Optional[List[str]] = None  # Filter by modality
    source_ids: Optional[List[str]] = None  # Filter by specific sources
    timestamp_from: Optional[datetime] = None  # Time range start
    timestamp_to: Optional[datetime] = None  # Time range end
    languages: Optional[List[str]] = None  # Filter by language (text only)
    custom_filters: Optional[Dict[str, Any]] = None  # User-defined filters


@dataclass
class StageConfig:
    """Configuration for one stage of a two-stage retrieval pipeline.

    Attributes:
        dimension: Matryoshka embedding dimension to use for this stage.
        limit: Maximum number of results to retrieve in this stage.

    Example:
        >>> first_stage = StageConfig(dimension=256, limit=100)
        >>> second_stage = StageConfig(dimension=1024, limit=10)
    """

    dimension: int
    limit: int


@dataclass
class VertexAIConfig:
    """Configuration for connecting to Vertex AI.

    Attributes:
        project_id: GCP project ID that has the Vertex AI API enabled.
        location: GCP region where the model is deployed (default:
            ``"us-central1"``).
        credentials_path: Optional filesystem path to a service account
            JSON key file.  When ``None``, Application Default
            Credentials are used.
        model: Embedding model name (default:
            ``"gemini-embedding-2-preview"``).

    Example:
        >>> config = VertexAIConfig(project_id="my-gcp-project")
        >>> config.location
        'us-central1'
    """

    project_id: str
    location: str = "us-central1"
    credentials_path: Optional[str] = None
    model: str = "gemini-embedding-2-preview"
    api_key: Optional[str] = None


@dataclass
class QdrantConfig:
    """Configuration for connecting to a Qdrant vector database.

    Attributes:
        url: Base URL of the Qdrant instance (e.g.
            ``"http://localhost:6333"`` or a Qdrant Cloud URL).
        api_key: API key for authenticated Qdrant Cloud instances.
            ``None`` for unauthenticated local instances.
        collection_name: Name of the Qdrant collection to use (default:
            ``"multimodal_embeddings"``).
        distance_metric: Vector distance metric — ``"cosine"``,
            ``"dot"``, or ``"euclid"`` (default: ``"cosine"``).
        enable_named_vectors: When ``True``, the collection is
            configured to support named vectors for two-stage retrieval.

    Example:
        >>> config = QdrantConfig(url="http://localhost:6333")
        >>> config.collection_name
        'multimodal_embeddings'
    """

    url: str
    api_key: Optional[str] = None
    collection_name: str = "multimodal_embeddings"
    distance_metric: str = "cosine"
    enable_named_vectors: bool = True
