"""Multimodal Search System using Vertex AI and Qdrant."""

from multimodal_search.content_processor import ContentProcessor
from multimodal_search.embedding_service import EmbeddingService
from multimodal_search.vector_store import VectorStore
from multimodal_search.search_engine import SearchEngine
from multimodal_search.api import (
    MultimodalSearchAPI,
    EmbeddingResponse,
    BatchEmbeddingResponse,
    SystemStatus,
)
from multimodal_search.models import (
    ContentItem,
    InterleavedPart,
    EmbeddingMetadata,
    EmbeddingResult,
    SearchResult,
    SearchResponse,
    ValidationResult,
    SearchFilters,
    StageConfig,
    VertexAIConfig,
    QdrantConfig,
)
from multimodal_search.exceptions import (
    ValidationError,
    EmbeddingError,
    StorageError,
    SearchError,
)

__version__ = "0.1.0"

__all__ = [
    # High-level API
    "MultimodalSearchAPI",
    "EmbeddingResponse",
    "BatchEmbeddingResponse",
    "SystemStatus",
    # Components
    "ContentProcessor",
    "EmbeddingService",
    "VectorStore",
    "SearchEngine",
    # Models
    "ContentItem",
    "InterleavedPart",
    "EmbeddingMetadata",
    "EmbeddingResult",
    "SearchResult",
    "SearchResponse",
    "ValidationResult",
    "SearchFilters",
    "StageConfig",
    "VertexAIConfig",
    "QdrantConfig",
    # Exceptions
    "ValidationError",
    "EmbeddingError",
    "StorageError",
    "SearchError",
]
