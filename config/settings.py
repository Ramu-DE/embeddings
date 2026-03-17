"""Configuration settings for Vertex AI and Qdrant.

This module defines Pydantic settings models for all configurable
parameters and a :func:`get_settings` factory that populates them from
environment variables.  It is intended as an alternative to the
dataclass-based config objects in
:mod:`multimodal_search.models` when Pydantic validation is preferred.

Environment variables recognised by :func:`get_settings`:

Vertex AI:
    ``VERTEX_AI_PROJECT_ID``, ``VERTEX_AI_LOCATION``,
    ``VERTEX_AI_MODEL``, ``VERTEX_AI_CREDENTIALS_PATH``

Qdrant:
    ``QDRANT_URL``, ``QDRANT_API_KEY``, ``QDRANT_COLLECTION_NAME``,
    ``QDRANT_DISTANCE_METRIC``, ``QDRANT_ENABLE_NAMED_VECTORS``

Embedding:
    ``EMBEDDING_DEFAULT_DIMENSION``, ``EMBEDDING_TWO_STAGE_ENABLED``,
    ``EMBEDDING_FIRST_STAGE_DIMENSION``,
    ``EMBEDDING_SECOND_STAGE_DIMENSION``
"""

import os
from typing import Optional
from pydantic import BaseModel, Field


class VertexAISettings(BaseModel):
    """Pydantic settings model for Vertex AI configuration.

    Attributes:
        project_id: GCP project ID that has the Vertex AI API enabled.
            Required — no default value.
        location: GCP region where the embedding model is deployed.
            Defaults to ``"us-central1"``.
        model: Embedding model name.  Defaults to
            ``"gemini-embedding-2-preview"``.
        credentials_path: Optional filesystem path to a service account
            JSON key file.  When ``None``, Application Default
            Credentials are used.

    Example:
        >>> settings = VertexAISettings(project_id="my-gcp-project")
        >>> settings.location
        'us-central1'
    """

    project_id: str = Field(..., description="GCP project ID")
    location: str = Field(default="us-central1", description="GCP location")
    model: str = Field(
        default="gemini-embedding-2-preview", description="Embedding model name"
    )
    credentials_path: Optional[str] = Field(
        default=None, description="Path to service account credentials JSON"
    )


class QdrantSettings(BaseModel):
    """Pydantic settings model for Qdrant configuration.

    Attributes:
        url: Base URL of the Qdrant instance (e.g.
            ``"http://localhost:6333"``).  Required — no default value.
        api_key: API key for authenticated Qdrant Cloud instances.
            ``None`` for unauthenticated local instances.
        collection_name: Name of the Qdrant collection.  Defaults to
            ``"multimodal_embeddings"``.
        distance_metric: Vector distance metric — ``"cosine"``,
            ``"dot"``, or ``"euclid"``.  Defaults to ``"cosine"``.
        enable_named_vectors: When ``True``, the collection supports
            named vectors for two-stage retrieval.

    Example:
        >>> settings = QdrantSettings(url="http://localhost:6333")
        >>> settings.collection_name
        'multimodal_embeddings'
    """

    url: str = Field(..., description="Qdrant server URL")
    api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    collection_name: str = Field(
        default="multimodal_embeddings", description="Collection name"
    )
    distance_metric: str = Field(default="cosine", description="Distance metric")
    enable_named_vectors: bool = Field(
        default=True, description="Enable named vectors for two-stage retrieval"
    )


class EmbeddingSettings(BaseModel):
    """Pydantic settings model for embedding behaviour.

    Attributes:
        default_dimension: Default Matryoshka dimension used when no
            explicit dimension is specified.  Defaults to ``756``.
        valid_dimensions: Complete list of supported Matryoshka
            dimensions.
        two_stage_enabled: When ``True``, two-stage retrieval is
            available.
        first_stage_dimension: Dimension used for the fast first stage
            of two-stage retrieval.  Defaults to ``256``.
        second_stage_dimension: Dimension used for the accurate second
            stage of two-stage retrieval.  Defaults to ``1024``.

    Example:
        >>> settings = EmbeddingSettings()
        >>> settings.default_dimension
        756
        >>> 256 in settings.valid_dimensions
        True
    """

    default_dimension: int = Field(default=756, description="Default embedding dimension")
    valid_dimensions: list[int] = Field(
        default=[128, 256, 512, 756, 1024, 1536, 2048, 3072],
        description="Valid Matryoshka dimensions",
    )
    two_stage_enabled: bool = Field(
        default=True, description="Enable two-stage retrieval"
    )
    first_stage_dimension: int = Field(
        default=256, description="First stage dimension for two-stage retrieval"
    )
    second_stage_dimension: int = Field(
        default=1024, description="Second stage dimension for two-stage retrieval"
    )


class Settings(BaseModel):
    """Root application settings aggregating all sub-settings.

    Attributes:
        vertex_ai: Vertex AI connection and model settings.
        qdrant: Qdrant connection and collection settings.
        embedding: Embedding dimension and two-stage retrieval settings.

    Example:
        >>> settings = get_settings()
        >>> settings.vertex_ai.location
        'us-central1'
        >>> settings.embedding.default_dimension
        756
    """

    vertex_ai: VertexAISettings
    qdrant: QdrantSettings
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)


def get_settings() -> Settings:
    """Load application settings from environment variables.

    Reads all recognised environment variables (see module docstring for
    the full list) and returns a fully populated :class:`Settings`
    instance.  Missing optional variables fall back to their defaults.

    Returns:
        A :class:`Settings` instance populated from the environment.

    Raises:
        pydantic.ValidationError: If a required environment variable
            (``VERTEX_AI_PROJECT_ID`` or ``QDRANT_URL``) is missing or
            if a value fails Pydantic type coercion.

    Example:
        >>> import os
        >>> os.environ["VERTEX_AI_PROJECT_ID"] = "my-project"
        >>> os.environ["QDRANT_URL"] = "http://localhost:6333"
        >>> settings = get_settings()
        >>> settings.vertex_ai.project_id
        'my-project'
    """
    vertex_ai = VertexAISettings(
        project_id=os.getenv("VERTEX_AI_PROJECT_ID", ""),
        location=os.getenv("VERTEX_AI_LOCATION", "us-central1"),
        model=os.getenv("VERTEX_AI_MODEL", "gemini-embedding-2-preview"),
        credentials_path=os.getenv("VERTEX_AI_CREDENTIALS_PATH"),
    )

    qdrant = QdrantSettings(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=os.getenv("QDRANT_COLLECTION_NAME", "multimodal_embeddings"),
        distance_metric=os.getenv("QDRANT_DISTANCE_METRIC", "cosine"),
        enable_named_vectors=os.getenv("QDRANT_ENABLE_NAMED_VECTORS", "true").lower()
        == "true",
    )

    embedding = EmbeddingSettings(
        default_dimension=int(os.getenv("EMBEDDING_DEFAULT_DIMENSION", "756")),
        two_stage_enabled=os.getenv("EMBEDDING_TWO_STAGE_ENABLED", "true").lower()
        == "true",
        first_stage_dimension=int(os.getenv("EMBEDDING_FIRST_STAGE_DIMENSION", "256")),
        second_stage_dimension=int(os.getenv("EMBEDDING_SECOND_STAGE_DIMENSION", "1024")),
    )

    return Settings(vertex_ai=vertex_ai, qdrant=qdrant, embedding=embedding)
