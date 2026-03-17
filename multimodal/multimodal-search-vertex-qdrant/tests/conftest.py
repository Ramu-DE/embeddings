"""Pytest configuration and fixtures."""

import pytest
from multimodal_search.models import VertexAIConfig, QdrantConfig


@pytest.fixture
def vertex_ai_config():
    """Fixture for Vertex AI configuration."""
    return VertexAIConfig(
        project_id="test-project",
        location="us-central1",
        model="gemini-embedding-2-preview",
    )


@pytest.fixture
def qdrant_config():
    """Fixture for Qdrant configuration."""
    return QdrantConfig(
        url="http://localhost:6333",
        collection_name="test_multimodal_embeddings",
        distance_metric="cosine",
        enable_named_vectors=True,
    )
