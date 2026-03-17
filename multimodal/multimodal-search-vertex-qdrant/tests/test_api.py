"""Tests for the high-level MultimodalSearchAPI."""

import os
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from multimodal_search.api import (
    MultimodalSearchAPI,
    BatchEmbeddingResponse,
    EmbeddingResponse,
    SystemStatus,
    _qdrant_config_from_env,
    _validate_qdrant_config,
    _validate_vertex_ai_config,
    _vertex_ai_config_from_env,
)
from multimodal_search.exceptions import (
    EmbeddingError,
    SearchError,
    StorageError,
    ValidationError,
)
from multimodal_search.models import (
    ContentItem,
    EmbeddingMetadata,
    EmbeddingResult,
    QdrantConfig,
    SearchFilters,
    SearchResponse,
    SearchResult,
    StageConfig,
    VertexAIConfig,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_embedding_result(dim: int = 756, content_type: str = "text") -> EmbeddingResult:
    return EmbeddingResult(
        vector=[0.1] * dim,
        dimension=dim,
        content_type=content_type,
        model_version="gemini-embedding-2-preview",
    )


def _make_search_result(score: float = 0.9, content_type: str = "text") -> SearchResult:
    meta = EmbeddingMetadata(
        content_type=content_type,
        source_id="src-1",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        dimension=756,
        model_version="gemini-embedding-2-preview",
    )
    return SearchResult(
        point_id="pt-1",
        score=score,
        content_type=content_type,
        source_id="src-1",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        metadata=meta,
    )


def _make_search_response(results=None) -> SearchResponse:
    if results is None:
        results = [_make_search_result()]
    return SearchResponse(
        results=results,
        query_metadata={"query_type": "text", "dimension": 756},
        total_results=len(results),
        search_time_ms=5.0,
        two_stage=False,
    )


@pytest.fixture
def mock_embedding_service():
    svc = MagicMock()
    svc.project_id = "test-project"
    svc.embed_text.return_value = _make_embedding_result()
    svc.embed_image.return_value = _make_embedding_result(content_type="image")
    svc.embed_batch.return_value = [_make_embedding_result()]
    return svc


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.store_embedding.return_value = "point-id-123"
    vs.store_embedding_with_named_vectors.return_value = "point-id-named"
    vs.search.return_value = [_make_search_result()]
    vs.initialize_collection.return_value = None
    return vs


@pytest.fixture
def mock_search_engine():
    se = MagicMock()
    se.search.return_value = _make_search_response()
    se.search_two_stage.return_value = _make_search_response()
    return se


@pytest.fixture
def api(mock_embedding_service, mock_vector_store, mock_search_engine):
    from multimodal_search.content_processor import ContentProcessor

    return MultimodalSearchAPI(
        content_processor=ContentProcessor(),
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        search_engine=mock_search_engine,
        default_dimension=756,
        enable_two_stage=True,
    )


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_vertex_ai_config_valid(self):
        cfg = VertexAIConfig(project_id="my-project")
        assert _validate_vertex_ai_config(cfg) == []

    def test_vertex_ai_config_missing_project(self):
        cfg = VertexAIConfig(project_id="")
        errors = _validate_vertex_ai_config(cfg)
        assert any("project_id" in e for e in errors)

    def test_qdrant_config_valid(self):
        cfg = QdrantConfig(url="http://localhost:6333")
        assert _validate_qdrant_config(cfg) == []

    def test_qdrant_config_missing_url(self):
        cfg = QdrantConfig(url="")
        errors = _validate_qdrant_config(cfg)
        assert any("url" in e for e in errors)

    def test_qdrant_config_invalid_metric(self):
        cfg = QdrantConfig(url="http://localhost:6333", distance_metric="invalid")
        errors = _validate_qdrant_config(cfg)
        assert any("distance_metric" in e for e in errors)

    def test_vertex_ai_config_from_env(self, monkeypatch):
        monkeypatch.setenv("VERTEX_AI_PROJECT_ID", "env-project")
        monkeypatch.setenv("VERTEX_AI_LOCATION", "europe-west1")
        cfg = _vertex_ai_config_from_env()
        assert cfg.project_id == "env-project"
        assert cfg.location == "europe-west1"

    def test_qdrant_config_from_env(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "secret")
        monkeypatch.setenv("QDRANT_COLLECTION_NAME", "my_collection")
        cfg = _qdrant_config_from_env()
        assert cfg.url == "http://qdrant:6333"
        assert cfg.api_key == "secret"
        assert cfg.collection_name == "my_collection"

    def test_from_config_raises_on_invalid_config(self):
        with pytest.raises(ValueError, match="Configuration errors"):
            MultimodalSearchAPI.from_config(
                vertex_ai_config=VertexAIConfig(project_id=""),
                qdrant_config=QdrantConfig(url=""),
            )


# ---------------------------------------------------------------------------
# initialize_system tests
# ---------------------------------------------------------------------------

class TestInitializeSystem:
    def test_initialize_system_success(self, api, mock_vector_store):
        status = api.initialize_system()
        assert isinstance(status, SystemStatus)
        assert status.vertex_ai_connected is True
        assert status.qdrant_connected is True
        assert status.collection_ready is True
        assert status.initialized is True
        mock_vector_store.initialize_collection.assert_called_once()

    def test_initialize_system_qdrant_failure(self, api, mock_vector_store):
        mock_vector_store.initialize_collection.side_effect = StorageError(
            "Connection refused", error_type="CONNECTION_FAILED"
        )
        status = api.initialize_system()
        assert status.initialized is False
        assert status.qdrant_connected is False
        assert len(status.errors) > 0

    def test_initialize_system_updates_dimension(self, api):
        status = api.initialize_system(default_dimension=1024)
        assert api.default_dimension == 1024
        assert status.default_dimension == 1024

    def test_initialize_system_updates_two_stage(self, api):
        api.initialize_system(enable_two_stage=False)
        assert api.enable_two_stage is False


# ---------------------------------------------------------------------------
# embed_content tests
# ---------------------------------------------------------------------------

class TestEmbedContent:
    def test_embed_text_no_store(self, api, mock_embedding_service):
        content = ContentItem(content_type="text", data="hello world")
        response = api.embed_content(content, store=False)
        assert isinstance(response, EmbeddingResponse)
        assert response.dimension == 756
        assert response.content_type == "text"
        assert response.point_id is None
        mock_embedding_service.embed_text.assert_called_once()

    def test_embed_text_with_store(self, api, mock_embedding_service, mock_vector_store):
        content = ContentItem(content_type="text", data="hello world", source_id="doc-1")
        response = api.embed_content(content, store=True)
        assert response.point_id == "point-id-123"
        mock_vector_store.store_embedding.assert_called_once()

    def test_embed_content_custom_dimension(self, api, mock_embedding_service):
        content = ContentItem(content_type="text", data="hello")
        api.embed_content(content, dimension=256, store=False)
        mock_embedding_service.embed_text.assert_called_with("hello", 256)

    def test_embed_content_named_vectors(self, api, mock_embedding_service, mock_vector_store):
        mock_embedding_service.embed_text.side_effect = [
            _make_embedding_result(756),
            _make_embedding_result(256),
        ]
        content = ContentItem(content_type="text", data="hello")
        response = api.embed_content(content, dimension=756, store=True, named_vectors=[256])
        assert response.point_id == "point-id-named"
        mock_vector_store.store_embedding_with_named_vectors.assert_called_once()

    def test_embed_content_validation_error(self, api):
        content = ContentItem(content_type="text", data="")
        with pytest.raises(ValidationError):
            api.embed_content(content)

    def test_embed_content_unknown_type(self, api):
        content = ContentItem(content_type="unknown", data="data")
        with pytest.raises(ValidationError, match="Unknown content type"):
            api.embed_content(content)

    def test_embed_content_embedding_error_propagates(self, api, mock_embedding_service):
        mock_embedding_service.embed_text.side_effect = EmbeddingError(
            "API down", error_type="API_ERROR"
        )
        content = ContentItem(content_type="text", data="hello")
        with pytest.raises(EmbeddingError):
            api.embed_content(content, store=False)

    def test_embed_content_storage_error_propagates(self, api, mock_vector_store):
        mock_vector_store.store_embedding.side_effect = StorageError(
            "Qdrant down", error_type="CONNECTION_FAILED"
        )
        content = ContentItem(content_type="text", data="hello")
        with pytest.raises(StorageError):
            api.embed_content(content, store=True)


# ---------------------------------------------------------------------------
# embed_batch tests
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    def test_embed_batch_empty(self, api):
        response = api.embed_batch([])
        assert response.total == 0
        assert response.stored == 0

    def test_embed_batch_single_item(self, api, mock_embedding_service, mock_vector_store):
        items = [ContentItem(content_type="text", data="hello")]
        response = api.embed_batch(items, store=True)
        assert response.total == 1
        assert response.stored == 1
        assert response.results[0].point_id == "point-id-123"

    def test_embed_batch_no_store(self, api, mock_embedding_service):
        items = [ContentItem(content_type="text", data="hello")]
        response = api.embed_batch(items, store=False)
        assert response.stored == 0
        assert response.results[0].point_id is None

    def test_embed_batch_validation_failure(self, api):
        items = [
            ContentItem(content_type="text", data="valid"),
            ContentItem(content_type="text", data=""),  # invalid
        ]
        with pytest.raises(ValidationError):
            api.embed_batch(items)

    def test_embed_batch_embedding_error_propagates(self, api, mock_embedding_service):
        mock_embedding_service.embed_batch.side_effect = EmbeddingError(
            "API error", error_type="API_ERROR"
        )
        items = [ContentItem(content_type="text", data="hello")]
        with pytest.raises(EmbeddingError):
            api.embed_batch(items, store=False)


# ---------------------------------------------------------------------------
# search tests
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_basic(self, api, mock_search_engine):
        query = ContentItem(content_type="text", data="find me something")
        response = api.search(query)
        assert isinstance(response, SearchResponse)
        assert response.total_results == 1
        mock_search_engine.search.assert_called_once()

    def test_search_with_filters(self, api, mock_search_engine):
        query = ContentItem(content_type="text", data="query")
        filters = SearchFilters(content_types=["image"])
        api.search(query, filters=filters)
        call_kwargs = mock_search_engine.search.call_args
        assert call_kwargs.kwargs.get("modality_filter") == ["image"] or \
               (call_kwargs.args and "image" in str(call_kwargs))

    def test_search_custom_dimension(self, api, mock_search_engine):
        query = ContentItem(content_type="text", data="query")
        api.search(query, dimension=512)
        call_kwargs = mock_search_engine.search.call_args
        assert call_kwargs.kwargs.get("dimension") == 512

    def test_search_validation_error(self, api):
        query = ContentItem(content_type="text", data="")
        with pytest.raises(ValidationError):
            api.search(query)

    def test_search_engine_error_propagates(self, api, mock_search_engine):
        mock_search_engine.search.side_effect = SearchError(
            "Search failed", error_type="SEARCH_ERROR"
        )
        query = ContentItem(content_type="text", data="query")
        with pytest.raises(SearchError):
            api.search(query)

    def test_search_score_threshold(self, api, mock_search_engine):
        query = ContentItem(content_type="text", data="query")
        api.search(query, score_threshold=0.8)
        call_kwargs = mock_search_engine.search.call_args
        assert call_kwargs.kwargs.get("score_threshold") == 0.8

    def test_search_extra_filters_applied(self, api, mock_search_engine):
        """Source ID filter should be applied as post-filter."""
        result_match = _make_search_result()
        result_match.source_id = "wanted"
        result_match.metadata.source_id = "wanted"
        result_no_match = _make_search_result()
        result_no_match.source_id = "other"
        result_no_match.metadata.source_id = "other"

        mock_search_engine.search.return_value = SearchResponse(
            results=[result_match, result_no_match],
            query_metadata={},
            total_results=2,
            search_time_ms=1.0,
            two_stage=False,
        )

        query = ContentItem(content_type="text", data="query")
        filters = SearchFilters(source_ids=["wanted"])
        response = api.search(query, filters=filters)
        assert response.total_results == 1
        assert response.results[0].source_id == "wanted"


# ---------------------------------------------------------------------------
# search_two_stage tests
# ---------------------------------------------------------------------------

class TestSearchTwoStage:
    def test_search_two_stage_basic(self, api, mock_search_engine):
        query = ContentItem(content_type="text", data="query")
        first = StageConfig(dimension=256, limit=100)
        second = StageConfig(dimension=1024, limit=10)
        response = api.search_two_stage(query, first, second)
        assert isinstance(response, SearchResponse)
        mock_search_engine.search_two_stage.assert_called_once_with(
            query=query,
            first_stage_dimension=256,
            second_stage_dimension=1024,
            first_stage_limit=100,
            final_limit=10,
            modality_filter=None,
        )

    def test_search_two_stage_with_filters(self, api, mock_search_engine):
        query = ContentItem(content_type="text", data="query")
        first = StageConfig(dimension=256, limit=50)
        second = StageConfig(dimension=1024, limit=5)
        filters = SearchFilters(content_types=["image", "video"])
        api.search_two_stage(query, first, second, filters=filters)
        call_kwargs = mock_search_engine.search_two_stage.call_args
        assert call_kwargs.kwargs.get("modality_filter") == ["image", "video"]

    def test_search_two_stage_validation_error(self, api):
        query = ContentItem(content_type="text", data="")
        with pytest.raises(ValidationError):
            api.search_two_stage(
                query,
                StageConfig(dimension=256, limit=100),
                StageConfig(dimension=1024, limit=10),
            )

    def test_search_two_stage_error_propagates(self, api, mock_search_engine):
        mock_search_engine.search_two_stage.side_effect = SearchError(
            "Two-stage failed", error_type="SEARCH_ERROR"
        )
        query = ContentItem(content_type="text", data="query")
        with pytest.raises(SearchError):
            api.search_two_stage(
                query,
                StageConfig(dimension=256, limit=100),
                StageConfig(dimension=1024, limit=10),
            )


# ---------------------------------------------------------------------------
# Error handling / formatting tests (req 12.1-12.5)
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_validation_error_has_error_type(self, api):
        content = ContentItem(content_type="text", data="")
        with pytest.raises(ValidationError) as exc_info:
            api.embed_content(content)
        assert exc_info.value.error_type is not None

    def test_embedding_error_has_error_type(self, api, mock_embedding_service):
        mock_embedding_service.embed_text.side_effect = EmbeddingError(
            "Auth failed", error_type="AUTH_FAILED"
        )
        content = ContentItem(content_type="text", data="hello")
        with pytest.raises(EmbeddingError) as exc_info:
            api.embed_content(content, store=False)
        assert exc_info.value.error_type == "AUTH_FAILED"

    def test_storage_error_has_error_type(self, api, mock_vector_store):
        mock_vector_store.store_embedding.side_effect = StorageError(
            "Connection failed", error_type="CONNECTION_FAILED"
        )
        content = ContentItem(content_type="text", data="hello")
        with pytest.raises(StorageError) as exc_info:
            api.embed_content(content, store=True)
        assert exc_info.value.error_type == "CONNECTION_FAILED"

    def test_search_error_has_error_type(self, api, mock_search_engine):
        mock_search_engine.search.side_effect = SearchError(
            "Timeout", error_type="TIMEOUT"
        )
        query = ContentItem(content_type="text", data="query")
        with pytest.raises(SearchError) as exc_info:
            api.search(query)
        assert exc_info.value.error_type == "TIMEOUT"

    def test_unexpected_exception_wrapped_as_search_error(self, api, mock_search_engine):
        mock_search_engine.search.side_effect = RuntimeError("unexpected")
        query = ContentItem(content_type="text", data="query")
        with pytest.raises(SearchError):
            api.search(query)
