"""Unit tests for data models."""

import pytest
from datetime import datetime
from multimodal_search.models import (
    ContentItem,
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


class TestContentItem:
    """Tests for ContentItem model."""

    def test_text_content_item(self):
        """Test creating a text content item."""
        item = ContentItem(
            content_type="text",
            data="Hello world",
            source_id="text_001",
        )
        assert item.content_type == "text"
        assert item.data == "Hello world"
        assert item.source_id == "text_001"
        assert item.mime_type is None
        assert item.metadata is None

    def test_image_content_item(self):
        """Test creating an image content item."""
        image_data = b"\x89PNG\r\n\x1a\n"
        item = ContentItem(
            content_type="image",
            data=image_data,
            mime_type="image/png",
            source_id="img_001",
            metadata={"width": 800, "height": 600},
        )
        assert item.content_type == "image"
        assert item.data == image_data
        assert item.mime_type == "image/png"
        assert item.metadata["width"] == 800

    def test_content_item_with_optional_fields(self):
        """Test content item with all optional fields."""
        item = ContentItem(
            content_type="video",
            data=b"video_data",
            mime_type="video/mp4",
            source_id="vid_001",
            metadata={"duration": 30.5, "resolution": "1080p"},
        )
        assert item.source_id == "vid_001"
        assert item.metadata["duration"] == 30.5


class TestEmbeddingMetadata:
    """Tests for EmbeddingMetadata model."""

    def test_basic_metadata(self):
        """Test creating basic embedding metadata."""
        timestamp = datetime.now()
        metadata = EmbeddingMetadata(
            content_type="text",
            source_id="doc_001",
            timestamp=timestamp,
            dimension=756,
            model_version="gemini-embedding-2-preview",
        )
        assert metadata.content_type == "text"
        assert metadata.source_id == "doc_001"
        assert metadata.timestamp == timestamp
        assert metadata.dimension == 756
        assert metadata.model_version == "gemini-embedding-2-preview"

    def test_metadata_with_language(self):
        """Test metadata with language field."""
        metadata = EmbeddingMetadata(
            content_type="text",
            source_id="doc_002",
            timestamp=datetime.now(),
            dimension=1024,
            model_version="gemini-embedding-2-preview",
            language="en",
        )
        assert metadata.language == "en"

    def test_metadata_with_duration(self):
        """Test metadata with duration for audio/video."""
        metadata = EmbeddingMetadata(
            content_type="audio",
            source_id="audio_001",
            timestamp=datetime.now(),
            dimension=512,
            model_version="gemini-embedding-2-preview",
            duration=45.5,
        )
        assert metadata.duration == 45.5

    def test_metadata_with_page_count(self):
        """Test metadata with page count for PDFs."""
        metadata = EmbeddingMetadata(
            content_type="pdf",
            source_id="pdf_001",
            timestamp=datetime.now(),
            dimension=756,
            model_version="gemini-embedding-2-preview",
            page_count=5,
        )
        assert metadata.page_count == 5

    def test_metadata_with_custom_fields(self):
        """Test metadata with custom fields."""
        metadata = EmbeddingMetadata(
            content_type="image",
            source_id="img_001",
            timestamp=datetime.now(),
            dimension=1024,
            model_version="gemini-embedding-2-preview",
            custom_metadata={"tags": ["sunset", "ocean"], "author": "user_123"},
        )
        assert metadata.custom_metadata["tags"] == ["sunset", "ocean"]
        assert metadata.custom_metadata["author"] == "user_123"


class TestEmbeddingResult:
    """Tests for EmbeddingResult model."""

    def test_basic_embedding_result(self):
        """Test creating basic embedding result."""
        vector = [0.1, 0.2, 0.3, 0.4]
        result = EmbeddingResult(
            vector=vector,
            dimension=4,
            content_type="text",
            model_version="gemini-embedding-2-preview",
        )
        assert result.vector == vector
        assert result.dimension == 4
        assert result.content_type == "text"
        assert result.metadata is None

    def test_embedding_result_with_metadata(self):
        """Test embedding result with metadata."""
        result = EmbeddingResult(
            vector=[0.5] * 756,
            dimension=756,
            content_type="image",
            model_version="gemini-embedding-2-preview",
            metadata={"processing_time": 0.5},
        )
        assert len(result.vector) == 756
        assert result.metadata["processing_time"] == 0.5


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_basic_search_result(self):
        """Test creating basic search result."""
        timestamp = datetime.now()
        metadata = EmbeddingMetadata(
            content_type="text",
            source_id="doc_001",
            timestamp=timestamp,
            dimension=756,
            model_version="gemini-embedding-2-preview",
        )
        result = SearchResult(
            point_id="point_123",
            score=0.95,
            content_type="text",
            source_id="doc_001",
            timestamp=timestamp,
            metadata=metadata,
        )
        assert result.point_id == "point_123"
        assert result.score == 0.95
        assert result.content_type == "text"
        assert result.vector is None

    def test_search_result_with_vector(self):
        """Test search result with vector included."""
        timestamp = datetime.now()
        metadata = EmbeddingMetadata(
            content_type="image",
            source_id="img_001",
            timestamp=timestamp,
            dimension=512,
            model_version="gemini-embedding-2-preview",
        )
        vector = [0.1] * 512
        result = SearchResult(
            point_id="point_456",
            score=0.88,
            content_type="image",
            source_id="img_001",
            timestamp=timestamp,
            metadata=metadata,
            vector=vector,
        )
        assert result.vector == vector
        assert len(result.vector) == 512


class TestSearchResponse:
    """Tests for SearchResponse model."""

    def test_basic_search_response(self):
        """Test creating basic search response."""
        response = SearchResponse(
            results=[],
            query_metadata={"query_type": "text"},
            total_results=0,
            search_time_ms=15.5,
        )
        assert len(response.results) == 0
        assert response.total_results == 0
        assert response.search_time_ms == 15.5
        assert response.two_stage is False

    def test_search_response_with_results(self):
        """Test search response with results."""
        timestamp = datetime.now()
        metadata = EmbeddingMetadata(
            content_type="text",
            source_id="doc_001",
            timestamp=timestamp,
            dimension=756,
            model_version="gemini-embedding-2-preview",
        )
        result = SearchResult(
            point_id="point_123",
            score=0.95,
            content_type="text",
            source_id="doc_001",
            timestamp=timestamp,
            metadata=metadata,
        )
        response = SearchResponse(
            results=[result],
            query_metadata={"query_type": "text", "dimension": 756},
            total_results=1,
            search_time_ms=25.3,
        )
        assert len(response.results) == 1
        assert response.results[0].score == 0.95

    def test_two_stage_search_response(self):
        """Test two-stage search response."""
        response = SearchResponse(
            results=[],
            query_metadata={
                "first_stage_dim": 256,
                "second_stage_dim": 1024,
                "candidates_retrieved": 100,
            },
            total_results=10,
            search_time_ms=45.8,
            two_stage=True,
        )
        assert response.two_stage is True
        assert response.query_metadata["first_stage_dim"] == 256


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_valid_result(self):
        """Test creating valid validation result."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.error_type is None
        assert result.error_message is None
        assert len(result.warnings) == 0

    def test_invalid_result_with_error(self):
        """Test creating invalid validation result."""
        result = ValidationResult(
            valid=False,
            error_type="INVALID_FORMAT",
            error_message="Unsupported image format",
        )
        assert result.valid is False
        assert result.error_type == "INVALID_FORMAT"
        assert result.error_message == "Unsupported image format"

    def test_valid_result_with_warnings(self):
        """Test validation result with warnings."""
        result = ValidationResult(
            valid=True,
            warnings=["File size is large", "Consider compression"],
        )
        assert result.valid is True
        assert len(result.warnings) == 2
        assert "File size is large" in result.warnings


class TestSearchFilters:
    """Tests for SearchFilters model."""

    def test_empty_filters(self):
        """Test creating empty filters."""
        filters = SearchFilters()
        assert filters.content_types is None
        assert filters.source_ids is None
        assert filters.timestamp_from is None
        assert filters.timestamp_to is None
        assert filters.languages is None
        assert filters.custom_filters is None

    def test_content_type_filter(self):
        """Test filter by content types."""
        filters = SearchFilters(content_types=["text", "image"])
        assert filters.content_types == ["text", "image"]

    def test_timestamp_range_filter(self):
        """Test filter by timestamp range."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        filters = SearchFilters(timestamp_from=start, timestamp_to=end)
        assert filters.timestamp_from == start
        assert filters.timestamp_to == end

    def test_multiple_filters(self):
        """Test combining multiple filters."""
        filters = SearchFilters(
            content_types=["video", "audio"],
            languages=["en", "es"],
            custom_filters={"category": "education"},
        )
        assert filters.content_types == ["video", "audio"]
        assert filters.languages == ["en", "es"]
        assert filters.custom_filters["category"] == "education"


class TestStageConfig:
    """Tests for StageConfig model."""

    def test_stage_config(self):
        """Test creating stage configuration."""
        config = StageConfig(dimension=256, limit=100)
        assert config.dimension == 256
        assert config.limit == 100


class TestVertexAIConfig:
    """Tests for VertexAIConfig model."""

    def test_basic_vertex_config(self):
        """Test creating basic Vertex AI config."""
        config = VertexAIConfig(project_id="my-project")
        assert config.project_id == "my-project"
        assert config.location in ("us-central1", "global")  # default varies by deployment
        assert config.model == "gemini-embedding-2-preview"
        assert config.credentials_path is None

    def test_vertex_config_with_custom_values(self):
        """Test Vertex AI config with custom values."""
        config = VertexAIConfig(
            project_id="my-project",
            location="europe-west1",
            credentials_path="/path/to/creds.json",
            model="custom-model",
        )
        assert config.location == "europe-west1"
        assert config.credentials_path == "/path/to/creds.json"
        assert config.model == "custom-model"


class TestQdrantConfig:
    """Tests for QdrantConfig model."""

    def test_basic_qdrant_config(self):
        """Test creating basic Qdrant config."""
        config = QdrantConfig(url="http://localhost:6333")
        assert config.url == "http://localhost:6333"
        assert config.api_key is None
        assert config.collection_name == "multimodal_embeddings"
        assert config.distance_metric == "cosine"
        assert config.enable_named_vectors is True

    def test_qdrant_config_with_custom_values(self):
        """Test Qdrant config with custom values."""
        config = QdrantConfig(
            url="https://qdrant.example.com",
            api_key="secret_key",
            collection_name="custom_collection",
            distance_metric="euclidean",
            enable_named_vectors=False,
        )
        assert config.url == "https://qdrant.example.com"
        assert config.api_key == "secret_key"
        assert config.collection_name == "custom_collection"
        assert config.distance_metric == "euclidean"
        assert config.enable_named_vectors is False
