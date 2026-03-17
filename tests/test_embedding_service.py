"""Unit tests for EmbeddingService (google-genai SDK / gemini-embedding-2-preview)."""

import sys
import pytest
from unittest.mock import MagicMock, patch

# Pre-mock google.genai before any import that might trigger it
_mock_genai = MagicMock()
_mock_genai_types = MagicMock()
sys.modules.setdefault("google.genai", _mock_genai)
sys.modules.setdefault("google.genai.types", _mock_genai_types)

from google.api_core import exceptions as google_exceptions

from multimodal_search.embedding_service import EmbeddingService
from multimodal_search.models import ContentItem, EmbeddingResult
from multimodal_search.exceptions import EmbeddingError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_genai_client():
    """Mock google.genai.Client instance."""
    client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 756
    client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])
    return client


@pytest.fixture
def embedding_service(mock_genai_client):
    """Create EmbeddingService with mocked google.genai.Client."""
    with patch("google.genai.Client", return_value=mock_genai_client):
        svc = EmbeddingService(project_id="test-project", location="global")
    svc._client = mock_genai_client
    return svc


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestEmbeddingServiceInitialization:

    def test_successful_initialization(self, mock_genai_client):
        with patch("google.genai.Client", return_value=mock_genai_client) as mock_cls:
            svc = EmbeddingService(project_id="test-project", location="global")

        assert svc.project_id == "test-project"
        assert svc.location == "global"
        assert svc.MODEL_NAME == "gemini-embedding-2-preview"
        mock_cls.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="global",
        )

    def test_initialization_with_custom_location(self, mock_genai_client):
        with patch("google.genai.Client", return_value=mock_genai_client) as mock_cls:
            svc = EmbeddingService(project_id="test-project", location="us-central1")

        assert svc.location == "us-central1"
        mock_cls.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="us-central1",
        )

    def test_initialization_failure_raises_embedding_error(self):
        with patch("google.genai.Client", side_effect=Exception("Auth failed")):
            with pytest.raises(EmbeddingError) as exc_info:
                EmbeddingService(project_id="test-project")

        assert exc_info.value.error_type == "AUTH_FAILED"
        assert "Failed to initialise google-genai client" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Dimension validation
# ---------------------------------------------------------------------------

class TestDimensionValidation:

    def test_valid_dimensions(self, embedding_service):
        for dim in [128, 256, 512, 756, 1024, 1536, 2048, 3072]:
            embedding_service.validate_dimension(dim)  # should not raise

    def test_invalid_dimension_raises(self, embedding_service):
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.validate_dimension(999)
        assert exc_info.value.error_type == "INVALID_DIMENSION"
        assert "999" in str(exc_info.value)
        assert "Valid dimensions" in str(exc_info.value)

    def test_zero_dimension_raises(self, embedding_service):
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.validate_dimension(0)
        assert exc_info.value.error_type == "INVALID_DIMENSION"

    def test_negative_dimension_raises(self, embedding_service):
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.validate_dimension(-100)
        assert exc_info.value.error_type == "INVALID_DIMENSION"


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

class TestEmbedText:

    def test_embed_text_success(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3] * 252  # 756 values
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        result = embedding_service.embed_text("Hello world", dimension=756)

        assert isinstance(result, EmbeddingResult)
        assert len(result.vector) == 756
        assert result.dimension == 756
        assert result.content_type == "text"
        assert result.model_version == "gemini-embedding-2-preview"

        call_kwargs = mock_genai_client.models.embed_content.call_args[1]
        assert call_kwargs["model"] == "gemini-embedding-2-preview"
        assert call_kwargs["contents"] == "Hello world"
        # config is a MagicMock (google.genai.types is mocked); just verify it was passed
        assert "config" in call_kwargs

    def test_embed_text_various_dimensions(self, embedding_service, mock_genai_client):
        for dim in [128, 256, 512, 1024]:
            mock_embedding = MagicMock()
            mock_embedding.values = [0.5] * dim
            mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

            result = embedding_service.embed_text("Test", dimension=dim)
            assert result.dimension == dim
            assert len(result.vector) == dim

    def test_embed_text_default_dimension(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        result = embedding_service.embed_text("Test")
        assert result.dimension == 756

    def test_embed_text_invalid_dimension(self, embedding_service):
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_text("Test", dimension=999)
        assert exc_info.value.error_type == "INVALID_DIMENSION"

    def test_embed_text_empty_string(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.0] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        result = embedding_service.embed_text("", dimension=756)
        assert isinstance(result, EmbeddingResult)
        assert result.content_type == "text"


# ---------------------------------------------------------------------------
# embed_image
# ---------------------------------------------------------------------------

class TestEmbedImage:

    def test_embed_image_success(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.2] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        result = embedding_service.embed_image(b"\x89PNG\r\n\x1a\n", dimension=756)

        assert isinstance(result, EmbeddingResult)
        assert len(result.vector) == 756
        assert result.content_type == "image"
        assert result.model_version == "gemini-embedding-2-preview"

    def test_embed_image_various_dimensions(self, embedding_service, mock_genai_client):
        for dim in [256, 512, 1024, 2048]:
            mock_embedding = MagicMock()
            mock_embedding.values = [0.1] * dim
            mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

            result = embedding_service.embed_image(b"fake_image", dimension=dim)
            assert result.dimension == dim
            assert result.content_type == "image"

    def test_embed_image_default_dimension(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.5] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        result = embedding_service.embed_image(b"image_data")
        assert result.dimension == 756

    def test_embed_image_invalid_dimension(self, embedding_service):
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_image(b"image_data", dimension=500)
        assert exc_info.value.error_type == "INVALID_DIMENSION"


# ---------------------------------------------------------------------------
# embed_video
# ---------------------------------------------------------------------------

class TestEmbedVideo:

    def test_embed_video_success(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.3] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        result = embedding_service.embed_video(b"fake_video_data", dimension=756)

        assert isinstance(result, EmbeddingResult)
        assert len(result.vector) == 756
        assert result.content_type == "video"
        assert result.model_version == "gemini-embedding-2-preview"

    def test_embed_video_various_dimensions(self, embedding_service, mock_genai_client):
        for dim in [128, 512, 1536]:
            mock_embedding = MagicMock()
            mock_embedding.values = [0.2] * dim
            mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

            result = embedding_service.embed_video(b"video_content", dimension=dim)
            assert result.dimension == dim
            assert result.content_type == "video"


# ---------------------------------------------------------------------------
# embed_audio
# ---------------------------------------------------------------------------

class TestEmbedAudio:

    def test_embed_audio_success(self, embedding_service, mock_genai_client):
        """Audio embedding should succeed via the new SDK."""
        mock_embedding = MagicMock()
        mock_embedding.values = [0.4] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        result = embedding_service.embed_audio(b"fake_audio_data", dimension=756)

        assert isinstance(result, EmbeddingResult)
        assert result.content_type == "audio"
        assert result.dimension == 756
        assert result.model_version == "gemini-embedding-2-preview"

    def test_embed_audio_invalid_dimension(self, embedding_service):
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_audio(b"audio", dimension=999)
        assert exc_info.value.error_type == "INVALID_DIMENSION"


# ---------------------------------------------------------------------------
# embed_pdf
# ---------------------------------------------------------------------------

class TestEmbedPDF:

    def test_embed_pdf_success(self, embedding_service, mock_genai_client):
        """PDF embedding should succeed via the new SDK."""
        mock_embedding = MagicMock()
        mock_embedding.values = [0.5] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        result = embedding_service.embed_pdf(b"%PDF-1.4", dimension=756)

        assert isinstance(result, EmbeddingResult)
        assert result.content_type == "pdf"
        assert result.dimension == 756
        assert result.model_version == "gemini-embedding-2-preview"

    def test_embed_pdf_invalid_dimension(self, embedding_service):
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_pdf(b"%PDF-1.4", dimension=999)
        assert exc_info.value.error_type == "INVALID_DIMENSION"


# ---------------------------------------------------------------------------
# embed_batch
# ---------------------------------------------------------------------------

class TestBatchEmbedding:

    def test_embed_batch_empty_list(self, embedding_service):
        assert embedding_service.embed_batch([], dimension=756) == []

    def test_embed_batch_single_text(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        results = embedding_service.embed_batch(
            [ContentItem(content_type="text", data="Hello")], dimension=756
        )
        assert len(results) == 1
        assert results[0].content_type == "text"
        assert results[0].dimension == 756

    def test_embed_batch_multiple_texts(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.2] * 512
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        items = [
            ContentItem(content_type="text", data="First"),
            ContentItem(content_type="text", data="Second"),
            ContentItem(content_type="text", data="Third"),
        ]
        results = embedding_service.embed_batch(items, dimension=512)

        assert len(results) == 3
        for r in results:
            assert r.content_type == "text"
            assert r.dimension == 512

    def test_embed_batch_mixed_modalities(self, embedding_service, mock_genai_client):
        """Mixed text/image/video batch should return correct content_types."""
        def _side_effect(**kwargs):
            mock_emb = MagicMock()
            mock_emb.values = [0.1] * 756
            return MagicMock(embeddings=[mock_emb])

        mock_genai_client.models.embed_content.side_effect = _side_effect

        items = [
            ContentItem(content_type="text", data="Text content"),
            ContentItem(content_type="image", data=b"image_data"),
            ContentItem(content_type="video", data=b"video_data"),
        ]
        results = embedding_service.embed_batch(items, dimension=756)

        assert len(results) == 3
        assert results[0].content_type == "text"
        assert results[1].content_type == "image"
        assert results[2].content_type == "video"

    def test_embed_batch_unknown_content_type(self, embedding_service):
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_batch(
                [ContentItem(content_type="unknown", data="data")], dimension=756
            )
        assert "Unknown content type" in str(exc_info.value)
        assert exc_info.value.error_type == "API_ERROR"

    def test_embed_batch_invalid_dimension(self, embedding_service):
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_batch(
                [ContentItem(content_type="text", data="Test")], dimension=999
            )
        assert exc_info.value.error_type == "INVALID_DIMENSION"

    def test_embed_batch_error_includes_item_index(self, embedding_service, mock_genai_client):
        """Failure on item 1 should mention index 1 in the error message."""
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        items = [
            ContentItem(content_type="text", data="Valid text"),
            ContentItem(content_type="unknown", data="bad"),
        ]
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_batch(items, dimension=756)
        assert "item 1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# embed_with_multiple_dimensions
# ---------------------------------------------------------------------------

class TestMultipleDimensions:

    def test_multiple_dimensions_text(self, embedding_service, mock_genai_client):
        # google.genai.types is mocked, so we can't read output_dimensionality back.
        # Instead return a fixed-size vector; the dimension on EmbeddingResult comes
        # from the dimension argument passed to embed_text, not the vector length.
        call_count = [0]
        dims = [256, 512, 1024]

        def _side_effect(**kwargs):
            dim = dims[call_count[0] % len(dims)]
            call_count[0] += 1
            mock_emb = MagicMock()
            mock_emb.values = [0.1] * dim
            return MagicMock(embeddings=[mock_emb])

        mock_genai_client.models.embed_content.side_effect = _side_effect

        content = ContentItem(content_type="text", data="Test text")
        results = embedding_service.embed_with_multiple_dimensions(content, dims)

        assert set(results.keys()) == {256, 512, 1024}
        # Verify each result has the correct declared dimension
        for dim in dims:
            assert results[dim].dimension == dim
            assert results[dim].content_type == "text"

    def test_multiple_dimensions_image(self, embedding_service, mock_genai_client):
        def _side_effect(**kwargs):
            dim = kwargs["config"].output_dimensionality
            mock_emb = MagicMock()
            mock_emb.values = [0.2] * dim
            return MagicMock(embeddings=[mock_emb])

        mock_genai_client.models.embed_content.side_effect = _side_effect

        content = ContentItem(content_type="image", data=b"image_data")
        results = embedding_service.embed_with_multiple_dimensions(content, [128, 756, 2048])

        for dim in [128, 756, 2048]:
            assert results[dim].dimension == dim
            assert results[dim].content_type == "image"

    def test_multiple_dimensions_invalid_raises(self, embedding_service):
        content = ContentItem(content_type="text", data="Test")
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_with_multiple_dimensions(content, [256, 999, 1024])
        assert exc_info.value.error_type == "INVALID_DIMENSION"

    def test_multiple_dimensions_unknown_type_raises(self, embedding_service):
        content = ContentItem(content_type="unknown", data="data")
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_with_multiple_dimensions(content, [256, 512])
        assert "Unknown content type" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Retry logic and error handling
# ---------------------------------------------------------------------------

class TestRetryLogic:

    def test_retry_on_rate_limit_succeeds(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 756

        mock_genai_client.models.embed_content.side_effect = [
            google_exceptions.ResourceExhausted("Rate limit"),
            google_exceptions.ResourceExhausted("Rate limit"),
            MagicMock(embeddings=[mock_embedding]),
        ]

        with patch("time.sleep"):
            result = embedding_service.embed_text("Test", dimension=756)

        assert isinstance(result, EmbeddingResult)
        assert mock_genai_client.models.embed_content.call_count == 3

    def test_retry_exhausted_on_rate_limit(self, embedding_service, mock_genai_client):
        mock_genai_client.models.embed_content.side_effect = google_exceptions.ResourceExhausted("Rate limit")

        with patch("time.sleep"):
            with pytest.raises(EmbeddingError) as exc_info:
                embedding_service.embed_text("Test", dimension=756)

        assert exc_info.value.error_type == "RATE_LIMIT"
        assert "after 3 retries" in str(exc_info.value)
        assert mock_genai_client.models.embed_content.call_count == 3

    def test_no_retry_on_auth_error(self, embedding_service, mock_genai_client):
        mock_genai_client.models.embed_content.side_effect = google_exceptions.Unauthenticated("Auth failed")

        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_text("Test", dimension=756)

        assert exc_info.value.error_type == "AUTH_FAILED"
        assert mock_genai_client.models.embed_content.call_count == 1

    def test_no_retry_on_permission_denied(self, embedding_service, mock_genai_client):
        mock_genai_client.models.embed_content.side_effect = google_exceptions.PermissionDenied("No permission")

        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_text("Test", dimension=756)

        assert exc_info.value.error_type == "AUTH_FAILED"
        assert mock_genai_client.models.embed_content.call_count == 1

    def test_retry_on_timeout_succeeds(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 756

        mock_genai_client.models.embed_content.side_effect = [
            google_exceptions.DeadlineExceeded("Timeout"),
            MagicMock(embeddings=[mock_embedding]),
        ]

        with patch("time.sleep"):
            result = embedding_service.embed_text("Test", dimension=756)

        assert isinstance(result, EmbeddingResult)
        assert mock_genai_client.models.embed_content.call_count == 2

    def test_retry_exhausted_on_timeout(self, embedding_service, mock_genai_client):
        mock_genai_client.models.embed_content.side_effect = google_exceptions.DeadlineExceeded("Timeout")

        with patch("time.sleep"):
            with pytest.raises(EmbeddingError) as exc_info:
                embedding_service.embed_text("Test", dimension=756)

        assert exc_info.value.error_type == "NETWORK_ERROR"
        assert "timeout" in str(exc_info.value).lower()
        assert mock_genai_client.models.embed_content.call_count == 3

    def test_retry_on_generic_api_error_succeeds(self, embedding_service, mock_genai_client):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 756

        mock_genai_client.models.embed_content.side_effect = [
            google_exceptions.GoogleAPIError("API error"),
            MagicMock(embeddings=[mock_embedding]),
        ]

        with patch("time.sleep"):
            result = embedding_service.embed_text("Test", dimension=756)

        assert isinstance(result, EmbeddingResult)

    def test_unexpected_error_raises_api_error(self, embedding_service, mock_genai_client):
        mock_genai_client.models.embed_content.side_effect = ValueError("Unexpected error")

        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_text("Test", dimension=756)

        assert exc_info.value.error_type == "API_ERROR"
        assert "Unexpected error" in str(exc_info.value)

    def test_exponential_backoff_delays(self, embedding_service, mock_genai_client):
        mock_genai_client.models.embed_content.side_effect = google_exceptions.ResourceExhausted("Rate limit")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(EmbeddingError):
                embedding_service.embed_text("Test", dimension=756)

        assert mock_sleep.call_count == 2
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays[0] == 1.0   # BASE_RETRY_DELAY * 2^0
        assert delays[1] == 2.0   # BASE_RETRY_DELAY * 2^1
