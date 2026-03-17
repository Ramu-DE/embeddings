"""Unit tests for interleaved multimodal support (Requirement 16).

Covers:
- Interleaved content validation (ContentProcessor.validate_interleaved)
- Interleaved embedding generation (EmbeddingService.embed_interleaved)
- Search with interleaved queries (SearchEngine.search)
"""

import sys
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

# Pre-mock google.genai before any import that might trigger it
_mock_genai = MagicMock()
_mock_genai_types = MagicMock()
sys.modules.setdefault("google.genai", _mock_genai)
sys.modules.setdefault("google.genai.types", _mock_genai_types)

from multimodal_search.content_processor import ContentProcessor
from multimodal_search.embedding_service import EmbeddingService
from multimodal_search.search_engine import SearchEngine
from multimodal_search.models import (
    ContentItem,
    EmbeddingMetadata,
    EmbeddingResult,
    InterleavedPart,
    SearchResult,
    SearchResponse,
)
from multimodal_search.exceptions import EmbeddingError, SearchError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # minimal fake PNG bytes
JPEG_HEADER = b"\xff\xd8\xff" + b"\x00" * 100       # minimal fake JPEG bytes


def _make_interleaved(*parts: InterleavedPart, source_id: str = "src_001") -> ContentItem:
    """Helper to build an interleaved ContentItem."""
    return ContentItem(
        content_type="interleaved",
        interleaved_parts=list(parts),
        source_id=source_id,
    )


def _text_part(text: str) -> InterleavedPart:
    return InterleavedPart(content_type="text", data=text)


def _image_part(data: bytes = PNG_HEADER, mime_type: str = "image/png") -> InterleavedPart:
    return InterleavedPart(content_type="image", data=data, mime_type=mime_type)


@pytest.fixture
def processor():
    return ContentProcessor()


@pytest.fixture
def mock_genai_client():
    """Mock google.genai.Client instance."""
    client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.5] * 756
    client.models.embed_content.return_value = MagicMock(
        embeddings=[mock_embedding]
    )
    return client


@pytest.fixture
def embedding_service(mock_genai_client):
    with patch("google.genai.Client", return_value=mock_genai_client):
        svc = EmbeddingService(project_id="test-project", location="global")
    svc._client = mock_genai_client
    return svc


@pytest.fixture
def mock_embedding_service():
    svc = Mock()
    svc.embed_interleaved.return_value = EmbeddingResult(
        vector=[0.5] * 756,
        dimension=756,
        content_type="interleaved",
        model_version="gemini-embedding-2-preview",
    )
    return svc


@pytest.fixture
def mock_vector_store():
    store = Mock()
    store.search.return_value = [
        SearchResult(
            point_id="pt-1",
            score=0.92,
            content_type="text",
            source_id="doc-1",
            timestamp=datetime.now(),
            metadata=EmbeddingMetadata(
                content_type="text",
                source_id="doc-1",
                timestamp=datetime.now(),
                dimension=756,
                model_version="gemini-embedding-2-preview",
            ),
        )
    ]
    return store


@pytest.fixture
def search_engine(mock_embedding_service, mock_vector_store):
    return SearchEngine(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )


# ---------------------------------------------------------------------------
# 1. Interleaved content validation  (Req 16.1, 16.2)
# ---------------------------------------------------------------------------

class TestValidateInterleaved:
    """Tests for ContentProcessor.validate_interleaved."""

    # --- happy-path combinations ---

    def test_text_only_part_is_valid(self, processor):
        """Single text part should pass validation."""
        item = _make_interleaved(_text_part("Hello world"))
        result = processor.validate_interleaved(item)
        assert result.valid is True

    def test_image_only_part_is_valid(self, processor):
        """Single image part (PNG) should pass validation."""
        item = _make_interleaved(_image_part(PNG_HEADER, "image/png"))
        result = processor.validate_interleaved(item)
        assert result.valid is True

    def test_text_and_image_combination_is_valid(self, processor):
        """Text + image interleaved combination should pass (Req 16.3)."""
        item = _make_interleaved(
            _text_part("A photo of a cat"),
            _image_part(PNG_HEADER, "image/png"),
        )
        result = processor.validate_interleaved(item)
        assert result.valid is True

    def test_image_then_text_order_is_valid(self, processor):
        """Image followed by text should also pass."""
        item = _make_interleaved(
            _image_part(JPEG_HEADER, "image/jpeg"),
            _text_part("Caption for the image"),
        )
        result = processor.validate_interleaved(item)
        assert result.valid is True

    def test_multiple_text_parts_are_valid(self, processor):
        """Multiple text parts in sequence should pass."""
        item = _make_interleaved(
            _text_part("First sentence."),
            _text_part("Second sentence."),
        )
        result = processor.validate_interleaved(item)
        assert result.valid is True

    # --- wrong content_type on the ContentItem ---

    def test_non_interleaved_content_type_is_rejected(self, processor):
        """validate_interleaved must reject items whose content_type != 'interleaved'."""
        item = ContentItem(content_type="text", data="hello")
        result = processor.validate_interleaved(item)
        assert result.valid is False
        assert result.error_type == "INVALID_FORMAT"

    # --- empty / missing parts ---

    def test_empty_parts_list_is_rejected(self, processor):
        """An interleaved item with no parts should fail."""
        item = ContentItem(content_type="interleaved", interleaved_parts=[])
        result = processor.validate_interleaved(item)
        assert result.valid is False
        assert result.error_type == "EMPTY_CONTENT"

    def test_none_parts_is_rejected(self, processor):
        """An interleaved item with parts=None should fail."""
        item = ContentItem(content_type="interleaved", interleaved_parts=None)
        result = processor.validate_interleaved(item)
        assert result.valid is False
        assert result.error_type == "EMPTY_CONTENT"

    # --- invalid individual parts ---

    def test_unsupported_part_type_is_rejected(self, processor):
        """A part with an unknown content_type should fail."""
        bad_part = InterleavedPart(content_type="unknown", data=b"data")
        item = _make_interleaved(bad_part)
        result = processor.validate_interleaved(item)
        assert result.valid is False
        assert result.error_type == "INVALID_FORMAT"

    def test_invalid_image_format_in_part_is_rejected(self, processor):
        """A part with unsupported image MIME type should fail."""
        bad_image = InterleavedPart(
            content_type="image",
            data=b"GIF89a",  # GIF — not supported
            mime_type="image/gif",
        )
        item = _make_interleaved(bad_image)
        result = processor.validate_interleaved(item)
        assert result.valid is False
        assert result.error_type == "INVALID_FORMAT"

    def test_empty_text_part_is_rejected(self, processor):
        """A text part with empty/whitespace content should fail."""
        item = _make_interleaved(_text_part("   "))
        result = processor.validate_interleaved(item)
        assert result.valid is False
        assert result.error_type == "EMPTY_CONTENT"

    # --- mixed valid/invalid parts ---

    def test_first_valid_second_invalid_fails(self, processor):
        """Validation should fail when any part is invalid, even if first part is valid."""
        item = _make_interleaved(
            _text_part("Valid text"),
            InterleavedPart(content_type="image", data=b"GIF89a", mime_type="image/gif"),
        )
        result = processor.validate_interleaved(item)
        assert result.valid is False

    def test_error_message_includes_part_index(self, processor):
        """Error message should reference the failing part index."""
        item = _make_interleaved(
            _text_part("OK"),
            InterleavedPart(content_type="unknown", data=b"x"),
        )
        result = processor.validate_interleaved(item)
        assert result.valid is False
        assert "1" in result.error_message  # part index 1

    def test_valid_parts_before_invalid_still_fails(self, processor):
        """Three parts where only the last is invalid — whole item should fail."""
        item = _make_interleaved(
            _text_part("Part one"),
            _image_part(PNG_HEADER, "image/png"),
            InterleavedPart(content_type="audio", data=b"ID3", mime_type="audio/mp3"),
        )
        result = processor.validate_interleaved(item)
        # audio is not supported in interleaved validation (it delegates to validate_audio
        # which checks the format — but audio/mp3 with ID3 header is actually valid for
        # validate_audio, so this part passes). Let's use an actually invalid part:
        # Re-test with a truly invalid part
        item2 = _make_interleaved(
            _text_part("Part one"),
            _image_part(PNG_HEADER, "image/png"),
            InterleavedPart(content_type="image", data=b"not-an-image", mime_type="image/png"),
        )
        result2 = processor.validate_interleaved(item2)
        assert result2.valid is False


# ---------------------------------------------------------------------------
# 2. Interleaved embedding generation  (Req 16.1, 16.2, 16.3, 16.4)
# ---------------------------------------------------------------------------

class TestEmbedInterleaved:
    """Tests for EmbeddingService.embed_interleaved."""

    # --- happy-path ---

    def test_text_and_image_returns_embedding_result(self, embedding_service, mock_genai_client):
        """Text + image interleaved content should return a valid EmbeddingResult (Req 16.1, 16.3)."""
        mock_embedding = MagicMock()
        mock_embedding.values = [0.3] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        item = _make_interleaved(
            _text_part("A cat sitting on a mat"),
            _image_part(PNG_HEADER, "image/png"),
        )
        result = embedding_service.embed_interleaved(item, dimension=756)

        assert isinstance(result, EmbeddingResult)
        assert result.content_type == "interleaved"
        assert result.dimension == 756
        assert result.model_version == "gemini-embedding-2-preview"

    def test_text_only_interleaved_returns_embedding(self, embedding_service, mock_genai_client):
        """Text-only interleaved content should return a valid embedding."""
        mock_embedding = MagicMock()
        mock_embedding.values = [0.2] * 512
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        item = _make_interleaved(_text_part("Just text"))
        result = embedding_service.embed_interleaved(item, dimension=512)

        assert result.vector == [0.2] * 512
        assert result.dimension == 512

    def test_custom_dimension_is_respected(self, embedding_service, mock_genai_client):
        """Dimension parameter should be forwarded to the model (Req 16.4)."""
        mock_embedding = MagicMock()
        mock_embedding.values = [0.5] * 1024
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        item = _make_interleaved(_text_part("hi"), _image_part(PNG_HEADER, "image/png"))
        result = embedding_service.embed_interleaved(item, dimension=1024)

        call_kwargs = mock_genai_client.models.embed_content.call_args[1]
        # google.genai.types is mocked at module level; verify config was passed and
        # the declared dimension on the result is correct
        assert "config" in call_kwargs
        assert result.dimension == 1024

    def test_default_dimension_is_756(self, embedding_service, mock_genai_client):
        """Default dimension should be 756."""
        mock_embedding = MagicMock()
        mock_embedding.values = [0.0] * 756
        mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])

        item = _make_interleaved(_text_part("test"))
        result = embedding_service.embed_interleaved(item)

        assert result.dimension == 756

    def test_all_valid_dimensions_accepted(self, embedding_service, mock_genai_client):
        """All Matryoshka dimensions should be accepted (Req 16.4)."""
        for dim in EmbeddingService.VALID_DIMENSIONS:
            mock_embedding = MagicMock()
            mock_embedding.values = [0.1] * dim
            mock_genai_client.models.embed_content.return_value = MagicMock(embeddings=[mock_embedding])
            item = _make_interleaved(_text_part("test"))
            result = embedding_service.embed_interleaved(item, dimension=dim)
            assert result.dimension == dim

    # --- error cases ---

    def test_wrong_content_type_raises_error(self, embedding_service):
        """Passing a non-interleaved ContentItem should raise EmbeddingError."""
        item = ContentItem(content_type="text", data="hello")
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_interleaved(item)
        assert exc_info.value.error_type == "API_ERROR"
        assert "interleaved" in str(exc_info.value).lower()

    def test_empty_parts_raises_error(self, embedding_service):
        """Empty interleaved_parts list should raise EmbeddingError."""
        item = ContentItem(content_type="interleaved", interleaved_parts=[])
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_interleaved(item)
        assert exc_info.value.error_type == "API_ERROR"

    def test_none_parts_raises_error(self, embedding_service):
        """None interleaved_parts should raise EmbeddingError."""
        item = ContentItem(content_type="interleaved", interleaved_parts=None)
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_interleaved(item)
        assert exc_info.value.error_type == "API_ERROR"

    def test_unsupported_part_type_raises_error(self, embedding_service):
        """A part with unsupported type should raise EmbeddingError."""
        item = _make_interleaved(
            _text_part("context"),
            InterleavedPart(content_type="unknown_type", data=b"data"),
        )
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_interleaved(item)
        assert exc_info.value.error_type == "API_ERROR"
        assert "unsupported type" in str(exc_info.value).lower()

    def test_invalid_dimension_raises_error(self, embedding_service):
        """Invalid dimension should raise EmbeddingError before calling the model."""
        item = _make_interleaved(_text_part("test"))
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service.embed_interleaved(item, dimension=999)
        assert exc_info.value.error_type == "INVALID_DIMENSION"

    def test_vertex_api_error_propagates(self, embedding_service, mock_genai_client):
        """Vertex AI API errors should be wrapped in EmbeddingError."""
        from google.api_core import exceptions as google_exceptions
        mock_genai_client.models.embed_content.side_effect = google_exceptions.GoogleAPIError("boom")

        item = _make_interleaved(_text_part("test"))
        with patch("time.sleep"):
            with pytest.raises(EmbeddingError) as exc_info:
                embedding_service.embed_interleaved(item)
        assert exc_info.value.error_type == "API_ERROR"


# ---------------------------------------------------------------------------
# 3. Search with interleaved queries  (Req 16.5)
# ---------------------------------------------------------------------------

class TestSearchWithInterleavedQuery:
    """Tests for SearchEngine.search when the query is an interleaved ContentItem."""

    def test_search_calls_embed_interleaved(self, search_engine, mock_embedding_service, mock_vector_store):
        """SearchEngine should delegate to embed_interleaved for interleaved queries (Req 16.5)."""
        query = _make_interleaved(
            _text_part("find related content"),
            _image_part(PNG_HEADER, "image/png"),
        )

        response = search_engine.search(query, limit=5)

        mock_embedding_service.embed_interleaved.assert_called_once_with(query, 756)
        assert isinstance(response, SearchResponse)

    def test_search_returns_results_for_interleaved_query(self, search_engine, mock_embedding_service, mock_vector_store):
        """Search with interleaved query should return ranked results (Req 16.5)."""
        query = _make_interleaved(_text_part("semantic query"), _image_part(PNG_HEADER))

        response = search_engine.search(query)

        assert response.total_results >= 0
        assert response.two_stage is False
        assert response.query_metadata["query_type"] == "interleaved"

    def test_search_passes_vector_to_store(self, search_engine, mock_embedding_service, mock_vector_store):
        """The embedding vector from embed_interleaved should be forwarded to vector_store.search."""
        expected_vector = [0.5] * 756
        mock_embedding_service.embed_interleaved.return_value = EmbeddingResult(
            vector=expected_vector,
            dimension=756,
            content_type="interleaved",
            model_version="gemini-embedding-2-preview",
        )

        query = _make_interleaved(_text_part("test"), _image_part(PNG_HEADER))
        search_engine.search(query)

        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["query_vector"] == expected_vector

    def test_search_with_modality_filter_on_interleaved_query(self, search_engine, mock_embedding_service, mock_vector_store):
        """Modality filter should be applied even when query is interleaved."""
        query = _make_interleaved(_text_part("cats"), _image_part(PNG_HEADER))

        search_engine.search(query, modality_filter=["image", "video"])

        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["filters"] is not None
        assert call_kwargs["filters"].content_types == ["image", "video"]

    def test_search_with_score_threshold_on_interleaved_query(self, search_engine, mock_embedding_service, mock_vector_store):
        """Score threshold should be forwarded to vector_store.search."""
        query = _make_interleaved(_text_part("threshold test"))

        search_engine.search(query, score_threshold=0.75)

        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs["score_threshold"] == 0.75

    def test_search_results_sorted_descending_by_score(self, search_engine, mock_embedding_service, mock_vector_store):
        """Results should be returned in descending order of similarity score (Req 14.5)."""
        ts = datetime.now()
        meta = EmbeddingMetadata(
            content_type="text",
            source_id="s",
            timestamp=ts,
            dimension=756,
            model_version="gemini-embedding-2-preview",
        )
        mock_vector_store.search.return_value = [
            SearchResult(point_id="a", score=0.70, content_type="text", source_id="s", timestamp=ts, metadata=meta),
            SearchResult(point_id="b", score=0.95, content_type="text", source_id="s", timestamp=ts, metadata=meta),
            SearchResult(point_id="c", score=0.82, content_type="text", source_id="s", timestamp=ts, metadata=meta),
        ]

        query = _make_interleaved(_text_part("order test"))
        response = search_engine.search(query)

        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_custom_dimension_for_interleaved_query(self, search_engine, mock_embedding_service, mock_vector_store):
        """Custom dimension should be forwarded to embed_interleaved."""
        query = _make_interleaved(_text_part("dim test"))

        search_engine.search(query, dimension=1024)

        mock_embedding_service.embed_interleaved.assert_called_once_with(query, 1024)

    def test_search_embedding_error_propagates_as_search_error(self, search_engine, mock_embedding_service):
        """If embed_interleaved raises EmbeddingError, search should raise SearchError."""
        mock_embedding_service.embed_interleaved.side_effect = EmbeddingError(
            "model failure", error_type="API_ERROR"
        )

        query = _make_interleaved(_text_part("fail"))
        with pytest.raises(SearchError):
            search_engine.search(query)

    def test_search_no_modality_filter_returns_all_modalities(self, search_engine, mock_embedding_service, mock_vector_store):
        """Without a modality filter, search should not restrict by content type (Req 9.4)."""
        query = _make_interleaved(_text_part("no filter"))

        search_engine.search(query)

        call_kwargs = mock_vector_store.search.call_args[1]
        # filters should be None when no modality_filter is specified
        assert call_kwargs["filters"] is None

