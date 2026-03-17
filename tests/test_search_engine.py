"""Tests for SearchEngine class."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from multimodal_search.search_engine import SearchEngine
from multimodal_search.models import (
    ContentItem,
    EmbeddingResult,
    SearchResult,
    SearchResponse,
    EmbeddingMetadata,
)
from multimodal_search.exceptions import SearchError


@pytest.fixture
def mock_embedding_service():
    """Create a mock EmbeddingService."""
    service = Mock()
    
    # Mock embed_text to return a valid EmbeddingResult
    service.embed_text.return_value = EmbeddingResult(
        vector=[0.1] * 756,
        dimension=756,
        content_type="text",
        model_version="gemini-embedding-2-preview"
    )
    
    # Mock embed_image
    service.embed_image.return_value = EmbeddingResult(
        vector=[0.2] * 756,
        dimension=756,
        content_type="image",
        model_version="gemini-embedding-2-preview"
    )
    
    return service


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore."""
    store = Mock()
    
    # Mock search to return sample results
    store.search.return_value = [
        SearchResult(
            point_id="test-id-1",
            score=0.95,
            content_type="text",
            source_id="source-1",
            timestamp=datetime.now(),
            metadata=EmbeddingMetadata(
                content_type="text",
                source_id="source-1",
                timestamp=datetime.now(),
                dimension=756,
                model_version="gemini-embedding-2-preview"
            )
        )
    ]
    
    # Mock search_with_named_vector
    store.search_with_named_vector.return_value = [
        SearchResult(
            point_id="test-id-2",
            score=0.90,
            content_type="image",
            source_id="source-2",
            timestamp=datetime.now(),
            metadata=EmbeddingMetadata(
                content_type="image",
                source_id="source-2",
                timestamp=datetime.now(),
                dimension=256,
                model_version="gemini-embedding-2-preview"
            )
        )
    ]
    
    return store


@pytest.fixture
def search_engine(mock_embedding_service, mock_vector_store):
    """Create a SearchEngine instance with mocked dependencies."""
    return SearchEngine(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store
    )


class TestSearchEngine:
    """Test cases for SearchEngine class."""
    
    def test_initialization(self, mock_embedding_service, mock_vector_store):
        """Test SearchEngine initialization."""
        engine = SearchEngine(mock_embedding_service, mock_vector_store)
        assert engine.embedding_service == mock_embedding_service
        assert engine.vector_store == mock_vector_store
    
    def test_search_with_text_query(self, search_engine, mock_embedding_service, mock_vector_store):
        """Test single-stage search with text query."""
        query = ContentItem(
            content_type="text",
            data="test query"
        )
        
        response = search_engine.search(query, limit=10)
        
        # Verify embedding service was called
        mock_embedding_service.embed_text.assert_called_once_with("test query", 756)
        
        # Verify vector store search was called
        mock_vector_store.search.assert_called_once()
        
        # Verify response structure
        assert isinstance(response, SearchResponse)
        assert response.total_results == 1
        assert response.two_stage is False
        assert "query_type" in response.query_metadata
        assert response.query_metadata["query_type"] == "text"
    
    def test_search_with_modality_filter(self, search_engine, mock_embedding_service, mock_vector_store):
        """Test search with modality filter."""
        query = ContentItem(
            content_type="text",
            data="test query"
        )
        
        response = search_engine.search(
            query,
            limit=10,
            modality_filter=["image", "video"]
        )
        
        # Verify search was called with filters
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["filters"] is not None
        assert call_args[1]["filters"].content_types == ["image", "video"]
    
    def test_search_with_score_threshold(self, search_engine, mock_embedding_service, mock_vector_store):
        """Test search with score threshold."""
        query = ContentItem(
            content_type="text",
            data="test query"
        )
        
        response = search_engine.search(
            query,
            limit=10,
            score_threshold=0.8
        )
        
        # Verify search was called with score threshold
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["score_threshold"] == 0.8
    
    def test_search_with_invalid_content_type(self, search_engine):
        """Test search with invalid content type raises error."""
        query = ContentItem(
            content_type="invalid_type",
            data="test"
        )
        
        with pytest.raises(SearchError) as exc_info:
            search_engine.search(query)
        
        assert "Unknown content type" in str(exc_info.value)
        assert exc_info.value.error_type == "INVALID_QUERY"
    
    def test_search_cross_modal(self, search_engine, mock_embedding_service, mock_vector_store):
        """Test cross-modal search."""
        query = ContentItem(
            content_type="text",
            data="find images of cats"
        )
        
        response = search_engine.search_cross_modal(
            query,
            target_modalities=["image"],
            limit=10
        )
        
        # Verify embedding service was called
        mock_embedding_service.embed_text.assert_called()
        
        # Verify search was called with modality filter
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["filters"] is not None
        assert call_args[1]["filters"].content_types == ["image"]
        
        # Verify response metadata
        assert response.query_metadata["cross_modal"] is True
        assert response.query_metadata["target_modalities"] == ["image"]
    
    def test_search_multilingual(self, search_engine, mock_embedding_service, mock_vector_store):
        """Test multilingual search."""
        response = search_engine.search_multilingual(
            query_text="hello world",
            query_language="en",
            target_languages=["es", "fr"],
            limit=10
        )
        
        # Verify embedding service was called
        mock_embedding_service.embed_text.assert_called_once_with("hello world", 756)
        
        # Verify search was called with language filter
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["filters"] is not None
        assert call_args[1]["filters"].languages == ["es", "fr"]
        
        # Verify response metadata
        assert response.query_metadata["multilingual"] is True
        assert response.query_metadata["query_language"] == "en"
        assert response.query_metadata["target_languages"] == ["es", "fr"]
    
    def test_search_two_stage(self, search_engine, mock_embedding_service, mock_vector_store):
        """Test two-stage retrieval."""
        query = ContentItem(
            content_type="text",
            data="test query"
        )
        
        # Mock both stages
        mock_embedding_service.embed_text.side_effect = [
            EmbeddingResult(
                vector=[0.1] * 256,
                dimension=256,
                content_type="text",
                model_version="gemini-embedding-2-preview"
            ),
            EmbeddingResult(
                vector=[0.1] * 1024,
                dimension=1024,
                content_type="text",
                model_version="gemini-embedding-2-preview"
            )
        ]
        
        response = search_engine.search_two_stage(
            query,
            first_stage_dimension=256,
            second_stage_dimension=1024,
            first_stage_limit=100,
            final_limit=10
        )
        
        # Verify embedding service was called twice (once per stage)
        assert mock_embedding_service.embed_text.call_count == 2
        
        # Verify vector store search was called twice
        assert mock_vector_store.search_with_named_vector.call_count == 2
        
        # Verify response metadata
        assert response.two_stage is True
        assert response.query_metadata["first_stage_dimension"] == 256
        assert response.query_metadata["second_stage_dimension"] == 1024
    
    def test_search_two_stage_no_candidates(self, search_engine, mock_embedding_service, mock_vector_store):
        """Test two-stage retrieval when no candidates found in first stage."""
        query = ContentItem(
            content_type="text",
            data="test query"
        )
        
        # Mock first stage to return empty results
        mock_vector_store.search_with_named_vector.return_value = []
        
        response = search_engine.search_two_stage(query)
        
        # Verify response is empty
        assert response.total_results == 0
        assert response.results == []
        assert response.two_stage is True
        assert response.query_metadata["candidates_retrieved"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
