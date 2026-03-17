"""Search orchestration and query processing.

This module provides :class:`SearchEngine`, which coordinates query
embedding via :class:`~multimodal_search.embedding_service.EmbeddingService`
and vector retrieval via :class:`~multimodal_search.vector_store.VectorStore`
to implement single-stage, two-stage, cross-modal, and multilingual search.
"""

import time
from typing import Any, Dict, List, Optional

from multimodal_search.models import (
    ContentItem,
    SearchFilters,
    SearchResponse,
)
from multimodal_search.embedding_service import EmbeddingService
from multimodal_search.vector_store import VectorStore
from multimodal_search.exceptions import SearchError


class SearchEngine:
    """Orchestrates search operations including query embedding, vector search, and result ranking.

    The engine supports four search strategies:

    1. **Single-stage** (:meth:`search`) – embed the query at a single
       dimension and retrieve the top-k most similar vectors.
    2. **Two-stage** (:meth:`search_two_stage`) – fast candidate
       retrieval with a low-dimension named vector followed by accurate
       re-ranking with a high-dimension named vector.
    3. **Cross-modal** (:meth:`search_cross_modal`) – query with any
       modality and restrict results to specific target modalities.
    4. **Multilingual** (:meth:`search_multilingual`) – text query in
       any language with optional language-based result filtering.

    All search methods return a :class:`~multimodal_search.models.SearchResponse`
    with results sorted in descending order of cosine similarity score.

    Example:
        >>> engine = SearchEngine(
        ...     embedding_service=embedding_service,
        ...     vector_store=vector_store,
        ... )
        >>> query = ContentItem(content_type="text", data="sunset over the ocean")
        >>> response = engine.search(query, limit=5)
        >>> response.total_results
        5
    """

    DEFAULT_DIMENSION = 756
    DEFAULT_LIMIT = 10
    DEFAULT_FIRST_STAGE_DIMENSION = 256
    DEFAULT_SECOND_STAGE_DIMENSION = 1024
    DEFAULT_FIRST_STAGE_LIMIT = 100

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        default_dimension: int = DEFAULT_DIMENSION,
        default_limit: int = DEFAULT_LIMIT,
    ):
        """
        Initialize SearchEngine with required services.

        Args:
            embedding_service: Service for generating embeddings via Vertex AI
            vector_store: Service for storing and retrieving vectors from Qdrant
            default_dimension: Default embedding dimension for searches
            default_limit: Default maximum number of results to return
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.default_dimension = default_dimension
        self.default_limit = default_limit

    def _embed_query(self, query: ContentItem, dimension: int) -> List[float]:
        """
        Embed a query content item at the specified dimension.

        Args:
            query: Query content item (any modality)
            dimension: Embedding dimension

        Returns:
            Embedding vector as list of floats

        Raises:
            SearchError: If the content type is unknown or embedding fails
        """
        try:
            if query.content_type == "text":
                result = self.embedding_service.embed_text(str(query.data), dimension)
            elif query.content_type == "image":
                result = self.embedding_service.embed_image(bytes(query.data), dimension)
            elif query.content_type == "audio":
                result = self.embedding_service.embed_audio(bytes(query.data), dimension)
            elif query.content_type == "video":
                result = self.embedding_service.embed_video(bytes(query.data), dimension)
            elif query.content_type == "pdf":
                result = self.embedding_service.embed_pdf(bytes(query.data), dimension)
            elif query.content_type == "interleaved":
                result = self.embedding_service.embed_interleaved(query, dimension)
            else:
                raise SearchError(
                    f"Unknown content type: {query.content_type}",
                    error_type="INVALID_QUERY",
                )
            return result.vector
        except SearchError:
            raise
        except Exception as e:
            raise SearchError(
                f"Failed to embed query: {str(e)}",
                error_type="INVALID_QUERY",
            )

    def search(
        self,
        query: ContentItem,
        limit: int = DEFAULT_LIMIT,
        modality_filter: Optional[List[str]] = None,
        dimension: int = DEFAULT_DIMENSION,
        score_threshold: Optional[float] = None,
    ) -> SearchResponse:
        """
        Perform single-stage semantic search.

        Embeds the query and retrieves semantically similar content from all modalities
        (or filtered modalities). Results are ranked by cosine similarity score.

        Args:
            query: Query content (any modality — text, image, audio, video, pdf)
            limit: Maximum number of results to return
            modality_filter: Optional list of modalities to restrict results to
            dimension: Matryoshka embedding dimension for search
            score_threshold: Minimum similarity score threshold (0-1)

        Returns:
            SearchResponse with results ranked by similarity score

        Raises:
            SearchError: If the search operation fails
        """
        try:
            start_time = time.time()

            query_vector = self._embed_query(query, dimension)

            filters = SearchFilters(content_types=modality_filter) if modality_filter else None

            results = self.vector_store.search(
                query_vector=query_vector,
                limit=limit,
                filters=filters,
                score_threshold=score_threshold,
            )

            # Ensure results are sorted descending by score (req 14.5)
            results.sort(key=lambda r: r.score, reverse=True)

            search_time_ms = (time.time() - start_time) * 1000

            return SearchResponse(
                results=results,
                query_metadata={
                    "query_type": query.content_type,
                    "dimension": dimension,
                    "modality_filter": modality_filter,
                    "score_threshold": score_threshold,
                },
                total_results=len(results),
                search_time_ms=search_time_ms,
                two_stage=False,
            )

        except SearchError:
            raise
        except Exception as e:
            raise SearchError(
                f"Search operation failed: {str(e)}",
                error_type="SEARCH_ERROR",
            )

    def search_two_stage(
        self,
        query: ContentItem,
        first_stage_dimension: int = DEFAULT_FIRST_STAGE_DIMENSION,
        second_stage_dimension: int = DEFAULT_SECOND_STAGE_DIMENSION,
        first_stage_limit: int = DEFAULT_FIRST_STAGE_LIMIT,
        final_limit: int = DEFAULT_LIMIT,
        modality_filter: Optional[List[str]] = None,
    ) -> SearchResponse:
        """
        Perform two-stage retrieval for speed-accuracy optimization.

        Stage 1: Fast candidate retrieval using a lower-dimension named vector.
        Stage 2: Re-rank candidates using a higher-dimension named vector.

        Requires that content was stored with named vectors at both dimensions
        (e.g., "dim_256" and "dim_1024").

        Args:
            query: Query content (any modality)
            first_stage_dimension: Dimension for initial fast retrieval (e.g., 256)
            second_stage_dimension: Dimension for accurate re-ranking (e.g., 1024)
            first_stage_limit: Number of candidates to retrieve in stage 1
            final_limit: Final number of results after re-ranking
            modality_filter: Optional list of modalities to restrict results to

        Returns:
            SearchResponse with re-ranked results and two-stage metadata

        Raises:
            SearchError: If the search operation fails
        """
        try:
            start_time = time.time()

            filters = SearchFilters(content_types=modality_filter) if modality_filter else None

            # Stage 1: fast retrieval with lower dimension
            low_vector = self._embed_query(query, first_stage_dimension)
            candidates = self.vector_store.search_with_named_vector(
                query_vector=low_vector,
                vector_name=f"dim_{first_stage_dimension}",
                limit=first_stage_limit,
                filters=filters,
            )

            if not candidates:
                search_time_ms = (time.time() - start_time) * 1000
                return SearchResponse(
                    results=[],
                    query_metadata={
                        "query_type": query.content_type,
                        "first_stage_dimension": first_stage_dimension,
                        "second_stage_dimension": second_stage_dimension,
                        "candidates_retrieved": 0,
                        "modality_filter": modality_filter,
                    },
                    total_results=0,
                    search_time_ms=search_time_ms,
                    two_stage=True,
                )

            # Stage 2: re-rank candidates with higher dimension
            high_vector = self._embed_query(query, second_stage_dimension)

            # Restrict re-ranking to the candidate point IDs via custom filter
            candidate_ids = [c.point_id for c in candidates]
            rerank_filters = SearchFilters(
                content_types=modality_filter,
                custom_filters={"_point_ids": candidate_ids},
            )

            final_results = self.vector_store.search_with_named_vector(
                query_vector=high_vector,
                vector_name=f"dim_{second_stage_dimension}",
                limit=final_limit,
                filters=rerank_filters,
            )

            # Sort final results descending by score
            final_results.sort(key=lambda r: r.score, reverse=True)

            search_time_ms = (time.time() - start_time) * 1000

            return SearchResponse(
                results=final_results,
                query_metadata={
                    "query_type": query.content_type,
                    "first_stage_dimension": first_stage_dimension,
                    "second_stage_dimension": second_stage_dimension,
                    "candidates_retrieved": len(candidates),
                    "modality_filter": modality_filter,
                },
                total_results=len(final_results),
                search_time_ms=search_time_ms,
                two_stage=True,
            )

        except SearchError:
            raise
        except Exception as e:
            raise SearchError(
                f"Two-stage search operation failed: {str(e)}",
                error_type="SEARCH_ERROR",
            )

    def search_cross_modal(
        self,
        query: ContentItem,
        target_modalities: List[str],
        limit: int = DEFAULT_LIMIT,
        dimension: int = DEFAULT_DIMENSION,
    ) -> SearchResponse:
        """
        Search across specific target modalities using any query modality.

        Leverages the unified vector space of Gemini Embedding 2 to retrieve
        semantically similar content from different modalities than the query.
        For example, a text query can retrieve images, videos, or PDFs.

        Args:
            query: Query content (any modality)
            target_modalities: Modalities to retrieve results from (e.g., ["image", "video"])
            limit: Maximum number of results to return
            dimension: Matryoshka embedding dimension for search

        Returns:
            SearchResponse with results from target modalities only

        Raises:
            SearchError: If the search operation fails
        """
        try:
            start_time = time.time()

            query_vector = self._embed_query(query, dimension)

            # Filter results to only the requested target modalities
            filters = SearchFilters(content_types=target_modalities)

            results = self.vector_store.search(
                query_vector=query_vector,
                limit=limit,
                filters=filters,
            )

            results.sort(key=lambda r: r.score, reverse=True)

            search_time_ms = (time.time() - start_time) * 1000

            return SearchResponse(
                results=results,
                query_metadata={
                    "query_type": query.content_type,
                    "target_modalities": target_modalities,
                    "dimension": dimension,
                    "cross_modal": True,
                },
                total_results=len(results),
                search_time_ms=search_time_ms,
                two_stage=False,
            )

        except SearchError:
            raise
        except Exception as e:
            raise SearchError(
                f"Cross-modal search operation failed: {str(e)}",
                error_type="SEARCH_ERROR",
            )

    def search_multilingual(
        self,
        query_text: str,
        query_language: str,
        target_languages: Optional[List[str]] = None,
        limit: int = DEFAULT_LIMIT,
        dimension: int = DEFAULT_DIMENSION,
    ) -> SearchResponse:
        """
        Perform cross-lingual semantic search.

        Gemini Embedding 2 natively supports 100+ languages and preserves
        cross-lingual semantic similarity in the unified vector space. A query
        in one language will retrieve semantically relevant content in any language.

        Args:
            query_text: Text query in any supported language
            query_language: BCP-47 language code of the query (e.g., "en", "es", "fr")
            target_languages: Optional list of language codes to restrict results to
            limit: Maximum number of results to return
            dimension: Matryoshka embedding dimension for search

        Returns:
            SearchResponse with multilingual results ranked by semantic similarity

        Raises:
            SearchError: If the search operation fails
        """
        try:
            start_time = time.time()

            # Gemini Embedding 2 auto-handles multilingual text — no language hint needed
            query_item = ContentItem(content_type="text", data=query_text)
            query_vector = self._embed_query(query_item, dimension)

            # Optionally restrict results to specific languages
            filters = SearchFilters(languages=target_languages) if target_languages else None

            results = self.vector_store.search(
                query_vector=query_vector,
                limit=limit,
                filters=filters,
            )

            results.sort(key=lambda r: r.score, reverse=True)

            search_time_ms = (time.time() - start_time) * 1000

            return SearchResponse(
                results=results,
                query_metadata={
                    "query_type": "text",
                    "query_language": query_language,
                    "target_languages": target_languages,
                    "dimension": dimension,
                    "multilingual": True,
                },
                total_results=len(results),
                search_time_ms=search_time_ms,
                two_stage=False,
            )

        except SearchError:
            raise
        except Exception as e:
            raise SearchError(
                f"Multilingual search operation failed: {str(e)}",
                error_type="SEARCH_ERROR",
            )
