"""Custom exceptions for the multimodal search system.

All exceptions derive from :class:`MultimodalSearchError` so callers can
catch the entire hierarchy with a single ``except`` clause when needed.
Each concrete exception class also exposes an ``ERROR_TYPES`` class
attribute that maps short error-code strings to human-readable
descriptions.
"""


class MultimodalSearchError(Exception):
    """Base exception for all multimodal search system errors.

    Catch this class to handle any error raised by the system without
    needing to enumerate every subclass.

    Example:
        >>> try:
        ...     api.embed_content(item)
        ... except MultimodalSearchError as exc:
        ...     print(f"Search system error: {exc}")
    """

    pass


class ValidationError(MultimodalSearchError):
    """Raised when content fails pre-embedding validation.

    The :class:`~multimodal_search.content_processor.ContentProcessor`
    raises this exception before any Vertex AI API call is made, so
    invalid content is rejected early without wasting quota.

    Attributes:
        error_type: Short error code identifying the failure reason.
            See :attr:`ERROR_TYPES` for valid values.

    Class Attributes:
        ERROR_TYPES: Mapping of error-code strings to descriptions.

    Example:
        >>> raise ValidationError(
        ...     "Image format 'bmp' not supported. Supported formats: PNG, JPEG",
        ...     error_type="INVALID_FORMAT",
        ... )
    """

    ERROR_TYPES = {
        "INVALID_FORMAT": "Content format not supported",
        "SIZE_EXCEEDED": "Content exceeds size limits",
        "DURATION_EXCEEDED": "Audio/video duration too long",
        "PAGE_LIMIT_EXCEEDED": "PDF has too many pages",
        "MIME_TYPE_MISMATCH": "MIME type doesn't match content",
        "EMPTY_CONTENT": "Content is empty or invalid",
    }

    def __init__(self, message: str, error_type: str = "VALIDATION_ERROR"):
        """Initialise with a descriptive message and an error type code.

        Args:
            message: Human-readable description of the validation failure.
            error_type: Short error code.  Should be one of the keys in
                :attr:`ERROR_TYPES` or ``"VALIDATION_ERROR"`` for generic
                failures.
        """
        self.error_type = error_type
        super().__init__(message)


class EmbeddingError(MultimodalSearchError):
    """Raised when a Vertex AI embedding request fails.

    This exception covers authentication failures, rate limiting, invalid
    dimension values, generic API errors, and network issues.

    Attributes:
        error_type: Short error code identifying the failure reason.
            See :attr:`ERROR_TYPES` for valid values.

    Class Attributes:
        ERROR_TYPES: Mapping of error-code strings to descriptions.

    Example:
        >>> raise EmbeddingError(
        ...     "Rate limit exceeded after 3 retries",
        ...     error_type="RATE_LIMIT",
        ... )
    """

    ERROR_TYPES = {
        "AUTH_FAILED": "Authentication failed - check credentials",
        "RATE_LIMIT": "Rate limit exceeded - retry after delay",
        "INVALID_DIMENSION": "Unsupported dimension value",
        "API_ERROR": "Vertex AI API error",
        "NETWORK_ERROR": "Network connectivity issue",
        "QUOTA_EXCEEDED": "Project quota exceeded",
    }

    def __init__(self, message: str, error_type: str = "EMBEDDING_ERROR"):
        """Initialise with a descriptive message and an error type code.

        Args:
            message: Human-readable description of the embedding failure.
            error_type: Short error code.  Should be one of the keys in
                :attr:`ERROR_TYPES` or ``"EMBEDDING_ERROR"`` for generic
                failures.
        """
        self.error_type = error_type
        super().__init__(message)


class StorageError(MultimodalSearchError):
    """Raised when a Qdrant storage or retrieval operation fails.

    Covers connection failures, missing collections, vector dimension
    mismatches, and point-not-found scenarios.

    Attributes:
        error_type: Short error code identifying the failure reason.
            See :attr:`ERROR_TYPES` for valid values.

    Class Attributes:
        ERROR_TYPES: Mapping of error-code strings to descriptions.

    Example:
        >>> raise StorageError(
        ...     "Collection 'multimodal_embeddings' does not exist",
        ...     error_type="COLLECTION_NOT_FOUND",
        ... )
    """

    ERROR_TYPES = {
        "CONNECTION_FAILED": "Cannot connect to Qdrant",
        "COLLECTION_NOT_FOUND": "Collection doesn't exist",
        "INVALID_VECTOR": "Vector dimension mismatch",
        "STORAGE_FULL": "Qdrant storage capacity reached",
        "POINT_NOT_FOUND": "Requested point ID not found",
    }

    def __init__(self, message: str, error_type: str = "STORAGE_ERROR"):
        """Initialise with a descriptive message and an error type code.

        Args:
            message: Human-readable description of the storage failure.
            error_type: Short error code.  Should be one of the keys in
                :attr:`ERROR_TYPES` or ``"STORAGE_ERROR"`` for generic
                failures.
        """
        self.error_type = error_type
        super().__init__(message)


class SearchError(MultimodalSearchError):
    """Raised when a search operation fails.

    This exception is raised by the
    :class:`~multimodal_search.search_engine.SearchEngine` when query
    embedding, vector search, or result processing encounters an error.

    Attributes:
        error_type: Short error code identifying the failure reason.
            See :attr:`ERROR_TYPES` for valid values.

    Class Attributes:
        ERROR_TYPES: Mapping of error-code strings to descriptions.

    Example:
        >>> raise SearchError(
        ...     "Unknown content type: 'binary'",
        ...     error_type="INVALID_QUERY",
        ... )
    """

    ERROR_TYPES = {
        "INVALID_QUERY": "Query content invalid",
        "NO_RESULTS": "No results found",
        "FILTER_ERROR": "Invalid filter configuration",
        "TIMEOUT": "Search operation timed out",
    }

    def __init__(self, message: str, error_type: str = "SEARCH_ERROR"):
        """Initialise with a descriptive message and an error type code.

        Args:
            message: Human-readable description of the search failure.
            error_type: Short error code.  Should be one of the keys in
                :attr:`ERROR_TYPES` or ``"SEARCH_ERROR"`` for generic
                failures.
        """
        self.error_type = error_type
        super().__init__(message)
