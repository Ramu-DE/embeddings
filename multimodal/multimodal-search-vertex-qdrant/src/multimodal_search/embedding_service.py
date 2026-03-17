"""Embedding service for generating embeddings using Vertex AI Gemini Embedding 2.

This module provides :class:`EmbeddingService`, which wraps the ``google-genai``
client to generate Matryoshka embeddings for text, images, audio, video, PDF,
and interleaved multimodal content using the ``gemini-embedding-2-preview`` model.
All modalities are mapped to a shared semantic vector space, enabling cross-modal
similarity search.

Usage::

    from google import genai
    service = EmbeddingService(project_id="my-project", location="global")
    result = service.embed_text("Hello, world!", dimension=756)
"""

import base64
import time
from typing import Dict, List, Optional

from google.api_core import exceptions as google_exceptions

from multimodal_search.models import ContentItem, EmbeddingResult
from multimodal_search.exceptions import EmbeddingError


# MIME type → genai Part inline_data content_type mapping
_MIME_FALLBACKS = {
    "image": "image/jpeg",
    "audio": "audio/mp3",
    "video": "video/mp4",
    "pdf": "application/pdf",
}


class EmbeddingService:
    """Interfaces with Vertex AI Gemini Embedding 2 via the google-genai SDK.

    Uses ``google.genai.Client(vertexai=True)`` to call the
    ``gemini-embedding-2-preview`` model, which supports text, image, audio,
    video, PDF, and interleaved multimodal content in a unified vector space.

    All embedding calls are executed through :meth:`_execute_with_retry`,
    which applies exponential-backoff retry logic for transient errors
    (rate limits, timeouts, generic API errors).

    Supported Matryoshka dimensions:
        128, 256, 512, 756, 1024, 1536, 2048, 3072

    Example:
        >>> service = EmbeddingService(project_id="my-gcp-project", location="global")
        >>> result = service.embed_text("Hello, world!", dimension=256)
        >>> len(result.vector)
        256
    """

    VALID_DIMENSIONS = [128, 256, 512, 756, 1024, 1536, 2048, 3072]
    DEFAULT_DIMENSION = 756
    MODEL_NAME = "gemini-embedding-2-preview"

    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0  # seconds

    def __init__(
        self,
        project_id: str,
        location: str = "global",
        credentials_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialise the EmbeddingService with a google-genai Vertex AI client.

        Args:
            project_id: GCP project ID that has the Vertex AI API enabled.
            location: GCP region for the Vertex AI endpoint.  Use ``"global"``
                for the Gemini Embedding 2 model (default).
            credentials_path: Optional filesystem path to a service account
                JSON key file.  When ``None``, Application Default Credentials
                are used (unless api_key is provided).
            api_key: Optional Vertex AI API key (``AQ.xxx`` format).  When
                provided, takes precedence over ADC and credentials_path.
                Falls back to the ``VERTEX_AI_API_KEY`` environment variable.

        Raises:
            EmbeddingError: If the google-genai client cannot be initialised
                (e.g. missing credentials or import error).
        """
        import os
        self.project_id = project_id
        self.location = location

        # Resolve API key: explicit arg > env var
        resolved_api_key = api_key or os.environ.get("VERTEX_AI_API_KEY")

        if resolved_api_key:
            # Use Vertex AI API key auth — no ADC needed
            os.environ["GOOGLE_API_KEY"] = resolved_api_key
        elif credentials_path:
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", credentials_path)

        try:
            from google import genai
            if resolved_api_key:
                # Google AI Studio API key — uses Gemini Developer API (no project/location needed)
                self._client = genai.Client(api_key=resolved_api_key)
            else:
                # ADC / service account auth via Vertex AI
                self._client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to initialise google-genai client: {exc}",
                error_type="AUTH_FAILED",
            )

    # ------------------------------------------------------------------
    # Dimension validation
    # ------------------------------------------------------------------

    def validate_dimension(self, dimension: int) -> None:
        """Validate that *dimension* is a supported Matryoshka value.

        Args:
            dimension: Embedding dimension to validate.

        Raises:
            EmbeddingError: If *dimension* is not in :attr:`VALID_DIMENSIONS`.
        """
        if dimension not in self.VALID_DIMENSIONS:
            raise EmbeddingError(
                f"Dimension {dimension} not supported. "
                f"Valid dimensions: {self.VALID_DIMENSIONS}",
                error_type="INVALID_DIMENSION",
            )

    # ------------------------------------------------------------------
    # Per-modality embedding methods
    # ------------------------------------------------------------------

    def embed_text(self, text: str, dimension: int = DEFAULT_DIMENSION) -> EmbeddingResult:
        """Generate an embedding for plain text content.

        Args:
            text: Text string to embed.  Supports 100+ languages.
            dimension: Matryoshka dimension (128–3072).  Defaults to 756.

        Returns:
            :class:`~multimodal_search.models.EmbeddingResult` containing the
            embedding vector and metadata.

        Raises:
            EmbeddingError: If *dimension* is invalid or the API call fails.

        Example:
            >>> result = service.embed_text("sunset over the ocean", dimension=512)
            >>> len(result.vector)
            512
        """
        self.validate_dimension(dimension)

        def _call():
            from google.genai import types
            response = self._client.models.embed_content(
                model=self.MODEL_NAME,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=dimension),
            )
            return response.embeddings[0].values

        vector = self._execute_with_retry(_call)
        return EmbeddingResult(
            vector=vector,
            dimension=dimension,
            content_type="text",
            model_version=self.MODEL_NAME,
        )

    def embed_image(self, image_data: bytes, dimension: int = DEFAULT_DIMENSION,
                    mime_type: str = "image/jpeg") -> EmbeddingResult:
        """Generate an embedding for image content (PNG or JPEG).

        Args:
            image_data: Raw image bytes.
            dimension: Matryoshka dimension (128–3072).  Defaults to 756.
            mime_type: MIME type of the image (``"image/jpeg"`` or
                ``"image/png"``).  Defaults to ``"image/jpeg"``.

        Returns:
            :class:`~multimodal_search.models.EmbeddingResult` with the image
            embedding vector.

        Raises:
            EmbeddingError: If *dimension* is invalid or the API call fails.
        """
        self.validate_dimension(dimension)

        def _call():
            from google.genai import types
            part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
            response = self._client.models.embed_content(
                model=self.MODEL_NAME,
                contents=part,
                config=types.EmbedContentConfig(output_dimensionality=dimension),
            )
            return response.embeddings[0].values

        vector = self._execute_with_retry(_call)
        return EmbeddingResult(
            vector=vector,
            dimension=dimension,
            content_type="image",
            model_version=self.MODEL_NAME,
        )

    def embed_audio(self, audio_data: bytes, dimension: int = DEFAULT_DIMENSION,
                    mime_type: str = "audio/mp3") -> EmbeddingResult:
        """Generate an embedding for audio content (MP3 or WAV).

        Args:
            audio_data: Raw audio bytes.
            dimension: Matryoshka dimension (128–3072).  Defaults to 756.
            mime_type: MIME type of the audio (``"audio/mp3"`` or
                ``"audio/wav"``).  Defaults to ``"audio/mp3"``.

        Returns:
            :class:`~multimodal_search.models.EmbeddingResult` with the audio
            embedding vector.

        Raises:
            EmbeddingError: If *dimension* is invalid or the API call fails.
        """
        self.validate_dimension(dimension)

        def _call():
            from google.genai import types
            part = types.Part.from_bytes(data=audio_data, mime_type=mime_type)
            response = self._client.models.embed_content(
                model=self.MODEL_NAME,
                contents=part,
                config=types.EmbedContentConfig(output_dimensionality=dimension),
            )
            return response.embeddings[0].values

        vector = self._execute_with_retry(_call)
        return EmbeddingResult(
            vector=vector,
            dimension=dimension,
            content_type="audio",
            model_version=self.MODEL_NAME,
        )

    def embed_video(self, video_data: bytes, dimension: int = DEFAULT_DIMENSION,
                    mime_type: str = "video/mp4") -> EmbeddingResult:
        """Generate an embedding for video content (MP4 or MOV).

        Args:
            video_data: Raw video bytes.
            dimension: Matryoshka dimension (128–3072).  Defaults to 756.
            mime_type: MIME type of the video (``"video/mp4"`` or
                ``"video/quicktime"``).  Defaults to ``"video/mp4"``.

        Returns:
            :class:`~multimodal_search.models.EmbeddingResult` with the video
            embedding vector.

        Raises:
            EmbeddingError: If *dimension* is invalid or the API call fails.
        """
        self.validate_dimension(dimension)

        def _call():
            from google.genai import types
            part = types.Part.from_bytes(data=video_data, mime_type=mime_type)
            response = self._client.models.embed_content(
                model=self.MODEL_NAME,
                contents=part,
                config=types.EmbedContentConfig(output_dimensionality=dimension),
            )
            return response.embeddings[0].values

        vector = self._execute_with_retry(_call)
        return EmbeddingResult(
            vector=vector,
            dimension=dimension,
            content_type="video",
            model_version=self.MODEL_NAME,
        )

    def embed_pdf(self, pdf_data: bytes, dimension: int = DEFAULT_DIMENSION) -> EmbeddingResult:
        """Generate an embedding for a PDF document (up to 6 pages).

        Args:
            pdf_data: Raw PDF bytes.
            dimension: Matryoshka dimension (128–3072).  Defaults to 756.

        Returns:
            :class:`~multimodal_search.models.EmbeddingResult` with the PDF
            embedding vector.

        Raises:
            EmbeddingError: If *dimension* is invalid or the API call fails.
        """
        self.validate_dimension(dimension)

        def _call():
            from google.genai import types
            part = types.Part.from_bytes(data=pdf_data, mime_type="application/pdf")
            response = self._client.models.embed_content(
                model=self.MODEL_NAME,
                contents=part,
                config=types.EmbedContentConfig(output_dimensionality=dimension),
            )
            return response.embeddings[0].values

        vector = self._execute_with_retry(_call)
        return EmbeddingResult(
            vector=vector,
            dimension=dimension,
            content_type="pdf",
            model_version=self.MODEL_NAME,
        )

    # ------------------------------------------------------------------
    # Batch and multi-dimension embedding
    # ------------------------------------------------------------------

    def embed_batch(
        self, content_items: List[ContentItem], dimension: int = DEFAULT_DIMENSION
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple content items.

        Items are processed sequentially.  If any item fails, an
        :class:`~multimodal_search.exceptions.EmbeddingError` is raised
        immediately with the index of the failing item.

        Args:
            content_items: List of :class:`~multimodal_search.models.ContentItem`
                objects to embed.  Mixed modalities are supported.
            dimension: Matryoshka dimension applied to all items.

        Returns:
            List of :class:`~multimodal_search.models.EmbeddingResult` in the
            same order as *content_items*.

        Raises:
            EmbeddingError: If *dimension* is invalid or any item fails.
        """
        self.validate_dimension(dimension)
        if not content_items:
            return []

        results: List[EmbeddingResult] = []
        for idx, item in enumerate(content_items):
            try:
                result = self._embed_item(item, dimension)
                results.append(result)
            except EmbeddingError as exc:
                raise EmbeddingError(
                    f"Batch embedding failed at item {idx}: {exc}",
                    error_type=exc.error_type,
                )
        return results

    def embed_with_multiple_dimensions(
        self, content: ContentItem, dimensions: List[int]
    ) -> Dict[int, EmbeddingResult]:
        """Generate embeddings at multiple Matryoshka dimensions.

        Useful for pre-computing named vectors required by two-stage retrieval.

        Args:
            content: Content item to embed.
            dimensions: List of Matryoshka dimensions to generate.

        Returns:
            Dictionary mapping each dimension to its
            :class:`~multimodal_search.models.EmbeddingResult`.

        Raises:
            EmbeddingError: If any dimension is invalid or an API call fails.

        Example:
            >>> results = service.embed_with_multiple_dimensions(item, [256, 1024])
            >>> len(results[256].vector)
            256
        """
        return {dim: self._embed_item(content, dim) for dim in dimensions}

    def embed_interleaved(
        self, content: ContentItem, dimension: int = DEFAULT_DIMENSION
    ) -> EmbeddingResult:
        """Generate a unified embedding for interleaved multimodal content.

        Builds a multi-part ``google.genai`` request from
        ``content.interleaved_parts``, preserving the order of parts so the
        model captures semantic relationships between modalities.

        Supported part types: ``"text"``, ``"image"``, ``"audio"``,
        ``"video"``, ``"pdf"``.

        Args:
            content: :class:`~multimodal_search.models.ContentItem` with
                ``content_type="interleaved"`` and a populated
                ``interleaved_parts`` list.
            dimension: Matryoshka dimension (128–3072).  Defaults to 756.

        Returns:
            :class:`~multimodal_search.models.EmbeddingResult` with a unified
            vector capturing all parts.

        Raises:
            EmbeddingError: If ``content_type`` is not ``"interleaved"``,
                ``interleaved_parts`` is empty, a part has an unsupported type,
                or the API call fails.
        """
        if content.content_type != "interleaved":
            raise EmbeddingError(
                "embed_interleaved requires content_type='interleaved'",
                error_type="API_ERROR",
            )
        parts = content.interleaved_parts or []
        if not parts:
            raise EmbeddingError(
                "Interleaved content must have at least one part",
                error_type="API_ERROR",
            )
        self.validate_dimension(dimension)

        def _call():
            from google.genai import types

            genai_parts = []
            for idx, part in enumerate(parts):
                if part.content_type == "text":
                    genai_parts.append(str(part.data))
                elif part.content_type in ("image", "audio", "video", "pdf"):
                    mime = part.mime_type or _MIME_FALLBACKS.get(part.content_type, "application/octet-stream")
                    genai_parts.append(
                        types.Part.from_bytes(data=bytes(part.data), mime_type=mime)
                    )
                else:
                    raise EmbeddingError(
                        f"Part {idx} has unsupported type '{part.content_type}' "
                        "for interleaved embedding.",
                        error_type="API_ERROR",
                    )

            response = self._client.models.embed_content(
                model=self.MODEL_NAME,
                contents=genai_parts,
                config=types.EmbedContentConfig(output_dimensionality=dimension),
            )
            return response.embeddings[0].values

        vector = self._execute_with_retry(_call)
        return EmbeddingResult(
            vector=vector,
            dimension=dimension,
            content_type="interleaved",
            model_version=self.MODEL_NAME,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_item(self, item: ContentItem, dimension: int) -> EmbeddingResult:
        """Dispatch a single :class:`~multimodal_search.models.ContentItem` to the correct embed method.

        Args:
            item: Content item to embed.
            dimension: Matryoshka dimension to use.

        Returns:
            :class:`~multimodal_search.models.EmbeddingResult`.

        Raises:
            EmbeddingError: If the content type is unknown.
        """
        ct = item.content_type
        if ct == "text":
            return self.embed_text(str(item.data), dimension)
        elif ct == "image":
            mime = item.mime_type or "image/jpeg"
            return self.embed_image(bytes(item.data), dimension, mime_type=mime)
        elif ct == "audio":
            mime = item.mime_type or "audio/mp3"
            return self.embed_audio(bytes(item.data), dimension, mime_type=mime)
        elif ct == "video":
            mime = item.mime_type or "video/mp4"
            return self.embed_video(bytes(item.data), dimension, mime_type=mime)
        elif ct == "pdf":
            return self.embed_pdf(bytes(item.data), dimension)
        elif ct == "interleaved":
            return self.embed_interleaved(item, dimension)
        else:
            raise EmbeddingError(
                f"Unknown content type: '{ct}'",
                error_type="API_ERROR",
            )

    def _execute_with_retry(self, func, max_retries: int = MAX_RETRIES):
        """Execute *func* with exponential-backoff retry for transient errors.

        Args:
            func: Zero-argument callable to execute.
            max_retries: Maximum number of attempts (default: 3).

        Returns:
            The return value of *func* on success.

        Raises:
            EmbeddingError: After all retries are exhausted, or immediately
                for non-retryable errors (auth failures, permission denied).
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                return func()
            except google_exceptions.ResourceExhausted as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(self.BASE_RETRY_DELAY * (2 ** attempt))
                    continue
                raise EmbeddingError(
                    f"Rate limit exceeded after {max_retries} retries: {exc}",
                    error_type="RATE_LIMIT",
                )
            except (google_exceptions.Unauthenticated, google_exceptions.PermissionDenied) as exc:
                raise EmbeddingError(str(exc), error_type="AUTH_FAILED")
            except google_exceptions.DeadlineExceeded as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(self.BASE_RETRY_DELAY * (2 ** attempt))
                    continue
                raise EmbeddingError(
                    f"Request timeout after {max_retries} retries: {exc}",
                    error_type="NETWORK_ERROR",
                )
            except google_exceptions.GoogleAPIError as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(self.BASE_RETRY_DELAY * (2 ** attempt))
                    continue
                raise EmbeddingError(
                    f"Vertex AI API error after {max_retries} retries: {exc}",
                    error_type="API_ERROR",
                )
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(
                    f"Unexpected error during embedding: {exc}",
                    error_type="API_ERROR",
                )

        raise EmbeddingError(
            f"Failed after {max_retries} retries: {last_error}",
            error_type="API_ERROR",
        )
