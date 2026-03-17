"""Content validation and preparation for embedding.

The :class:`ContentProcessor` is the first line of defence before any
Vertex AI API call is made.  It validates that content meets the format
and size constraints required by the embedding model, returning
descriptive :class:`~multimodal_search.models.ValidationResult` objects
rather than raising exceptions so callers can decide how to handle
failures.
"""

import io
import struct
from typing import List

from multimodal_search.models import ContentItem, InterleavedPart, ValidationResult


class ContentProcessor:
    """Validates and prepares content for embedding.

    All public ``validate_*`` methods return a
    :class:`~multimodal_search.models.ValidationResult` rather than
    raising exceptions, allowing callers to inspect failures before
    deciding whether to abort or continue.

    Validation is intentionally performed *before* any Vertex AI API
    call so that invalid content is rejected early without consuming
    quota.

    Supported modalities and their constraints:

    * **Text** – must be non-empty; estimated token count ≤ 8 192.
    * **Image** – PNG or JPEG only; max 6 images per batch.
    * **Audio** – MP3 or WAV only; duration ≤ 80 s (when detectable).
    * **Video** – MP4 or MOV only; duration ≤ 128 s (when detectable).
    * **PDF** – must start with ``%PDF-``; page count ≤ 6.

    Example:
        >>> processor = ContentProcessor()
        >>> result = processor.validate_text("Hello, world!")
        >>> result.valid
        True
        >>> result = processor.validate_text("")
        >>> result.valid
        False
        >>> result.error_type
        'EMPTY_CONTENT'
    """

    # Validation constants
    MAX_TEXT_TOKENS = 8192
    MAX_AUDIO_DURATION = 80  # seconds
    MAX_VIDEO_DURATION = 120  # seconds (per Gemini Embedding 2 blog post)
    MAX_PDF_PAGES = 6
    MAX_IMAGES_PER_BATCH = 6

    # Supported formats
    SUPPORTED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}
    SUPPORTED_AUDIO_MIME_TYPES = {"audio/mp3", "audio/mpeg", "audio/wav", "audio/x-wav"}
    SUPPORTED_VIDEO_MIME_TYPES = {"video/mp4", "video/quicktime"}

    def validate_text(self, text: str) -> ValidationResult:
        """
        Validates text content for embedding.

        Args:
            text: Text string to validate

        Returns:
            ValidationResult indicating if text is valid
        """
        if not text or not text.strip():
            return ValidationResult(
                valid=False,
                error_type="EMPTY_CONTENT",
                error_message="Text content is empty or contains only whitespace",
            )

        # Approximate token count (rough estimate: 1 token ≈ 4 characters)
        # This is a simple heuristic; actual tokenization would require the model's tokenizer
        estimated_tokens = len(text) // 4

        if estimated_tokens > self.MAX_TEXT_TOKENS:
            return ValidationResult(
                valid=False,
                error_type="SIZE_EXCEEDED",
                error_message=f"Text exceeds maximum token limit of {self.MAX_TEXT_TOKENS} tokens (estimated: {estimated_tokens} tokens)",
            )

        return ValidationResult(valid=True)

    def validate_image(self, image_data: bytes, mime_type: str) -> ValidationResult:
        """
        Validates image format (PNG/JPEG) and size.

        Args:
            image_data: Binary image data
            mime_type: MIME type of the image

        Returns:
            ValidationResult indicating if image is valid
        """
        if not image_data:
            return ValidationResult(
                valid=False,
                error_type="EMPTY_CONTENT",
                error_message="Image data is empty",
            )

        # Validate MIME type
        if mime_type.lower() not in self.SUPPORTED_IMAGE_MIME_TYPES:
            return ValidationResult(
                valid=False,
                error_type="INVALID_FORMAT",
                error_message=f"Image format '{mime_type}' not supported. Supported formats: PNG, JPEG",
            )

        # Validate actual image format matches MIME type
        actual_format = self._detect_image_format(image_data)
        if actual_format is None:
            return ValidationResult(
                valid=False,
                error_type="INVALID_FORMAT",
                error_message="Unable to detect image format. File may be corrupted",
            )

        # Check if detected format matches declared MIME type
        if not self._mime_matches_format(mime_type, actual_format):
            return ValidationResult(
                valid=False,
                error_type="MIME_TYPE_MISMATCH",
                error_message=f"MIME type '{mime_type}' does not match actual image format '{actual_format}'",
            )

        return ValidationResult(valid=True)

    def validate_audio(self, audio_data: bytes, mime_type: str) -> ValidationResult:
        """
        Validates audio format (MP3/WAV) and duration.

        Args:
            audio_data: Binary audio data
            mime_type: MIME type of the audio

        Returns:
            ValidationResult indicating if audio is valid
        """
        if not audio_data:
            return ValidationResult(
                valid=False,
                error_type="EMPTY_CONTENT",
                error_message="Audio data is empty",
            )

        # Validate MIME type
        if mime_type.lower() not in self.SUPPORTED_AUDIO_MIME_TYPES:
            return ValidationResult(
                valid=False,
                error_type="INVALID_FORMAT",
                error_message=f"Audio format '{mime_type}' not supported. Supported formats: MP3, WAV",
            )

        # Validate actual audio format
        actual_format = self._detect_audio_format(audio_data)
        if actual_format is None:
            return ValidationResult(
                valid=False,
                error_type="INVALID_FORMAT",
                error_message="Unable to detect audio format. File may be corrupted",
            )

        # Check duration (basic validation)
        duration = self._estimate_audio_duration(audio_data, actual_format)
        if duration is not None and duration > self.MAX_AUDIO_DURATION:
            return ValidationResult(
                valid=False,
                error_type="DURATION_EXCEEDED",
                error_message=f"Audio duration ({duration:.1f}s) exceeds maximum of {self.MAX_AUDIO_DURATION}s",
            )

        return ValidationResult(valid=True)

    def validate_video(self, video_data: bytes, mime_type: str) -> ValidationResult:
        """
        Validates video format (MP4/MOV) and duration.

        Args:
            video_data: Binary video data
            mime_type: MIME type of the video

        Returns:
            ValidationResult indicating if video is valid
        """
        if not video_data:
            return ValidationResult(
                valid=False,
                error_type="EMPTY_CONTENT",
                error_message="Video data is empty",
            )

        # Validate MIME type
        if mime_type.lower() not in self.SUPPORTED_VIDEO_MIME_TYPES:
            return ValidationResult(
                valid=False,
                error_type="INVALID_FORMAT",
                error_message=f"Video format '{mime_type}' not supported. Supported formats: MP4, MOV",
            )

        # Validate actual video format
        actual_format = self._detect_video_format(video_data)
        if actual_format is None:
            return ValidationResult(
                valid=False,
                error_type="INVALID_FORMAT",
                error_message="Unable to detect video format. File may be corrupted",
            )

        # Check duration (basic validation)
        duration = self._estimate_video_duration(video_data)
        if duration is not None and duration > self.MAX_VIDEO_DURATION:
            return ValidationResult(
                valid=False,
                error_type="DURATION_EXCEEDED",
                error_message=f"Video duration ({duration:.1f}s) exceeds maximum of {self.MAX_VIDEO_DURATION}s",
            )

        return ValidationResult(valid=True)

    def validate_pdf(self, pdf_data: bytes) -> ValidationResult:
        """
        Validates PDF format and page count.

        Args:
            pdf_data: Binary PDF data

        Returns:
            ValidationResult indicating if PDF is valid
        """
        if not pdf_data:
            return ValidationResult(
                valid=False,
                error_type="EMPTY_CONTENT",
                error_message="PDF data is empty",
            )

        # Check PDF signature
        if not pdf_data.startswith(b"%PDF-"):
            return ValidationResult(
                valid=False,
                error_type="INVALID_FORMAT",
                error_message="File is not a valid PDF (missing PDF signature)",
            )

        # Count pages
        page_count = self._count_pdf_pages(pdf_data)
        if page_count is None:
            return ValidationResult(
                valid=False,
                error_type="INVALID_FORMAT",
                error_message="Unable to determine PDF page count. File may be corrupted",
            )

        if page_count > self.MAX_PDF_PAGES:
            return ValidationResult(
                valid=False,
                error_type="PAGE_LIMIT_EXCEEDED",
                error_message=f"PDF has {page_count} pages, exceeds maximum of {self.MAX_PDF_PAGES} pages",
            )

        return ValidationResult(valid=True)

    def validate_batch(self, content_items: List[ContentItem]) -> List[ValidationResult]:
        """
        Validates all items in a batch before processing.

        Args:
            content_items: List of content items to validate

        Returns:
            List of ValidationResult for each item
        """
        if not content_items:
            return [
                ValidationResult(
                    valid=False,
                    error_type="EMPTY_CONTENT",
                    error_message="Batch is empty",
                )
            ]

        results = []
        image_count = 0

        for item in content_items:
            # Count images for batch limit
            if item.content_type == "image":
                image_count += 1

            # Validate each item based on its type
            if item.content_type == "text":
                result = self.validate_text(str(item.data))
            elif item.content_type == "image":
                result = self.validate_image(bytes(item.data), item.mime_type or "")
            elif item.content_type == "audio":
                result = self.validate_audio(bytes(item.data), item.mime_type or "")
            elif item.content_type == "video":
                result = self.validate_video(bytes(item.data), item.mime_type or "")
            elif item.content_type == "pdf":
                result = self.validate_pdf(bytes(item.data))
            else:
                result = ValidationResult(
                    valid=False,
                    error_type="INVALID_FORMAT",
                    error_message=f"Unknown content type: {item.content_type}",
                )

            results.append(result)

        # Check image batch limit
        if image_count > self.MAX_IMAGES_PER_BATCH:
            # Add warning to all image results
            for i, item in enumerate(content_items):
                if item.content_type == "image":
                    results[i].warnings.append(
                        f"Batch contains {image_count} images, exceeds maximum of {self.MAX_IMAGES_PER_BATCH}"
                    )

        return results

    def validate_interleaved(self, content: ContentItem) -> ValidationResult:
        """
        Validates an interleaved multimodal ContentItem.

        Checks that:
        - content_type is "interleaved"
        - interleaved_parts is a non-empty list
        - Each part passes its own modality validation

        Args:
            content: ContentItem with content_type="interleaved"

        Returns:
            ValidationResult — valid only if all parts pass validation
        """
        if content.content_type != "interleaved":
            return ValidationResult(
                valid=False,
                error_type="INVALID_FORMAT",
                error_message="validate_interleaved requires content_type='interleaved'",
            )

        parts = content.interleaved_parts
        if not parts:
            return ValidationResult(
                valid=False,
                error_type="EMPTY_CONTENT",
                error_message="Interleaved content must have at least one part",
            )

        for idx, part in enumerate(parts):
            if part.content_type == "text":
                result = self.validate_text(str(part.data))
            elif part.content_type == "image":
                result = self.validate_image(bytes(part.data), part.mime_type or "")
            elif part.content_type == "audio":
                result = self.validate_audio(bytes(part.data), part.mime_type or "")
            elif part.content_type == "video":
                result = self.validate_video(bytes(part.data), part.mime_type or "")
            elif part.content_type == "pdf":
                result = self.validate_pdf(bytes(part.data))
            else:
                result = ValidationResult(
                    valid=False,
                    error_type="INVALID_FORMAT",
                    error_message=f"Unknown content type in part {idx}: {part.content_type}",
                )

            if not result.valid:
                return ValidationResult(
                    valid=False,
                    error_type=result.error_type,
                    error_message=f"Part {idx} ({part.content_type}) failed validation: {result.error_message}",
                )

        return ValidationResult(valid=True)

    def prepare_for_embedding(self, content: ContentItem) -> ContentItem:
        """
        Prepares validated content for Vertex AI API call.

        Args:
            content: Content item to prepare

        Returns:
            Prepared ContentItem ready for embedding
        """
        # For now, just return the content as-is
        # Future enhancements could include:
        # - Text normalization
        # - Image resizing/optimization
        # - Format conversions
        return content

    # Helper methods for format detection

    def _detect_image_format(self, data: bytes) -> str | None:
        """Detect image format from binary magic bytes.

        Args:
            data: Raw image bytes.

        Returns:
            ``"png"`` or ``"jpeg"`` if the magic bytes match a known
            format, otherwise ``None``.
        """
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        elif data.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        return None

    def _mime_matches_format(self, mime_type: str, format: str) -> bool:
        """Check whether a declared MIME type is consistent with a detected format.

        Args:
            mime_type: Declared MIME type string (e.g. ``"image/jpeg"``).
            format: Detected format string — ``"png"`` or ``"jpeg"``.

        Returns:
            ``True`` if the MIME type is consistent with the detected
            format, ``False`` otherwise.
        """
        mime_lower = mime_type.lower()
        if format == "png":
            return "png" in mime_lower
        elif format == "jpeg":
            return "jpeg" in mime_lower or "jpg" in mime_lower
        return False

    def _detect_audio_format(self, data: bytes) -> str | None:
        """Detect audio format from binary magic bytes.

        Args:
            data: Raw audio bytes.

        Returns:
            ``"mp3"`` for ID3-tagged or sync-word MP3 data, ``"wav"``
            for RIFF/WAVE data, or ``None`` if the format is unrecognised.
        """
        if data.startswith(b"ID3") or data.startswith(b"\xff\xfb") or data.startswith(b"\xff\xf3"):
            return "mp3"
        elif data.startswith(b"RIFF") and b"WAVE" in data[:12]:
            return "wav"
        return None

    def _estimate_audio_duration(self, data: bytes, format: str) -> float | None:
        """Estimate audio duration in seconds from raw bytes.

        Only WAV files are supported via header parsing.  MP3 and other
        formats return ``None`` because accurate duration extraction
        requires a specialised library (e.g. *mutagen* or *pydub*).

        Args:
            data: Raw audio bytes.
            format: Detected format string — ``"mp3"`` or ``"wav"``.

        Returns:
            Estimated duration in seconds, or ``None`` if the duration
            cannot be determined.
        """
        # This is a simplified estimation
        # For production, use a proper audio library like pydub or mutagen
        if format == "wav":
            try:
                # WAV header parsing
                if len(data) < 44:
                    return None
                # Read sample rate (bytes 24-27)
                sample_rate = struct.unpack("<I", data[24:28])[0]
                # Read byte rate (bytes 28-31)
                byte_rate = struct.unpack("<I", data[28:32])[0]
                if byte_rate > 0:
                    duration = (len(data) - 44) / byte_rate
                    return duration
            except Exception:
                pass
        # For MP3 and other formats, return None (requires specialized parsing)
        return None

    def _detect_video_format(self, data: bytes) -> str | None:
        """Detect video format from the MP4/MOV container header.

        Both MP4 and MOV use the ISO Base Media File Format (ISOBMFF)
        container.  The format is identified by inspecting the ``ftyp``
        box in the first 12 bytes.

        Args:
            data: Raw video bytes.

        Returns:
            ``"mp4"`` or ``"mov"`` if the header matches, otherwise
            ``None``.
        """
        # MP4 signature
        if len(data) >= 12:
            if b"ftyp" in data[4:12]:
                return "mp4"
            # MOV is also MP4-based
            if b"ftypqt" in data[4:12] or b"moov" in data[:100]:
                return "mov"
        return None

    def _estimate_video_duration(self, data: bytes) -> float | None:
        """Estimate video duration in seconds.

        Proper duration extraction requires parsing the MP4/MOV container
        (``mvhd`` box) or using an external library such as *ffmpeg*.
        This placeholder always returns ``None``, which causes the
        duration validation step to be skipped.

        Args:
            data: Raw video bytes.

        Returns:
            Always ``None`` in the current implementation.
        """
        # This is a placeholder - proper video duration extraction requires
        # parsing the MP4/MOV container format or using a library like ffmpeg
        # For now, return None to skip duration validation
        return None

    def _count_pdf_pages(self, data: bytes) -> int | None:
        """Count the number of pages in a PDF document.

        Uses a heuristic approach: first looks for a ``/Count`` entry in
        the ``/Pages`` dictionary, then falls back to counting
        ``/Type /Page`` occurrences.  This works for most standard PDFs
        but may be inaccurate for heavily compressed or encrypted files.

        Args:
            data: Raw PDF bytes.

        Returns:
            Page count as an integer, or ``None`` if the count cannot be
            determined (e.g. the file is corrupted or encrypted).
        """
        try:
            # Simple page counting by looking for /Type /Page entries
            # This is a basic heuristic and may not work for all PDFs
            text = data.decode("latin-1", errors="ignore")
            # Count occurrences of /Type /Page or /Type/Page
            count = text.count("/Type /Page")
            count += text.count("/Type/Page")
            # Also check for /Count in Pages object
            import re

            count_match = re.search(r"/Type\s*/Pages.*?/Count\s+(\d+)", text, re.DOTALL)
            if count_match:
                return int(count_match.group(1))
            return count if count > 0 else None
        except Exception:
            return None
