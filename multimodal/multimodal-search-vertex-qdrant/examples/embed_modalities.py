"""
Example: Embedding Different Modalities
========================================
Demonstrates how to embed text, image, audio, video, and PDF content
using the MultimodalSearchAPI. Each modality is embedded and stored in
Qdrant so it can later be retrieved via semantic search.

Required environment variables:
    VERTEX_AI_PROJECT_ID  - Your GCP project ID
    VERTEX_AI_LOCATION    - GCP region (default: us-central1)
    QDRANT_URL            - Qdrant server URL (e.g. http://localhost:6333)
    QDRANT_API_KEY        - Qdrant API key (optional for local instances)
"""

import os
import sys

from multimodal_search.api import MultimodalSearchAPI
from multimodal_search.models import ContentItem, VertexAIConfig, QdrantConfig
from multimodal_search.exceptions import ValidationError, EmbeddingError, StorageError


# ---------------------------------------------------------------------------
# Helper: load a file as bytes (falls back to a tiny synthetic payload)
# ---------------------------------------------------------------------------

def _load_or_stub(path: str, stub: bytes) -> bytes:
    """Return file bytes if the path exists, otherwise return the stub."""
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return fh.read()
    print(f"  [info] '{path}' not found – using synthetic stub for demo purposes")
    return stub


# ---------------------------------------------------------------------------
# Minimal valid stubs so the script can run without real media files
# ---------------------------------------------------------------------------

# 1×1 white JPEG (smallest valid JPEG)
_STUB_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x0a\xff\xd9"
)

# Minimal WAV header (44 bytes, 0 audio samples)
_STUB_WAV = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)

# Minimal MP4 ftyp box
_STUB_MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"

# Minimal PDF (1 page, no content)
_STUB_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f\n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n9\n%%EOF"
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Initialise the API from environment variables
    # ------------------------------------------------------------------
    print("=== Multimodal Embedding Demo ===\n")
    print("Step 1: Initialising API from environment variables …")

    api = MultimodalSearchAPI.from_env()
    status = api.initialize_system()

    if not status.initialized:
        print("ERROR: System initialisation failed:")
        for err in status.errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"  ✓ System ready (default dimension: {status.default_dimension})\n")

    # ------------------------------------------------------------------
    # 2. Embed text
    # ------------------------------------------------------------------
    print("Step 2: Embedding text content …")
    text_item = ContentItem(
        content_type="text",
        data="The quick brown fox jumps over the lazy dog.",
        source_id="text-demo-001",
        metadata={"language": "en", "category": "demo"},
    )
    try:
        text_resp = api.embed_content(text_item, dimension=756, store=True)
        print(f"  ✓ Text embedded  | dim={text_resp.dimension} | point_id={text_resp.point_id}")
    except (ValidationError, EmbeddingError, StorageError) as exc:
        print(f"  ✗ Text embedding failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Embed image
    # ------------------------------------------------------------------
    print("\nStep 3: Embedding image content …")
    image_bytes = _load_or_stub("assets/sample.jpg", _STUB_JPEG)
    image_item = ContentItem(
        content_type="image",
        data=image_bytes,
        mime_type="image/jpeg",
        source_id="image-demo-001",
        metadata={"tags": ["demo", "sample"]},
    )
    try:
        img_resp = api.embed_content(image_item, dimension=756, store=True)
        print(f"  ✓ Image embedded | dim={img_resp.dimension} | point_id={img_resp.point_id}")
    except (ValidationError, EmbeddingError, StorageError) as exc:
        print(f"  ✗ Image embedding failed: {exc}")

    # ------------------------------------------------------------------
    # 4. Embed audio
    # ------------------------------------------------------------------
    print("\nStep 4: Embedding audio content …")
    audio_bytes = _load_or_stub("assets/sample.wav", _STUB_WAV)
    audio_item = ContentItem(
        content_type="audio",
        data=audio_bytes,
        mime_type="audio/wav",
        source_id="audio-demo-001",
    )
    try:
        aud_resp = api.embed_content(audio_item, dimension=756, store=True)
        print(f"  ✓ Audio embedded | dim={aud_resp.dimension} | point_id={aud_resp.point_id}")
    except (ValidationError, EmbeddingError, StorageError) as exc:
        print(f"  ✗ Audio embedding failed: {exc}")

    # ------------------------------------------------------------------
    # 5. Embed video
    # ------------------------------------------------------------------
    print("\nStep 5: Embedding video content …")
    video_bytes = _load_or_stub("assets/sample.mp4", _STUB_MP4)
    video_item = ContentItem(
        content_type="video",
        data=video_bytes,
        mime_type="video/mp4",
        source_id="video-demo-001",
    )
    try:
        vid_resp = api.embed_content(video_item, dimension=756, store=True)
        print(f"  ✓ Video embedded | dim={vid_resp.dimension} | point_id={vid_resp.point_id}")
    except (ValidationError, EmbeddingError, StorageError) as exc:
        print(f"  ✗ Video embedding failed: {exc}")

    # ------------------------------------------------------------------
    # 6. Embed PDF
    # ------------------------------------------------------------------
    print("\nStep 6: Embedding PDF content …")
    pdf_bytes = _load_or_stub("assets/sample.pdf", _STUB_PDF)
    pdf_item = ContentItem(
        content_type="pdf",
        data=pdf_bytes,
        source_id="pdf-demo-001",
        metadata={"title": "Sample Document"},
    )
    try:
        pdf_resp = api.embed_content(pdf_item, dimension=756, store=True)
        print(f"  ✓ PDF embedded   | dim={pdf_resp.dimension} | point_id={pdf_resp.point_id}")
    except (ValidationError, EmbeddingError, StorageError) as exc:
        print(f"  ✗ PDF embedding failed: {exc}")

    print("\nDone. All modalities processed.")


if __name__ == "__main__":
    main()
