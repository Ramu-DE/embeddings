"""
Example: Interleaved Multimodal Input
=======================================
Demonstrates how to embed content that combines *multiple modalities in a
single request* (interleaved multimodal input).  Gemini Embedding 2 can
process text and image parts together, generating a unified embedding that
captures the semantic relationship between them.

Use cases:
  - Index a product image together with its description so that either
    the visual or textual aspect can be used to retrieve it.
  - Search with an image + caption to find the most relevant documents.

Required environment variables:
    VERTEX_AI_PROJECT_ID  - Your GCP project ID
    VERTEX_AI_LOCATION    - GCP region (default: us-central1)
    QDRANT_URL            - Qdrant server URL
    QDRANT_API_KEY        - Qdrant API key (optional)
"""

import sys

from multimodal_search.api import MultimodalSearchAPI
from multimodal_search.models import ContentItem, InterleavedPart
from multimodal_search.exceptions import ValidationError, EmbeddingError, StorageError, SearchError


# ---------------------------------------------------------------------------
# Minimal 1×1 white JPEG stub (used when no real image file is available)
# ---------------------------------------------------------------------------
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


def _load_image(path: str) -> bytes:
    """Load image bytes from disk, falling back to the JPEG stub."""
    import os
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return fh.read()
    print(f"  [info] '{path}' not found – using synthetic JPEG stub")
    return _STUB_JPEG


def _print_results(label: str, response) -> None:
    print(f"\n{label}")
    print(f"  results={response.total_results}  time={response.search_time_ms:.1f} ms")
    for i, r in enumerate(response.results, 1):
        print(f"  [{i}] score={r.score:.4f}  type={r.content_type}  source={r.source_id}")


def main() -> None:
    print("=== Interleaved Multimodal Demo ===\n")

    # ------------------------------------------------------------------
    # 1. Initialise
    # ------------------------------------------------------------------
    print("Step 1: Initialising API …")
    api = MultimodalSearchAPI.from_env()
    status = api.initialize_system()
    if not status.initialized:
        print("ERROR: System initialisation failed:")
        for err in status.errors:
            print(f"  - {err}")
        sys.exit(1)
    print("  ✓ Ready\n")

    # ------------------------------------------------------------------
    # 2. Index interleaved content (image + caption)
    # ------------------------------------------------------------------
    print("Step 2: Indexing interleaved content (image + caption) …")

    product_image = _load_image("assets/product.jpg")

    # Build an interleaved ContentItem with an image part and a text part
    interleaved_item = ContentItem(
        content_type="interleaved",
        source_id="product-001",
        metadata={"category": "electronics", "brand": "Acme"},
        interleaved_parts=[
            InterleavedPart(
                content_type="image",
                data=product_image,
                mime_type="image/jpeg",
            ),
            InterleavedPart(
                content_type="text",
                data="Wireless noise-cancelling headphones with 30-hour battery life.",
            ),
        ],
    )

    try:
        resp = api.embed_content(interleaved_item, dimension=756, store=True)
        print(f"  ✓ Interleaved item indexed | dim={resp.dimension} | point_id={resp.point_id}")
    except (ValidationError, EmbeddingError, StorageError) as exc:
        print(f"  ✗ Failed to index interleaved item: {exc}")

    # Index a few plain-text items for comparison
    plain_items = [
        ("text-001", "Bluetooth speaker with deep bass and waterproof design."),
        ("text-002", "Over-ear headphones with active noise cancellation."),
        ("text-003", "True wireless earbuds with long battery life."),
    ]
    for source_id, text in plain_items:
        item = ContentItem(content_type="text", data=text, source_id=source_id)
        try:
            resp = api.embed_content(item, store=True)
            print(f"  ✓ {source_id} indexed → point_id={resp.point_id}")
        except Exception as exc:
            print(f"  ✗ {source_id} failed: {exc}")

    print()

    # ------------------------------------------------------------------
    # 3. Search with a plain text query
    # ------------------------------------------------------------------
    print("Step 3: Text query → retrieve all content …")
    text_query = ContentItem(
        content_type="text",
        data="noise cancelling headphones battery",
    )
    try:
        response = api.search(text_query, limit=5)
        _print_results("Text query results:", response)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Search failed: {exc}")

    print()

    # ------------------------------------------------------------------
    # 4. Search with an interleaved query (image + text)
    # ------------------------------------------------------------------
    print("Step 4: Interleaved query (image + text) → retrieve all content …")
    query_image = _load_image("assets/query_headphones.jpg")

    interleaved_query = ContentItem(
        content_type="interleaved",
        interleaved_parts=[
            InterleavedPart(
                content_type="image",
                data=query_image,
                mime_type="image/jpeg",
            ),
            InterleavedPart(
                content_type="text",
                data="headphones audio",
            ),
        ],
    )

    try:
        response = api.search(interleaved_query, limit=5)
        _print_results("Interleaved query results:", response)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Search failed: {exc}")

    print()

    # ------------------------------------------------------------------
    # 5. Validate interleaved content before embedding
    # ------------------------------------------------------------------
    print("Step 5: Validating interleaved content …")
    from multimodal_search.content_processor import ContentProcessor

    processor = ContentProcessor()

    # Valid interleaved item
    valid_result = processor.validate_interleaved(interleaved_item)
    print(f"  Valid item → valid={valid_result.valid}")

    # Invalid: wrong content_type
    bad_item = ContentItem(content_type="text", data="hello")
    bad_result = processor.validate_interleaved(bad_item)
    print(f"  Wrong type → valid={bad_result.valid}  error={bad_result.error_type}")

    # Invalid: empty parts list
    empty_item = ContentItem(content_type="interleaved", interleaved_parts=[])
    empty_result = processor.validate_interleaved(empty_item)
    print(f"  Empty parts → valid={empty_result.valid}  error={empty_result.error_type}")

    print("\nDone.")


if __name__ == "__main__":
    main()
