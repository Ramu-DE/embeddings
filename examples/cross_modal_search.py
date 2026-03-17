"""
Example: Cross-Modal Search
=============================
Demonstrates how to query with one modality and retrieve semantically
relevant results from *different* modalities.  For example, a text query
like "sunset over the ocean" can retrieve matching images, videos, and
audio clips stored in Qdrant.

All modalities share a unified vector space via Gemini Embedding 2, so
cosine similarity is meaningful across content types.

Required environment variables:
    VERTEX_AI_PROJECT_ID  - Your GCP project ID
    VERTEX_AI_LOCATION    - GCP region (default: us-central1)
    QDRANT_URL            - Qdrant server URL
    QDRANT_API_KEY        - Qdrant API key (optional)
"""

import sys

from multimodal_search.api import MultimodalSearchAPI
from multimodal_search.models import ContentItem, SearchFilters
from multimodal_search.exceptions import ValidationError, EmbeddingError, SearchError


def _print_results(response) -> None:
    """Pretty-print a SearchResponse."""
    print(f"  Found {response.total_results} result(s) in {response.search_time_ms:.1f} ms")
    for i, r in enumerate(response.results, 1):
        print(f"  [{i}] score={r.score:.4f}  type={r.content_type}  source={r.source_id}")


def main() -> None:
    print("=== Cross-Modal Search Demo ===\n")

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
    # 2. Index sample content across modalities
    #    (skip if your Qdrant collection is already populated)
    # ------------------------------------------------------------------
    print("Step 2: Indexing sample content …")

    samples = [
        ContentItem(
            content_type="text",
            data="A beautiful sunset over the Pacific Ocean with orange and pink hues.",
            source_id="text-sunset-001",
        ),
        ContentItem(
            content_type="text",
            data="Machine learning algorithms for image classification.",
            source_id="text-ml-001",
        ),
        ContentItem(
            content_type="text",
            data="The sound of ocean waves crashing on a sandy beach.",
            source_id="text-ocean-sound-001",
        ),
    ]

    for item in samples:
        try:
            resp = api.embed_content(item, store=True)
            print(f"  ✓ Indexed {item.source_id} → point_id={resp.point_id}")
        except Exception as exc:
            print(f"  ✗ Failed to index {item.source_id}: {exc}")

    print()

    # ------------------------------------------------------------------
    # 3. Text query → retrieve ALL modalities (cross-modal)
    # ------------------------------------------------------------------
    print("Step 3: Text query → all modalities …")
    text_query = ContentItem(
        content_type="text",
        data="sunset ocean waves",
    )
    try:
        response = api.search(text_query, limit=5)
        _print_results(response)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Search failed: {exc}")

    print()

    # ------------------------------------------------------------------
    # 4. Text query → retrieve only images and videos
    # ------------------------------------------------------------------
    print("Step 4: Text query → images and videos only …")
    try:
        filters = SearchFilters(content_types=["image", "video"])
        response = api.search(text_query, limit=5, filters=filters)
        _print_results(response)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Search failed: {exc}")

    print()

    # ------------------------------------------------------------------
    # 5. Text query → retrieve only text documents
    # ------------------------------------------------------------------
    print("Step 5: Text query → text documents only …")
    try:
        filters = SearchFilters(content_types=["text"])
        response = api.search(text_query, limit=5, filters=filters)
        _print_results(response)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Search failed: {exc}")

    print()

    # ------------------------------------------------------------------
    # 6. Apply a similarity score threshold
    # ------------------------------------------------------------------
    print("Step 6: Text query with score threshold (≥ 0.7) …")
    try:
        response = api.search(text_query, limit=10, score_threshold=0.7)
        _print_results(response)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Search failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
