"""
Example: Multilingual and Cross-Lingual Search
================================================
Demonstrates how Gemini Embedding 2 preserves cross-lingual semantic
similarity so that a query in one language retrieves relevant content
stored in *any* language.

Key points:
  - No language specification is required when embedding; the model
    auto-detects the language.
  - A Spanish query will retrieve semantically similar English documents
    (and vice-versa) because all languages share the same vector space.
  - The `target_languages` filter lets you restrict results to specific
    languages when needed.

Required environment variables:
    VERTEX_AI_PROJECT_ID  - Your GCP project ID
    VERTEX_AI_LOCATION    - GCP region (default: us-central1)
    QDRANT_URL            - Qdrant server URL
    QDRANT_API_KEY        - Qdrant API key (optional)
"""

import sys

from multimodal_search.api import MultimodalSearchAPI
from multimodal_search.models import ContentItem, SearchFilters
from multimodal_search.exceptions import ValidationError, EmbeddingError, SearchError, StorageError


def _print_results(label: str, response) -> None:
    print(f"\n{label}")
    print(f"  results={response.total_results}  time={response.search_time_ms:.1f} ms")
    meta = response.query_metadata
    print(f"  query_language={meta.get('query_language')}  "
          f"target_languages={meta.get('target_languages')}")
    for i, r in enumerate(response.results, 1):
        lang = r.metadata.language or "unknown"
        print(f"  [{i}] score={r.score:.4f}  lang={lang}  source={r.source_id}")


def main() -> None:
    print("=== Multilingual Search Demo ===\n")

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
    # 2. Index multilingual content
    # ------------------------------------------------------------------
    print("Step 2: Indexing multilingual content …")

    multilingual_docs = [
        # English
        ("en-doc-001", "en", "The Eiffel Tower is a famous landmark in Paris, France."),
        ("en-doc-002", "en", "Machine learning enables computers to learn from data."),
        ("en-doc-003", "en", "Climate change is affecting global weather patterns."),
        # Spanish
        ("es-doc-001", "es", "La Torre Eiffel es un famoso monumento en París, Francia."),
        ("es-doc-002", "es", "El aprendizaje automático permite a las computadoras aprender de datos."),
        # French
        ("fr-doc-001", "fr", "La Tour Eiffel est un monument célèbre à Paris, en France."),
        ("fr-doc-002", "fr", "L'apprentissage automatique permet aux ordinateurs d'apprendre à partir de données."),
        # German
        ("de-doc-001", "de", "Der Eiffelturm ist ein berühmtes Wahrzeichen in Paris, Frankreich."),
        # Japanese
        ("ja-doc-001", "ja", "エッフェル塔はフランスのパリにある有名なランドマークです。"),
    ]

    for source_id, lang, text in multilingual_docs:
        item = ContentItem(
            content_type="text",
            data=text,
            source_id=source_id,
            metadata={"language": lang},
        )
        try:
            resp = api.embed_content(item, store=True)
            print(f"  ✓ [{lang}] {source_id} → point_id={resp.point_id}")
        except (ValidationError, EmbeddingError, StorageError) as exc:
            print(f"  ✗ Failed to index {source_id}: {exc}")

    print()

    # ------------------------------------------------------------------
    # 3. English query → retrieve all languages
    # ------------------------------------------------------------------
    print("Step 3: English query → all languages …")
    try:
        from multimodal_search.search_engine import SearchEngine
        # Use the search_engine directly for multilingual search
        response = api._search_engine.search_multilingual(
            query_text="Eiffel Tower Paris landmark",
            query_language="en",
            target_languages=None,  # no language restriction
            limit=5,
        )
        _print_results("English query → all languages:", response)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Search failed: {exc}")

    print()

    # ------------------------------------------------------------------
    # 4. Spanish query → retrieve English results only
    # ------------------------------------------------------------------
    print("Step 4: Spanish query → English results only …")
    try:
        response = api._search_engine.search_multilingual(
            query_text="Torre Eiffel monumento famoso",
            query_language="es",
            target_languages=["en"],
            limit=5,
        )
        _print_results("Spanish query → English only:", response)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Search failed: {exc}")

    print()

    # ------------------------------------------------------------------
    # 5. Japanese query → retrieve all languages
    # ------------------------------------------------------------------
    print("Step 5: Japanese query → all languages …")
    try:
        response = api._search_engine.search_multilingual(
            query_text="機械学習 データ",
            query_language="ja",
            target_languages=None,
            limit=5,
        )
        _print_results("Japanese query → all languages:", response)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Search failed: {exc}")

    print()

    # ------------------------------------------------------------------
    # 6. Cross-lingual cross-modal: text query → filter by language via API
    # ------------------------------------------------------------------
    print("Step 6: Cross-lingual search via API (French results only) …")
    try:
        query_item = ContentItem(content_type="text", data="Eiffel Tower Paris")
        filters = SearchFilters(languages=["fr"])
        response = api.search(query_item, limit=5, filters=filters)
        print(f"  Found {response.total_results} French result(s)")
        for i, r in enumerate(response.results, 1):
            lang = r.metadata.language or "unknown"
            print(f"  [{i}] score={r.score:.4f}  lang={lang}  source={r.source_id}")
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Search failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
