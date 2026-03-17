"""
Example: Two-Stage Retrieval
==============================
Demonstrates the two-stage retrieval pattern for balancing search speed
and accuracy:

  Stage 1 – Fast candidate retrieval using a low-dimension named vector
             (e.g. dim_256).  Scans the entire collection quickly.
  Stage 2 – Accurate re-ranking of the top-N candidates using a
             high-dimension named vector (e.g. dim_1024).

Content must be indexed with named vectors at *both* dimensions before
two-stage search can be used.  This example shows how to do that via
the `named_vectors` parameter of `embed_content`.

Required environment variables:
    VERTEX_AI_PROJECT_ID  - Your GCP project ID
    VERTEX_AI_LOCATION    - GCP region (default: us-central1)
    QDRANT_URL            - Qdrant server URL
    QDRANT_API_KEY        - Qdrant API key (optional)
"""

import sys

from multimodal_search.api import MultimodalSearchAPI
from multimodal_search.models import ContentItem, StageConfig
from multimodal_search.exceptions import ValidationError, EmbeddingError, StorageError, SearchError


# Two-stage dimensions – must match what was used during indexing
FIRST_STAGE_DIM = 256    # fast, lower-quality
SECOND_STAGE_DIM = 1024  # slower, higher-quality
FIRST_STAGE_CANDIDATES = 50  # retrieve this many in stage 1
FINAL_RESULTS = 10           # return this many after re-ranking


def _print_results(label: str, response) -> None:
    print(f"\n{label}")
    print(f"  two_stage={response.two_stage}  results={response.total_results}"
          f"  time={response.search_time_ms:.1f} ms")
    meta = response.query_metadata
    if response.two_stage:
        print(f"  stage1_dim={meta.get('first_stage_dimension')}  "
              f"stage2_dim={meta.get('second_stage_dimension')}  "
              f"candidates={meta.get('candidates_retrieved')}")
    for i, r in enumerate(response.results, 1):
        print(f"  [{i}] score={r.score:.4f}  type={r.content_type}  source={r.source_id}")


def main() -> None:
    print("=== Two-Stage Retrieval Demo ===\n")

    # ------------------------------------------------------------------
    # 1. Initialise
    # ------------------------------------------------------------------
    print("Step 1: Initialising API …")
    api = MultimodalSearchAPI.from_env(enable_two_stage=True)
    status = api.initialize_system()
    if not status.initialized:
        print("ERROR: System initialisation failed:")
        for err in status.errors:
            print(f"  - {err}")
        sys.exit(1)
    print(f"  ✓ Ready  two_stage_enabled={status.two_stage_enabled}\n")

    # ------------------------------------------------------------------
    # 2. Index content with named vectors at both dimensions
    # ------------------------------------------------------------------
    print("Step 2: Indexing content with named vectors …")

    documents = [
        ("doc-001", "Introduction to neural networks and deep learning."),
        ("doc-002", "Convolutional neural networks for image recognition."),
        ("doc-003", "Recurrent neural networks and sequence modelling."),
        ("doc-004", "Transformer architecture and attention mechanisms."),
        ("doc-005", "Reinforcement learning and reward optimisation."),
        ("doc-006", "Natural language processing with large language models."),
        ("doc-007", "Graph neural networks for relational data."),
        ("doc-008", "Generative adversarial networks for image synthesis."),
    ]

    for source_id, text in documents:
        item = ContentItem(
            content_type="text",
            data=text,
            source_id=source_id,
        )
        try:
            # Store embeddings at both dimensions as named vectors
            resp = api.embed_content(
                item,
                dimension=FIRST_STAGE_DIM,
                store=True,
                named_vectors=[SECOND_STAGE_DIM],  # also store at 1024
            )
            print(f"  ✓ {source_id} → point_id={resp.point_id}")
        except (ValidationError, EmbeddingError, StorageError) as exc:
            print(f"  ✗ Failed to index {source_id}: {exc}")

    print()

    # ------------------------------------------------------------------
    # 3. Single-stage search (baseline)
    # ------------------------------------------------------------------
    query = ContentItem(content_type="text", data="attention mechanism in transformers")

    print("Step 3: Single-stage search (dim=1024, baseline) …")
    try:
        single_resp = api.search(query, limit=FINAL_RESULTS, dimension=SECOND_STAGE_DIM)
        _print_results("Single-stage results:", single_resp)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Single-stage search failed: {exc}")

    # ------------------------------------------------------------------
    # 4. Two-stage retrieval
    # ------------------------------------------------------------------
    print("\nStep 4: Two-stage retrieval …")
    first_stage = StageConfig(dimension=FIRST_STAGE_DIM, limit=FIRST_STAGE_CANDIDATES)
    second_stage = StageConfig(dimension=SECOND_STAGE_DIM, limit=FINAL_RESULTS)

    try:
        two_stage_resp = api.search_two_stage(
            query=query,
            first_stage_config=first_stage,
            second_stage_config=second_stage,
        )
        _print_results("Two-stage results:", two_stage_resp)
    except (ValidationError, EmbeddingError, SearchError) as exc:
        print(f"  ✗ Two-stage search failed: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
