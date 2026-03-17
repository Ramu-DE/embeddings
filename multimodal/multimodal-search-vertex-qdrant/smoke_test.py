"""End-to-end smoke test for the multimodal search system.

Tests the complete flow:
  1. Load config from .env
  2. Initialize system (Vertex AI + Qdrant)
  3. Embed text and store in Qdrant
  4. Embed image and store in Qdrant
  5. Single-stage search
  6. Two-stage retrieval
  7. Cross-modal search (text query → image results)
  8. Batch embedding
"""

import sys
import os

# Load .env before importing anything else
from dotenv import load_dotenv
load_dotenv()

from multimodal_search.api import MultimodalSearchAPI
from multimodal_search.models import ContentItem, StageConfig, SearchFilters, InterleavedPart

PASS = "✓"
FAIL = "✗"
results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  {status} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((label, condition))
    return condition


# ---------------------------------------------------------------------------
# Minimal 1×1 white JPEG stub
# ---------------------------------------------------------------------------
STUB_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x0a\xff\xd9"
)


def main():
    print("=" * 60)
    print("  Multimodal Search — End-to-End Smoke Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Init
    # ------------------------------------------------------------------
    print("\n[1] System initialisation")
    try:
        api = MultimodalSearchAPI.from_env()
        status = api.initialize_system()
        check("API constructed", True)
        check("Vertex AI connected", status.vertex_ai_connected)
        check("Qdrant connected", status.qdrant_connected)
        check("Collection ready", status.collection_ready)
        if not status.initialized:
            print("\n  FATAL: system not ready — aborting.")
            for e in status.errors:
                print(f"    {e}")
            sys.exit(1)
    except Exception as exc:
        check("System init", False, str(exc))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Embed text
    # ------------------------------------------------------------------
    print("\n[2] Text embedding + storage")
    text_point_id = None
    try:
        item = ContentItem(
            content_type="text",
            data="A golden retriever playing fetch on a sunny beach.",
            source_id="smoke-text-001",
        )
        resp = api.embed_content(item, dimension=756, store=True)
        text_point_id = resp.point_id
        check("Text embedded", len(resp.vector) == 756, f"dim={resp.dimension}")
        check("Text stored in Qdrant", resp.point_id is not None, f"point_id={resp.point_id}")
    except Exception as exc:
        check("Text embedding", False, str(exc))

    # ------------------------------------------------------------------
    # 3. Embed image
    # ------------------------------------------------------------------
    print("\n[3] Image embedding + storage")
    image_point_id = None
    try:
        item = ContentItem(
            content_type="image",
            data=STUB_JPEG,
            mime_type="image/jpeg",
            source_id="smoke-image-001",
        )
        resp = api.embed_content(item, dimension=756, store=True)
        image_point_id = resp.point_id
        check("Image embedded", len(resp.vector) == 756, f"dim={resp.dimension}")
        check("Image stored in Qdrant", resp.point_id is not None, f"point_id={resp.point_id}")
    except Exception as exc:
        check("Image embedding", False, str(exc))

    # ------------------------------------------------------------------
    # 4. Single-stage search
    # ------------------------------------------------------------------
    print("\n[4] Single-stage search")
    try:
        query = ContentItem(content_type="text", data="dog playing on beach")
        resp = api.search(query, limit=5, dimension=756)
        check("Search returned results", resp.total_results > 0,
              f"{resp.total_results} result(s) in {resp.search_time_ms:.0f}ms")
        check("Results have scores", all(0 <= r.score <= 1 for r in resp.results))
        check("Results ordered descending",
              resp.results == sorted(resp.results, key=lambda r: r.score, reverse=True))
    except Exception as exc:
        check("Single-stage search", False, str(exc))

    # ------------------------------------------------------------------
    # 5. Modality filter
    # ------------------------------------------------------------------
    print("\n[5] Modality filter (text only)")
    try:
        query = ContentItem(content_type="text", data="dog playing on beach")
        filters = SearchFilters(content_types=["text"])
        resp = api.search(query, limit=5, filters=filters)
        all_text = all(r.content_type == "text" for r in resp.results)
        check("Filter returns only text", all_text,
              f"{resp.total_results} result(s)")
    except Exception as exc:
        check("Modality filter", False, str(exc))

    # ------------------------------------------------------------------
    # 6. Batch embedding
    # ------------------------------------------------------------------
    print("\n[6] Batch embedding")
    try:
        items = [
            ContentItem(content_type="text", data="The Eiffel Tower in Paris.", source_id="smoke-batch-001"),
            ContentItem(content_type="text", data="Machine learning with neural networks.", source_id="smoke-batch-002"),
            ContentItem(content_type="text", data="Ocean waves at sunset.", source_id="smoke-batch-003"),
        ]
        batch_resp = api.embed_batch(items, dimension=256, store=True)
        check("Batch count matches", batch_resp.total == 3, f"total={batch_resp.total}")
        check("Batch stored", batch_resp.stored == 3, f"stored={batch_resp.stored}")
        check("Batch vectors correct dim", all(len(r.vector) == 256 for r in batch_resp.results))
    except Exception as exc:
        check("Batch embedding", False, str(exc))

    # ------------------------------------------------------------------
    # 7. Two-stage retrieval
    # ------------------------------------------------------------------
    print("\n[7] Two-stage retrieval")
    try:
        # First index content with named vectors at both dims
        item = ContentItem(
            content_type="text",
            data="Transformer attention mechanism in deep learning.",
            source_id="smoke-twostage-001",
        )
        api.embed_content(item, dimension=256, store=True, named_vectors=[1024])

        query = ContentItem(content_type="text", data="attention mechanism transformers")
        first = StageConfig(dimension=256, limit=20)
        second = StageConfig(dimension=1024, limit=5)
        resp = api.search_two_stage(query, first_stage_config=first, second_stage_config=second)
        check("Two-stage returned results", resp.total_results > 0,
              f"{resp.total_results} result(s)")
        check("Two-stage flag set", resp.two_stage is True)
    except Exception as exc:
        check("Two-stage retrieval", False, str(exc))

    # ------------------------------------------------------------------
    # 8. Score threshold
    # ------------------------------------------------------------------
    print("\n[8] Score threshold filter")
    try:
        query = ContentItem(content_type="text", data="dog playing on beach")
        resp = api.search(query, limit=10, score_threshold=0.5)
        all_above = all(r.score >= 0.5 for r in resp.results)
        check("All results above threshold", all_above,
              f"{resp.total_results} result(s) with score≥0.5")
    except Exception as exc:
        check("Score threshold", False, str(exc))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"  Result: {passed}/{total} checks passed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
