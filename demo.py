"""
Multimodal Search — Capability Demo
=====================================
Showcases all major capabilities of the system:

  1.  System init & health check
  2.  Text embedding (multiple dimensions)
  3.  Image embedding (cross-modal)
  4.  Batch embedding (3 docs at once)
  5.  Two-stage retrieval (speed vs accuracy)
  6.  Cross-modal search  (text query → image results)
  7.  Modality filter     (text-only, image-only)
  8.  Score threshold     (high-confidence only)
  9.  Multilingual search (EN / ES / FR / JA)
  10. Interleaved multimodal (image + caption → unified vector)
  11. RAG knowledge base   (index + retrieve enterprise docs)
  12. Recommendation engine (find similar items)

Run:
    py demo.py
"""

import sys
import textwrap
import io
# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from multimodal_search.api import MultimodalSearchAPI
from multimodal_search.models import ContentItem, StageConfig, SearchFilters, InterleavedPart

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}[OK]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"

# Minimal 1×1 JPEG stub (no real image file needed)
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

results_summary = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'-'*60}{RESET}")


def ok(msg: str, detail: str = "") -> None:
    line = f"  {PASS} {msg}"
    if detail:
        line += f"  {YELLOW}({detail}){RESET}"
    print(line)
    results_summary.append((msg, True))


def fail(msg: str, err) -> None:
    print(f"  {FAIL} {msg}: {RED}{err}{RESET}")
    results_summary.append((msg, False))


def print_results(resp, label: str = "") -> None:
    if label:
        print(f"  {BOLD}{label}{RESET}")
    print(f"  {resp.total_results} result(s)  |  {resp.search_time_ms:.0f} ms")
    for i, r in enumerate(resp.results, 1):
        bar = "█" * int(r.score * 20)
        print(f"    [{i}] {bar:<20} {r.score:.4f}  [{r.content_type:>12}]  {r.source_id}")


def embed(api, item, **kwargs):
    return api.embed_content(item, **kwargs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*60}")
    print(f"  Multimodal Search -- Full Capability Demo")
    print(f"{'='*60}{RESET}")
    print("  Model : gemini-embedding-2-preview")
    print("  Store : Qdrant Cloud")
    print("  Dims  : 128 / 256 / 512 / 756 / 1024 / 1536 / 2048 / 3072")

    # ── 1. Init ───────────────────────────────────────────────────────
    section("1. System Initialisation & Health Check")
    try:
        api = MultimodalSearchAPI.from_env()
        status = api.initialize_system()
        ok("API constructed")
        ok("Vertex AI (Gemini Embedding 2)", "connected" if status.vertex_ai_connected else "FAILED")
        ok("Qdrant Cloud", "connected" if status.qdrant_connected else "FAILED")
        ok("Collection ready", f"named vectors: dim_128 … dim_3072")
        if not status.initialized:
            print(f"\n  {RED}FATAL — aborting{RESET}")
            for e in status.errors:
                print(f"    {e}")
            sys.exit(1)
    except Exception as exc:
        fail("System init", exc)
        sys.exit(1)

    # ── 2. Text embedding at multiple Matryoshka dimensions ───────────
    section("2. Text Embedding — Matryoshka Dimensions")
    print("  Gemini Embedding 2 supports truncated dimensions from a single model call.")
    for dim in [128, 256, 756, 1024, 3072]:
        try:
            item = ContentItem(content_type="text",
                               data="A golden retriever playing fetch on a sunny beach.",
                               source_id=f"demo-text-dim{dim}")
            resp = embed(api, item, dimension=dim, store=(dim == 756))
            stored = f"stored={resp.point_id is not None}" if dim == 756 else "not stored"
            ok(f"dim={dim:>4}  vector[0:3]={[round(v,4) for v in resp.vector[:3]]}", stored)
        except Exception as exc:
            fail(f"dim={dim}", exc)

    # ── 3. Image embedding ────────────────────────────────────────────
    section("3. Image Embedding (Cross-Modal Shared Space)")
    print("  Images and text share the same vector space — enabling cross-modal search.")
    try:
        item = ContentItem(content_type="image", data=STUB_JPEG,
                           mime_type="image/jpeg", source_id="demo-image-001")
        resp = embed(api, item, dimension=756, store=True)
        ok("Image embedded + stored", f"dim=756  point_id={resp.point_id}")
    except Exception as exc:
        fail("Image embedding", exc)

    # ── 4. Batch embedding ────────────────────────────────────────────
    section("4. Batch Embedding (3 documents at dim=256)")
    docs = [
        ContentItem(content_type="text", data="The Eiffel Tower stands in Paris, France.", source_id="rag-doc-001"),
        ContentItem(content_type="text", data="Neural networks learn representations from data.", source_id="rag-doc-002"),
        ContentItem(content_type="text", data="Ocean waves crash on the sandy shore at sunset.", source_id="rag-doc-003"),
    ]
    try:
        batch = api.embed_batch(docs, dimension=256, store=True)
        ok(f"Batch processed", f"total={batch.total}  stored={batch.stored}")
        for r in batch.results:
            ok(f"  {r.content_type}  dim={r.dimension}  point_id={r.point_id}")
    except Exception as exc:
        fail("Batch embedding", exc)

    # ── 5. Two-stage retrieval ────────────────────────────────────────
    section("5. Two-Stage Retrieval (Speed ↔ Accuracy Trade-off)")
    print("  Stage 1: fast scan at dim=256 → top-20 candidates")
    print("  Stage 2: accurate re-rank at dim=1024 → top-5 final results")
    try:
        # Index with both dims
        item = ContentItem(content_type="text",
                           data="Transformer attention mechanism in deep learning.",
                           source_id="twostage-doc-001")
        embed(api, item, dimension=256, store=True, named_vectors=[1024])

        query = ContentItem(content_type="text", data="attention mechanism transformers")
        resp = api.search_two_stage(
            query,
            first_stage_config=StageConfig(dimension=256, limit=20),
            second_stage_config=StageConfig(dimension=1024, limit=5),
        )
        ok("Two-stage search completed", f"two_stage={resp.two_stage}")
        print_results(resp)
    except Exception as exc:
        fail("Two-stage retrieval", exc)

    # ── 6. Cross-modal search ─────────────────────────────────────────
    section("6. Cross-Modal Search (Text Query → All Modalities)")
    print("  A text query retrieves semantically similar text AND images.")
    try:
        query = ContentItem(content_type="text", data="dog playing on beach")
        resp = api.search(query, limit=8, dimension=756)
        ok("Cross-modal search", f"{resp.total_results} result(s)")
        print_results(resp)
    except Exception as exc:
        fail("Cross-modal search", exc)

    # ── 7. Modality filters ───────────────────────────────────────────
    section("7. Modality Filters")
    query = ContentItem(content_type="text", data="ocean beach sunset")

    print(f"\n  {BOLD}Text-only filter:{RESET}")
    try:
        resp = api.search(query, limit=5, filters=SearchFilters(content_types=["text"]))
        ok("text-only filter", f"{resp.total_results} result(s)")
        print_results(resp)
    except Exception as exc:
        fail("text-only filter", exc)

    print(f"\n  {BOLD}Image-only filter:{RESET}")
    try:
        resp = api.search(query, limit=5, filters=SearchFilters(content_types=["image"]))
        ok("image-only filter", f"{resp.total_results} result(s)")
        print_results(resp)
    except Exception as exc:
        fail("image-only filter", exc)

    # ── 8. Score threshold ────────────────────────────────────────────
    section("8. Score Threshold (High-Confidence Results Only)")
    for threshold in [0.3, 0.5, 0.7]:
        try:
            query = ContentItem(content_type="text", data="dog playing on beach")
            resp = api.search(query, limit=20, score_threshold=threshold)
            ok(f"threshold≥{threshold}", f"{resp.total_results} result(s) above threshold")
        except Exception as exc:
            fail(f"threshold={threshold}", exc)

    # ── 9. Multilingual search ────────────────────────────────────────
    section("9. Multilingual Search (100+ Languages, Shared Vector Space)")
    print("  Index content in multiple languages, query in any language.")

    multilingual = [
        ("ml-en-001", "The Eiffel Tower is a famous landmark in Paris."),
        ("ml-es-001", "La Torre Eiffel es un famoso monumento en París."),
        ("ml-fr-001", "La Tour Eiffel est un monument célèbre à Paris."),
        ("ml-ja-001", "エッフェル塔はパリにある有名なランドマークです。"),
    ]
    for src, text in multilingual:
        try:
            item = ContentItem(content_type="text", data=text, source_id=src)
            embed(api, item, dimension=756, store=True)
            ok(f"Indexed [{src}]", text[:50])
        except Exception as exc:
            fail(f"Index {src}", exc)

    for lang, query_text in [("EN", "Eiffel Tower Paris landmark"),
                              ("ES", "Torre Eiffel monumento famoso"),
                              ("JA", "エッフェル塔 パリ")]:
        try:
            query = ContentItem(content_type="text", data=query_text)
            resp = api.search(query, limit=4, dimension=756)
            ok(f"{lang} query → {resp.total_results} result(s)", query_text)
            for r in resp.results:
                print(f"    score={r.score:.4f}  source={r.source_id}")
        except Exception as exc:
            fail(f"{lang} query", exc)

    # ── 10. Interleaved multimodal ────────────────────────────────────
    section("10. Interleaved Multimodal (Image + Caption → Unified Vector)")
    print("  Combine image bytes + text description into a single embedding.")
    try:
        item = ContentItem(
            content_type="interleaved",
            source_id="interleaved-product-001",
            interleaved_parts=[
                InterleavedPart(content_type="image", data=STUB_JPEG, mime_type="image/jpeg"),
                InterleavedPart(content_type="text",  data="Wireless noise-cancelling headphones, 30h battery."),
            ],
        )
        resp = embed(api, item, dimension=756, store=True)
        ok("Interleaved (image+text) embedded + stored",
           f"dim=756  point_id={resp.point_id}")

        # Search for it
        query = ContentItem(content_type="text", data="noise cancelling headphones battery life")
        search_resp = api.search(query, limit=5)
        ok("Retrieved via text query", f"{search_resp.total_results} result(s)")
        print_results(search_resp)
    except Exception as exc:
        fail("Interleaved multimodal", exc)

    # ── 11. RAG knowledge base ────────────────────────────────────────
    section("11. RAG — Enterprise Knowledge Base")
    print("  Index enterprise documents, retrieve relevant context for a question.")

    kb_docs = [
        ("kb-hr-001",      "Our parental leave policy provides 16 weeks of paid leave for primary caregivers."),
        ("kb-hr-002",      "Employees are eligible for a $2,000 annual learning & development budget."),
        ("kb-infra-001",   "All production deployments require two approvals and a passing CI pipeline."),
        ("kb-infra-002",   "Database backups run every 6 hours and are retained for 30 days."),
        ("kb-product-001", "The API rate limit is 1,000 requests per minute per API key."),
        ("kb-product-002", "Webhooks support retry with exponential backoff up to 5 attempts."),
    ]
    for src, text in kb_docs:
        try:
            item = ContentItem(content_type="text", data=text, source_id=src)
            embed(api, item, dimension=756, store=True)
        except Exception as exc:
            fail(f"Index {src}", exc)

    questions = [
        "How much parental leave do employees get?",
        "What is the deployment approval process?",
        "What are the API rate limits?",
    ]
    for q in questions:
        try:
            query = ContentItem(content_type="text", data=q)
            resp = api.search(query, limit=2, score_threshold=0.4)
            print(f"\n  {BOLD}Q: {q}{RESET}")
            for r in resp.results:
                wrapped = textwrap.fill(r.source_id + ": " + (r.metadata.custom_metadata or {}).get("text", ""), 70, initial_indent="    ", subsequent_indent="      ")
                print(f"    score={r.score:.4f}  source={r.source_id}")
            ok(f"RAG retrieval", f"{resp.total_results} relevant doc(s) found")
        except Exception as exc:
            fail(f"RAG query: {q[:40]}", exc)

    # ── 12. Recommendation engine ─────────────────────────────────────
    section("12. Recommendation Engine (Find Similar Items)")
    print("  Index a product catalogue, then find items similar to a seed item.")

    products = [
        ("prod-001", "Sony WH-1000XM5 wireless noise-cancelling headphones"),
        ("prod-002", "Bose QuietComfort 45 Bluetooth headphones"),
        ("prod-003", "Apple AirPods Pro 2nd generation earbuds"),
        ("prod-004", "Samsung Galaxy Buds2 Pro true wireless earbuds"),
        ("prod-005", "Logitech MX Master 3 wireless ergonomic mouse"),
        ("prod-006", "Apple Magic Keyboard with Touch ID"),
    ]
    for src, text in products:
        try:
            item = ContentItem(content_type="text", data=text, source_id=src)
            embed(api, item, dimension=756, store=True)
        except Exception as exc:
            fail(f"Index {src}", exc)

    seed = "Sony WH-1000XM5 wireless noise-cancelling headphones"
    try:
        query = ContentItem(content_type="text", data=seed)
        resp = api.search(query, limit=4, score_threshold=0.5)
        ok(f"Recommendations for: {seed[:45]}…", f"{resp.total_results} similar item(s)")
        print_results(resp)
    except Exception as exc:
        fail("Recommendation search", exc)

    # ── Summary ───────────────────────────────────────────────────────
    passed = sum(1 for _, ok_ in results_summary if ok_)
    total  = len(results_summary)
    print(f"\n{BOLD}{'='*60}")
    print(f"  Demo complete: {passed}/{total} checks passed")
    print(f"{'='*60}{RESET}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
