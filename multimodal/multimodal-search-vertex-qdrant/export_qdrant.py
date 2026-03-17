"""
Export points from Qdrant multimodal_embeddings collection.

Fetches all points (or a filtered subset) and saves them as:
  - export_qdrant.json   — full payload + vector metadata
  - export_qdrant.csv    — flat summary (id, content_type, source_id, dimension, timestamp, score)

Usage:
    python export_qdrant.py                          # export all points
    python export_qdrant.py --limit 100              # first 100 points
    python export_qdrant.py --type image             # only image points
    python export_qdrant.py --type text --limit 50   # 50 text points
    python export_qdrant.py --with-vectors           # include raw vectors in JSON
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

load_dotenv()

COLLECTION = "multimodal_embeddings"
DEFAULT_OUTPUT_JSON = "export_qdrant.json"
DEFAULT_OUTPUT_CSV  = "export_qdrant.csv"
SCROLL_BATCH = 100   # points per scroll page


def build_filter(content_type: str | None) -> Filter | None:
    if not content_type:
        return None
    return Filter(
        must=[FieldCondition(key="content_type", match=MatchValue(value=content_type))]
    )


def scroll_all(client: QdrantClient, collection: str, limit: int | None,
               content_type: str | None, with_vectors: bool) -> list[dict]:
    """Scroll through all points in batches and return as list of dicts."""
    results = []
    offset = None
    filt = build_filter(content_type)

    while True:
        batch_size = min(SCROLL_BATCH, limit - len(results)) if limit else SCROLL_BATCH

        response, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )

        for point in response:
            entry = {
                "id": str(point.id),
                "payload": point.payload,
            }
            if with_vectors and point.vector:
                # Named vectors: store dimension sizes, not full arrays (too large)
                if isinstance(point.vector, dict):
                    entry["vector_dims"] = {k: len(v) for k, v in point.vector.items()}
                    entry["vectors"] = point.vector  # full vectors if requested
                else:
                    entry["vector_dims"] = {"default": len(point.vector)}
                    entry["vectors"] = point.vector
            results.append(entry)

        if next_offset is None or (limit and len(results) >= limit):
            break
        offset = next_offset

    return results[:limit] if limit else results


def to_csv_rows(points: list[dict]) -> list[dict]:
    rows = []
    for p in points:
        payload = p.get("payload", {})
        rows.append({
            "id":           p["id"],
            "content_type": payload.get("content_type", ""),
            "source_id":    payload.get("source_id", ""),
            "dimension":    payload.get("dimension", ""),
            "model_version":payload.get("model_version", ""),
            "timestamp":    payload.get("timestamp", ""),
            "language":     payload.get("language", ""),
            "duration":     payload.get("duration", ""),
            "page_count":   payload.get("page_count", ""),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Export Qdrant multimodal_embeddings collection")
    parser.add_argument("--limit",       type=int,  default=None,  help="Max points to export (default: all)")
    parser.add_argument("--type",        type=str,  default=None,  help="Filter by content_type: text|image|audio|video|pdf")
    parser.add_argument("--with-vectors",action="store_true",      help="Include raw vectors in JSON output")
    parser.add_argument("--out-json",    type=str,  default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--out-csv",     type=str,  default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--no-csv",      action="store_true",      help="Skip CSV output")
    parser.add_argument("--no-json",     action="store_true",      help="Skip JSON output")
    args = parser.parse_args()

    # ── Connect ──────────────────────────────────────────────────────
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")

    if not qdrant_url:
        print("ERROR: QDRANT_URL not set in environment / .env", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to Qdrant: {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    # ── Collection info ───────────────────────────────────────────────
    try:
        info = client.get_collection(COLLECTION)
        total = info.points_count
        print(f"Collection : {COLLECTION}")
        print(f"Total pts  : {total}")
    except Exception as e:
        print(f"ERROR: Could not access collection '{COLLECTION}': {e}", file=sys.stderr)
        sys.exit(1)

    # ── Scroll & fetch ────────────────────────────────────────────────
    filter_msg = f"  type={args.type}" if args.type else "  (all types)"
    limit_msg  = f"  limit={args.limit}" if args.limit else f"  limit=all ({total})"
    print(f"Fetching:{filter_msg}{limit_msg}  with_vectors={args.with_vectors}")

    points = scroll_all(
        client=client,
        collection=COLLECTION,
        limit=args.limit,
        content_type=args.type,
        with_vectors=args.with_vectors,
    )

    print(f"Fetched    : {len(points)} points")

    # ── Content type summary ──────────────────────────────────────────
    type_counts: dict[str, int] = {}
    for p in points:
        ct = p.get("payload", {}).get("content_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print("Breakdown  :", "  ".join(f"{k}={v}" for k, v in sorted(type_counts.items())))

    # ── Write JSON ────────────────────────────────────────────────────
    if not args.no_json:
        export_data = {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "collection":  COLLECTION,
            "qdrant_url":  qdrant_url,
            "total_fetched": len(points),
            "filters": {"content_type": args.type, "limit": args.limit},
            "type_breakdown": type_counts,
            "points": points,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"JSON saved : {args.out_json}")

    # ── Write CSV ─────────────────────────────────────────────────────
    if not args.no_csv:
        rows = to_csv_rows(points)
        if rows:
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"CSV saved  : {args.out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
