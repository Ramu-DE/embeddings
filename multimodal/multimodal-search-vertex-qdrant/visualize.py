"""
Multimodal Search — Live Visualization
========================================
Runs live queries against Gemini + Qdrant and produces a
multi-panel chart saved as  charts.png

Panels:
  1. Matryoshka Dimensions  — vector norm vs dimension size
  2. Cross-Modal Search     — similarity scores by modality
  3. Multilingual Search    — heatmap: query lang x doc lang
  4. Score Threshold        — result count vs threshold
  5. Two-Stage vs Single    — latency comparison
  6. RAG Retrieval          — top-2 scores per question
  7. Recommendation Engine  — similarity scores for seed item
  8. Batch Embedding        — throughput (docs/sec) by dim
"""

import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from multimodal_search.api import MultimodalSearchAPI
from multimodal_search.models import ContentItem, StageConfig, SearchFilters, InterleavedPart

# ── Stub JPEG ─────────────────────────────────────────────────────────────────
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

COLORS = {
    "text":        "#4C9BE8",
    "image":       "#F4845F",
    "interleaved": "#A78BFA",
    "en":          "#34D399",
    "es":          "#FBBF24",
    "fr":          "#60A5FA",
    "ja":          "#F87171",
    "accent":      "#6366F1",
    "bg":          "#0F172A",
    "panel":       "#1E293B",
    "grid":        "#334155",
    "text_col":    "#E2E8F0",
    "muted":       "#94A3B8",
}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  COLORS["bg"],
    "axes.facecolor":    COLORS["panel"],
    "axes.edgecolor":    COLORS["grid"],
    "axes.labelcolor":   COLORS["text_col"],
    "axes.titlecolor":   COLORS["text_col"],
    "xtick.color":       COLORS["muted"],
    "ytick.color":       COLORS["muted"],
    "text.color":        COLORS["text_col"],
    "grid.color":        COLORS["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
})


def step(msg):
    print(f"  >> {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Collect live data
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Collecting live data from Gemini + Qdrant ===\n")

api = MultimodalSearchAPI.from_env()
status = api.initialize_system()
if not status.initialized:
    print("ERROR: system not ready"); sys.exit(1)
print("  System ready\n")

# ── 1a. Matryoshka: embed same text at all dims, record norm ──────────────────
step("Matryoshka dimensions")
DIMS = [128, 256, 512, 756, 1024, 1536, 2048, 3072]
matryoshka_norms = []
matryoshka_times = []
text = "A golden retriever playing fetch on a sunny beach."
for d in DIMS:
    t0 = time.time()
    r = api._embedding_service.embed_text(text, dimension=d)
    elapsed = (time.time() - t0) * 1000
    matryoshka_norms.append(float(np.linalg.norm(r.vector)))
    matryoshka_times.append(elapsed)

# ── 1b. Index fresh content ───────────────────────────────────────────────────
step("Indexing content")

# Text docs
text_docs = [
    ("viz-text-001", "A golden retriever playing fetch on a sunny beach."),
    ("viz-text-002", "Ocean waves crashing on the shore at sunset."),
    ("viz-text-003", "Machine learning with neural networks and transformers."),
    ("viz-text-004", "The Eiffel Tower stands tall in Paris, France."),
]
for src, txt in text_docs:
    api.embed_content(ContentItem(content_type="text", data=txt, source_id=src),
                      dimension=756, store=True)

# Image
api.embed_content(ContentItem(content_type="image", data=STUB_JPEG,
                               mime_type="image/jpeg", source_id="viz-image-001"),
                  dimension=756, store=True)

# Interleaved
api.embed_content(ContentItem(
    content_type="interleaved", source_id="viz-interleaved-001",
    interleaved_parts=[
        InterleavedPart(content_type="image", data=STUB_JPEG, mime_type="image/jpeg"),
        InterleavedPart(content_type="text",  data="Dog playing on a sunny beach."),
    ]), dimension=756, store=True)

# Multilingual
ml_docs = [
    ("ml-en", "en", "The Eiffel Tower is a famous landmark in Paris."),
    ("ml-es", "es", "La Torre Eiffel es un famoso monumento en Paris."),
    ("ml-fr", "fr", "La Tour Eiffel est un monument celebre a Paris."),
    ("ml-ja", "ja", "The Eiffel Tower is located in Paris France."),
]
for src, lang, txt in ml_docs:
    api.embed_content(ContentItem(content_type="text", data=txt, source_id=src),
                      dimension=756, store=True)

# Two-stage docs (need both dim=256 and dim=1024)
ts_docs = [
    ("ts-001", "Transformer attention mechanism in deep learning."),
    ("ts-002", "BERT and GPT language model architectures."),
    ("ts-003", "Convolutional neural networks for image recognition."),
    ("ts-004", "Reinforcement learning and reward optimization."),
    ("ts-005", "Graph neural networks for relational data."),
]
for src, txt in ts_docs:
    api.embed_content(ContentItem(content_type="text", data=txt, source_id=src),
                      dimension=256, store=True, named_vectors=[1024])

# RAG KB
rag_docs = [
    ("kb-hr-001",    "Parental leave policy: 16 weeks paid leave for primary caregivers."),
    ("kb-hr-002",    "Learning budget: $2,000 annual learning and development allowance."),
    ("kb-infra-001", "Deployments require two approvals and a passing CI pipeline."),
    ("kb-infra-002", "Database backups run every 6 hours, retained for 30 days."),
    ("kb-api-001",   "API rate limit is 1,000 requests per minute per API key."),
    ("kb-api-002",   "Webhooks support retry with exponential backoff, up to 5 attempts."),
]
for src, txt in rag_docs:
    api.embed_content(ContentItem(content_type="text", data=txt, source_id=src),
                      dimension=756, store=True)

# Recommendation catalogue
products = [
    ("prod-001", "Sony WH-1000XM5 wireless noise-cancelling headphones"),
    ("prod-002", "Bose QuietComfort 45 Bluetooth headphones"),
    ("prod-003", "Apple AirPods Pro 2nd generation earbuds"),
    ("prod-004", "Samsung Galaxy Buds2 Pro true wireless earbuds"),
    ("prod-005", "Logitech MX Master 3 wireless ergonomic mouse"),
    ("prod-006", "Apple Magic Keyboard with Touch ID"),
]
for src, txt in products:
    api.embed_content(ContentItem(content_type="text", data=txt, source_id=src),
                      dimension=756, store=True)

# ── 1c. Cross-modal search ────────────────────────────────────────────────────
step("Cross-modal search")
query = ContentItem(content_type="text", data="dog playing on beach")
cross_resp = api.search(query, limit=10, dimension=756)
cross_scores  = [r.score        for r in cross_resp.results]
cross_types   = [r.content_type for r in cross_resp.results]
cross_sources = [r.source_id    for r in cross_resp.results]

# ── 1d. Multilingual heatmap ──────────────────────────────────────────────────
step("Multilingual heatmap")
query_langs  = ["EN", "ES", "FR", "JA"]
query_texts  = [
    "Eiffel Tower Paris landmark",
    "Torre Eiffel monumento famoso",
    "Tour Eiffel monument Paris",
    "Eiffel Tower Paris France famous",
]
doc_langs = ["EN", "ES", "FR", "JA"]
doc_srcs  = ["ml-en", "ml-es", "ml-fr", "ml-ja"]

ml_matrix = np.zeros((4, 4))
for qi, (qlang, qtext) in enumerate(zip(query_langs, query_texts)):
    resp = api.search(ContentItem(content_type="text", data=qtext),
                      limit=10, dimension=756)
    score_map = {r.source_id: r.score for r in resp.results}
    for di, src in enumerate(doc_srcs):
        ml_matrix[qi, di] = score_map.get(src, 0.0)

# ── 1e. Score threshold ───────────────────────────────────────────────────────
step("Score threshold sweep")
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_counts = []
for t in thresholds:
    r = api.search(query, limit=50, score_threshold=t if t > 0 else None)
    threshold_counts.append(r.total_results)

# ── 1f. Two-stage vs single-stage latency ─────────────────────────────────────
step("Two-stage vs single-stage latency")
ts_query = ContentItem(content_type="text", data="attention mechanism transformers")
latencies = {"single_256": [], "single_1024": [], "two_stage": []}
for _ in range(3):
    t0 = time.time()
    api.search(ts_query, limit=5, dimension=256)
    latencies["single_256"].append((time.time()-t0)*1000)

    t0 = time.time()
    api.search(ts_query, limit=5, dimension=1024)
    latencies["single_1024"].append((time.time()-t0)*1000)

    t0 = time.time()
    api.search_two_stage(ts_query,
                         first_stage_config=StageConfig(dimension=256, limit=20),
                         second_stage_config=StageConfig(dimension=1024, limit=5))
    latencies["two_stage"].append((time.time()-t0)*1000)

avg_lat = {k: np.mean(v) for k, v in latencies.items()}

# ── 1g. RAG retrieval scores ──────────────────────────────────────────────────
step("RAG retrieval")
rag_questions = [
    ("Parental leave?",    "How much parental leave do employees get?"),
    ("Deploy process?",    "What is the deployment approval process?"),
    ("API rate limits?",   "What are the API rate limits?"),
]
rag_data = {}
for label, q in rag_questions:
    resp = api.search(ContentItem(content_type="text", data=q),
                      limit=2, score_threshold=0.3)
    rag_data[label] = [(r.source_id, r.score) for r in resp.results]

# ── 1h. Recommendation scores ─────────────────────────────────────────────────
step("Recommendation engine")
seed = "Sony WH-1000XM5 wireless noise-cancelling headphones"
rec_resp = api.search(ContentItem(content_type="text", data=seed),
                      limit=6, dimension=756)
rec_sources = [r.source_id for r in rec_resp.results]
rec_scores  = [r.score     for r in rec_resp.results]

# ── 1i. Batch throughput ──────────────────────────────────────────────────────
step("Batch throughput")
batch_items = [ContentItem(content_type="text",
                            data=f"Sample document number {i} about various topics.",
                            source_id=f"batch-{i}")
               for i in range(5)]
batch_dims = [128, 256, 512, 756, 1024]
batch_throughput = []
for d in batch_dims:
    t0 = time.time()
    api.embed_batch(batch_items, dimension=d, store=False)
    elapsed = time.time() - t0
    batch_throughput.append(len(batch_items) / elapsed)

print("\n  All data collected. Building charts...\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Build the figure
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor(COLORS["bg"])

gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
              left=0.07, right=0.97, top=0.94, bottom=0.04)

# Title
fig.text(0.5, 0.97,
         "Gemini Embedding 2 + Qdrant  |  Multimodal Search Capabilities",
         ha="center", va="top", fontsize=16, fontweight="bold",
         color=COLORS["text_col"])
fig.text(0.5, 0.955,
         "Live results from gemini-embedding-2-preview  x  Qdrant Cloud",
         ha="center", va="top", fontsize=10, color=COLORS["muted"])


# ── Panel 1: Matryoshka Dimensions ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("1. Matryoshka Dimensions  (vector L2 norm vs size)", pad=8)

color_norm = COLORS["accent"]
color_time = "#F59E0B"

ax1b = ax1.twinx()
bars = ax1.bar([str(d) for d in DIMS], matryoshka_norms,
               color=color_norm, alpha=0.75, zorder=3)
ax1b.plot([str(d) for d in DIMS], matryoshka_times,
          color=color_time, marker="o", linewidth=2, zorder=4, label="Latency (ms)")

ax1.set_xlabel("Dimension")
ax1.set_ylabel("L2 Norm", color=color_norm)
ax1b.set_ylabel("Latency (ms)", color=color_time)
ax1b.tick_params(axis="y", colors=color_time)
ax1.tick_params(axis="y", colors=color_norm)
ax1.set_ylim(0, max(matryoshka_norms) * 1.3)
ax1.grid(axis="y", zorder=0)

for bar, norm in zip(bars, matryoshka_norms):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{norm:.2f}", ha="center", va="bottom", fontsize=7,
             color=COLORS["text_col"])

ax1b.legend(loc="upper right", fontsize=8,
            facecolor=COLORS["panel"], edgecolor=COLORS["grid"])


# ── Panel 2: Cross-Modal Search ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("2. Cross-Modal Search  (text query -> text + image results)", pad=8)

type_colors = [COLORS.get(t, COLORS["accent"]) for t in cross_types]
short_src   = [s.replace("viz-", "").replace("ml-", "") for s in cross_sources]

bars2 = ax2.barh(range(len(cross_scores)), cross_scores,
                 color=type_colors, alpha=0.85, zorder=3)
ax2.set_yticks(range(len(cross_scores)))
ax2.set_yticklabels(short_src, fontsize=8)
ax2.set_xlabel("Cosine Similarity Score")
ax2.set_xlim(0, 1.1)
ax2.grid(axis="x", zorder=0)
ax2.invert_yaxis()

for bar, score, ctype in zip(bars2, cross_scores, cross_types):
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{score:.3f} [{ctype}]", va="center", fontsize=7.5,
             color=COLORS["text_col"])

patches = [mpatches.Patch(color=COLORS["text"],  label="text"),
           mpatches.Patch(color=COLORS["image"], label="image"),
           mpatches.Patch(color=COLORS["interleaved"], label="interleaved")]
ax2.legend(handles=patches, loc="lower right", fontsize=8,
           facecolor=COLORS["panel"], edgecolor=COLORS["grid"])


# ── Panel 3: Multilingual Heatmap ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title("3. Multilingual Search  (query lang x doc lang similarity)", pad=8)

im = ax3.imshow(ml_matrix, cmap="YlOrRd", vmin=0.5, vmax=1.0, aspect="auto")
ax3.set_xticks(range(4)); ax3.set_xticklabels(doc_langs)
ax3.set_yticks(range(4)); ax3.set_yticklabels(query_langs)
ax3.set_xlabel("Document Language")
ax3.set_ylabel("Query Language")

for i in range(4):
    for j in range(4):
        val = ml_matrix[i, j]
        color = "black" if val > 0.75 else "white"
        ax3.text(j, i, f"{val:.3f}", ha="center", va="center",
                 fontsize=9, fontweight="bold", color=color)

cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color=COLORS["muted"])
cbar.set_label("Similarity", color=COLORS["muted"])


# ── Panel 4: Score Threshold ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title("4. Score Threshold  (result count vs minimum similarity)", pad=8)

ax4.fill_between(thresholds, threshold_counts, alpha=0.3, color=COLORS["accent"])
ax4.plot(thresholds, threshold_counts, color=COLORS["accent"],
         marker="o", linewidth=2.5, zorder=4)

for x, y in zip(thresholds, threshold_counts):
    if y > 0:
        ax4.annotate(str(y), (x, y), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=8,
                     color=COLORS["text_col"])

ax4.set_xlabel("Score Threshold")
ax4.set_ylabel("Results Returned")
ax4.set_xticks(thresholds)
ax4.set_xticklabels([str(t) for t in thresholds], fontsize=8)
ax4.grid(zorder=0)
ax4.set_ylim(0, max(threshold_counts) * 1.25)


# ── Panel 5: Two-Stage vs Single Latency ─────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_title("5. Two-Stage vs Single-Stage  (avg latency, 3 runs)", pad=8)

labels5  = ["Single\ndim=256", "Single\ndim=1024", "Two-Stage\n256->1024"]
values5  = [avg_lat["single_256"], avg_lat["single_1024"], avg_lat["two_stage"]]
colors5  = [COLORS["text"], COLORS["image"], COLORS["accent"]]

bars5 = ax5.bar(labels5, values5, color=colors5, alpha=0.85, width=0.5, zorder=3)
ax5.set_ylabel("Latency (ms)")
ax5.grid(axis="y", zorder=0)

for bar, val in zip(bars5, values5):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{val:.0f}ms", ha="center", va="bottom", fontsize=9,
             fontweight="bold", color=COLORS["text_col"])

ax5.set_ylim(0, max(values5) * 1.3)

note = "Two-stage: fast scan + accurate re-rank"
ax5.text(0.5, 0.95, note, transform=ax5.transAxes,
         ha="center", va="top", fontsize=8, color=COLORS["muted"],
         style="italic")


# ── Panel 6: RAG Retrieval ────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_title("6. RAG Knowledge Base  (top-2 retrieval scores per question)", pad=8)

q_labels = list(rag_data.keys())
x6 = np.arange(len(q_labels))
width = 0.35

for qi, (label, hits) in enumerate(rag_data.items()):
    for rank, (src, score) in enumerate(hits):
        offset = (rank - 0.5) * width
        bar = ax6.bar(qi + offset, score, width * 0.9,
                      color=COLORS["accent"] if rank == 0 else COLORS["text"],
                      alpha=0.85, zorder=3)
        ax6.text(qi + offset, score + 0.005, f"{score:.3f}",
                 ha="center", va="bottom", fontsize=7.5,
                 color=COLORS["text_col"])
        ax6.text(qi + offset, score / 2,
                 src.replace("kb-", "").replace("-001", "").replace("-002", ""),
                 ha="center", va="center", fontsize=6.5,
                 color="white", rotation=90)

ax6.set_xticks(x6)
ax6.set_xticklabels(q_labels, fontsize=8.5)
ax6.set_ylabel("Similarity Score")
ax6.set_ylim(0, 1.1)
ax6.axhline(0.8, color=COLORS["muted"], linestyle="--", linewidth=1, alpha=0.6)
ax6.text(len(q_labels)-0.5, 0.81, "0.8 threshold", fontsize=7,
         color=COLORS["muted"])
ax6.grid(axis="y", zorder=0)

rank1 = mpatches.Patch(color=COLORS["accent"], label="Rank 1")
rank2 = mpatches.Patch(color=COLORS["text"],   label="Rank 2")
ax6.legend(handles=[rank1, rank2], fontsize=8,
           facecolor=COLORS["panel"], edgecolor=COLORS["grid"])


# ── Panel 7: Recommendation Engine ───────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
ax7.set_title("7. Recommendation Engine  (similar items to Sony WH-1000XM5)", pad=8)

short_rec = [s.replace("prod-", "P") for s in rec_sources]
rec_colors = [COLORS["accent"] if s == 1.0 else COLORS["text"]
              for s in rec_scores]

bars7 = ax7.bar(short_rec, rec_scores, color=rec_colors, alpha=0.85, zorder=3)
ax7.set_ylabel("Cosine Similarity")
ax7.set_ylim(0, 1.15)
ax7.grid(axis="y", zorder=0)

prod_names = {
    "prod-001": "Sony XM5",
    "prod-002": "Bose QC45",
    "prod-003": "AirPods Pro",
    "prod-004": "Galaxy Buds2",
    "prod-005": "MX Master 3",
    "prod-006": "Magic KB",
}
for bar, src, score in zip(bars7, rec_sources, rec_scores):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{score:.3f}", ha="center", va="bottom", fontsize=8,
             color=COLORS["text_col"])
    ax7.text(bar.get_x() + bar.get_width()/2, 0.02,
             prod_names.get(src, src), ha="center", va="bottom",
             fontsize=7, color="white", rotation=45)

ax7.text(0.5, 0.97, "Seed: Sony WH-1000XM5 (score=1.0 = exact match)",
         transform=ax7.transAxes, ha="center", va="top",
         fontsize=8, color=COLORS["muted"], style="italic")


# ── Panel 8: Batch Throughput ─────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
ax8.set_title("8. Batch Embedding Throughput  (docs/sec by dimension)", pad=8)

bar_colors8 = plt.cm.plasma(np.linspace(0.2, 0.8, len(batch_dims)))
bars8 = ax8.bar([str(d) for d in batch_dims], batch_throughput,
                color=bar_colors8, alpha=0.9, zorder=3)
ax8.set_xlabel("Embedding Dimension")
ax8.set_ylabel("Throughput (docs / sec)")
ax8.grid(axis="y", zorder=0)

for bar, tp in zip(bars8, batch_throughput):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{tp:.2f}", ha="center", va="bottom", fontsize=8,
             color=COLORS["text_col"])

ax8.set_ylim(0, max(batch_throughput) * 1.3)


# ── Save ──────────────────────────────────────────────────────────────────────
out = "charts.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
print(f"  Saved -> {out}")
print(f"  Open charts.png to view all 8 panels.\n")
