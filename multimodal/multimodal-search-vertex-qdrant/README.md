# Multimodal Semantic Search — Vertex AI + Qdrant

A production-ready PoC for multimodal semantic search using **Google Vertex AI Gemini Embedding 2** and **Qdrant** vector database. The system maps text, images, audio, video, and PDFs into a single shared vector space, enabling cross-modal similarity search with a single query.

---

## What We Built

### The Problem
Traditional search systems are modality-specific — a text search engine can't find images, and an image retrieval system can't be queried with text. Building a system that understands "find me images of a dog on a beach" from a text query, or "find documents similar to this image" requires a unified semantic space across all modalities.

### The Solution
We built a multimodal search engine that:
- Embeds any content type (text, image, audio, video, PDF) into the **same vector space** using Gemini Embedding 2
- Stores vectors in **Qdrant** with rich metadata for filtering
- Supports **cross-modal queries** — query with text, get back images (and vice versa)
- Supports **100+ languages** natively with no translation step
- Implements **two-stage retrieval** for speed/accuracy trade-offs at scale

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MultimodalSearchAPI                       │
│  (single entry point — wraps all components)                │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────┐           ┌─────────────────────┐
│ EmbeddingService│           │    VectorStore       │
│                 │           │                      │
│ Vertex AI       │           │ Qdrant (Cloud/Local) │
│ Gemini Emb 2    │           │ Named vectors        │
│                 │           │ Payload filtering    │
│ Modalities:     │           │ Cosine similarity    │
│  text           │           └─────────────────────┘
│  image          │
│  audio          │           ┌─────────────────────┐
│  video          │           │   SearchEngine       │
│  pdf            │           │                      │
│  interleaved    │           │ Single-stage search  │
└─────────────────┘           │ Two-stage retrieval  │
                              │ Cross-modal search   │
                              │ Multilingual search  │
                              └─────────────────────┘
```

---

## Key Capabilities

### 1. Matryoshka Embeddings
Gemini Embedding 2 supports **truncated dimensions** from a single model call — no need to re-embed content at different sizes. Supported dimensions:

| Dimension | Use Case |
|---|---|
| 128 / 256 | Fast first-stage retrieval, low memory |
| 512 / 756 | Balanced (default) |
| 1024 / 1536 | High-accuracy re-ranking |
| 2048 / 3072 | Maximum precision |

### 2. Cross-Modal Search
All modalities share the same vector space. A text query like `"dog playing on beach"` can retrieve both text documents and images with matching semantics — no separate pipelines needed.

### 3. Two-Stage Retrieval
For large-scale collections, we implemented a speed/accuracy pipeline:
- **Stage 1**: Fast scan at `dim=256` → retrieve top-100 candidates
- **Stage 2**: Accurate re-rank at `dim=1024` → return top-10 final results

This uses Qdrant's **named vectors** — each point stores embeddings at multiple dimensions simultaneously.

### 4. Interleaved Multimodal
Combine multiple modalities into a single unified embedding. Example: an image + its text caption embedded together captures richer semantics than either alone.

```python
item = ContentItem(
    content_type="interleaved",
    interleaved_parts=[
        InterleavedPart(content_type="image", data=image_bytes, mime_type="image/jpeg"),
        InterleavedPart(content_type="text", data="Wireless noise-cancelling headphones"),
    ]
)
```

### 5. Multilingual Search
Gemini Embedding 2 natively supports 100+ languages in the same vector space. Index content in English, French, Japanese — query in any language and get semantically relevant results across all languages.

### 6. Rich Filtering
Every stored vector carries metadata payload in Qdrant, enabling filtered search:
- Filter by **modality** (text only, images only, etc.)
- Filter by **language**
- Filter by **timestamp range**
- Filter by **source ID**
- Custom metadata filters

---

## Project Structure

```
multimodal-search-vertex-qdrant/
├── src/multimodal_search/
│   ├── api.py                  # Main entry point — MultimodalSearchAPI
│   ├── embedding_service.py    # Vertex AI Gemini Embedding 2 wrapper
│   ├── search_engine.py        # Search orchestration (4 search strategies)
│   ├── vector_store.py         # Qdrant client wrapper
│   ├── models.py               # All dataclasses (ContentItem, SearchResult, etc.)
│   ├── content_processor.py    # Content validation and preprocessing
│   └── exceptions.py           # Custom exception types
├── examples/
│   ├── cross_modal_search.py   # Text → image retrieval example
│   ├── embed_modalities.py     # Embedding each modality
│   ├── interleaved_multimodal.py # Combined image+text embedding
│   ├── multilingual_search.py  # Cross-language search
│   └── two_stage_retrieval.py  # Speed/accuracy pipeline
├── tests/
│   ├── test_embedding_service.py
│   ├── test_search_engine.py
│   ├── test_vector_store.py
│   ├── test_api.py
│   ├── test_models.py
│   └── test_interleaved.py
├── config/
│   ├── settings.py             # Environment-based configuration
│   └── logging_config.py
├── demo.py                     # Full capability demo (12 scenarios)
├── .env.example                # Required environment variables
└── pyproject.toml
```

---

## Setup

### Prerequisites
- Python 3.9+
- GCP project with Vertex AI API enabled (or a Google AI Studio API key)
- Qdrant instance (Cloud or local via Docker)

### Install

```bash
pip install -e .
```

### Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
# Google Vertex AI
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
VERTEX_AI_LOCATION=global
VERTEX_AI_API_KEY=your-api-key        # or use ADC

# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

### Quick Start

```python
from multimodal_search.api import MultimodalSearchAPI
from multimodal_search.models import ContentItem

api = MultimodalSearchAPI.from_env()
api.initialize_system()

# Embed and store a text document
item = ContentItem(content_type="text", data="sunset over the ocean", source_id="doc-001")
result = api.embed_content(item, dimension=756, store=True)

# Search
query = ContentItem(content_type="text", data="beach at dusk")
response = api.search(query, limit=5)
for r in response.results:
    print(f"{r.score:.4f}  {r.source_id}")
```

### Run the Full Demo

```bash
python demo.py
```

The demo covers 12 scenarios: system init, text/image embedding, batch processing, two-stage retrieval, cross-modal search, modality filters, score thresholds, multilingual search, interleaved multimodal, RAG knowledge base, and a recommendation engine.

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding Model | Google Gemini Embedding 2 (`gemini-embedding-2-preview`) |
| Vector Database | Qdrant (Cloud or self-hosted) |
| SDK | `google-genai >= 1.0.0`, `qdrant-client >= 1.7.0` |
| Auth | Vertex AI ADC or API key |
| Python | 3.9+ |

---

## Use Cases Demonstrated

- **Semantic search** across mixed-modality content libraries
- **RAG (Retrieval-Augmented Generation)** — index enterprise docs, retrieve relevant context
- **Recommendation engine** — find similar products/content by semantic similarity
- **Cross-lingual search** — query in any language, retrieve in any language
- **Multimodal indexing** — index image+caption pairs as unified vectors
