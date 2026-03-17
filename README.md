# Embeddings

A structured knowledge base and collection of projects covering the full spectrum of embedding techniques and applications.

## Structure

| Folder | Description |
|---|---|
| [`fundamentals/`](./fundamentals/) | Core concepts: vector spaces, similarity metrics, distance functions |
| [`models/`](./models/) | Embedding models — text, image, audio, multimodal |
| [`training/`](./training/) | Fine-tuning, contrastive learning, custom training pipelines |
| [`multimodal/`](./multimodal/) | Multimodal embedding projects and PoCs |
| [`vector_databases/`](./vector_databases/) | Qdrant, Pinecone, Weaviate, pgvector guides and examples |
| [`retrieval/`](./retrieval/) | RAG, semantic search, hybrid retrieval patterns |
| [`evaluation/`](./evaluation/) | Benchmarks, metrics, and evaluation frameworks |
| [`production/`](./production/) | Deployment, scaling, monitoring embedding systems |
| [`advanced/`](./advanced/) | Quantization, matryoshka, late interaction, sparse-dense hybrid |
| [`use_cases/`](./use_cases/) | End-to-end use case implementations |
| [`resources/`](./resources/) | Papers, links, and reference materials |

## Projects

### multimodal/multimodal-search-vertex-qdrant
Multimodal semantic search PoC using Google Vertex AI (Gemini) embeddings and Qdrant vector database.
Supports text, image, and interleaved multimodal queries with cross-modal search.
