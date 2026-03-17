# Embeddings

A structured knowledge base and collection of projects covering the full spectrum of embedding techniques and applications.

## Structure

| Folder | Description | Reference |
|---|---|---|
| [`fundamentals/`](./fundamentals/) | Core concepts: vector spaces, similarity metrics, distance functions | [What are Embeddings?](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) |
| [`models/`](./models/) | Embedding models — text, image, audio, multimodal | [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) |
| [`training/`](./training/) | Fine-tuning, contrastive learning, custom training pipelines | [Sentence Transformers Training](https://www.sbert.net/docs/training/overview.html) |
| [`multimodal/`](./multimodal/) | Multimodal embedding projects and PoCs | [Vertex AI Multimodal Embeddings](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings) |
| [`vector_databases/`](./vector_databases/) | Qdrant, Pinecone, Weaviate, pgvector guides and examples | [Qdrant Docs](https://qdrant.tech/documentation/) |
| [`retrieval/`](./retrieval/) | RAG, semantic search, hybrid retrieval patterns | [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/) |
| [`evaluation/`](./evaluation/) | Benchmarks, metrics, and evaluation frameworks | [BEIR Benchmark](https://github.com/beir-cellar/beir) |
| [`production/`](./production/) | Deployment, scaling, monitoring embedding systems | [Qdrant Production Guide](https://qdrant.tech/documentation/guides/deployment/) |
| [`advanced/`](./advanced/) | Quantization, matryoshka, late interaction, sparse-dense hybrid | [Matryoshka Embeddings](https://huggingface.co/blog/matryoshka) |
| [`use_cases/`](./use_cases/) | End-to-end use case implementations | [Semantic Search Tutorial](https://www.pinecone.io/learn/semantic-search/) |
| [`resources/`](./resources/) | Papers, links, and reference materials | [Awesome Embeddings](https://github.com/Hironsan/awesome-embedding-models) |

## Projects

### multimodal/multimodal-search-vertex-qdrant
Multimodal semantic search PoC using Google Vertex AI (Gemini) embeddings and Qdrant vector database.
Supports text, image, and interleaved multimodal queries with cross-modal search.
