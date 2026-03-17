# Implementation Plan: Multimodal Search with Vertex AI and Qdrant

## Overview

This implementation plan breaks down the multimodal search system into discrete coding tasks. The system will enable semantic search across text, images, audio, video, and PDF content using Vertex AI's Gemini Embedding 2 model and Qdrant vector database. The implementation follows a bottom-up approach, building core components first, then integrating them into higher-level functionality.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create Python project structure with appropriate directories (src, tests, config)
  - Set up pyproject.toml or requirements.txt with dependencies: google-cloud-aiplatform, qdrant-client, pydantic
  - Create configuration module for Vertex AI and Qdrant settings
  - Set up logging configuration
  - _Requirements: All requirements depend on proper project setup_

- [x] 2. Implement core data models
  - [x] 2.1 Create data model classes using dataclasses or Pydantic
    - Implement ContentItem, EmbeddingMetadata, EmbeddingResult, SearchResult, SearchResponse, ValidationResult, SearchFilters
    - Add type hints and validation for all fields
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.2, 14.1-14.4_

  - [x]* 2.2 Write unit tests for data models
    - Test model instantiation, validation, and serialization
    - Test edge cases for optional fields
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1_

- [x] 3. Implement Content_Processor with validation
  - [x] 3.1 Create ContentProcessor class with validation methods
    - Implement validate_text, validate_image, validate_audio, validate_video, validate_pdf
    - Add format validation (MIME type checking)
    - Add size/duration/page count validation
    - Implement validate_batch for batch validation
    - _Requirements: 1.4, 1.5, 2.2, 2.4, 3.2, 3.3, 4.2, 4.3, 4.4, 5.2, 5.3, 5.4, 6.5, 13.1-13.5_

  - [ ]* 3.2 Write unit tests for Content_Processor
    - Test validation for each modality with valid inputs
    - Test error cases: invalid formats, size exceeded, duration exceeded, page limit exceeded
    - Test batch validation with mixed valid/invalid items
    - _Requirements: 1.4, 1.5, 2.4, 3.3, 4.3, 4.4, 5.3, 5.4, 13.1-13.5_

- [x] 4. Implement Embedding_Service with Vertex AI integration
  - [x] 4.1 Create EmbeddingService class with Vertex AI client initialization
    - Initialize Vertex AI client with project_id and location
    - Set up authentication handling
    - Implement connection validation
    - _Requirements: 1.1, 12.3_

  - [x] 4.2 Implement single-modality embedding methods
    - Implement embed_text using gemini-embedding-2-preview model
    - Implement embed_image for PNG/JPEG images
    - Implement embed_audio for MP3/WAV audio
    - Implement embed_video for MP4/MOV video
    - Implement embed_pdf for PDF documents
    - Add Matryoshka dimension parameter support (128-3072)
    - _Requirements: 1.1-1.7, 2.1, 2.2, 2.5, 3.1, 3.4, 4.1, 4.5, 5.1, 5.5, 10.1-10.3_

  - [x] 4.3 Implement batch embedding functionality
    - Implement embed_batch for mixed modality batching
    - Handle batch size limits (max 6 images per batch)
    - Ensure embeddings returned in same order as input
    - _Requirements: 6.1-6.5_

  - [x] 4.4 Implement multi-dimension embedding for two-stage retrieval
    - Implement embed_with_multiple_dimensions method
    - Generate embeddings at multiple specified dimensions
    - _Requirements: 10.1-10.6, 11.1-11.3_

  - [x] 4.5 Add error handling and retry logic
    - Implement error parsing for Vertex AI API responses
    - Add exponential backoff retry for rate limits
    - Handle authentication, network, and quota errors
    - _Requirements: 12.1-12.5_

  - [x] 4.6 Write unit tests for Embedding_Service
    - Test embedding generation for each modality
    - Test dimension parameter handling
    - Test batch embedding with mixed modalities
    - Test error handling and retry logic
    - Mock Vertex AI API calls for testing
    - _Requirements: 1.1-1.7, 2.1-2.5, 3.1-3.4, 4.1-4.5, 5.1-5.5, 6.1-6.5, 10.1-10.6, 12.1-12.5_

- [x] 5. Checkpoint - Ensure embedding service works correctly
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Vector_Store with Qdrant integration
  - [x] 6.1 Create VectorStore class with Qdrant client initialization
    - Initialize Qdrant client with URL and API key
    - Implement connection validation
    - _Requirements: 7.1, 12.2_

  - [x] 6.2 Implement collection initialization
    - Create initialize_collection method with dimension configuration
    - Set up cosine distance metric
    - Configure named vectors support for two-stage retrieval
    - Add metadata field indexing
    - _Requirements: 7.1, 7.3, 7.5, 11.3_

  - [x] 6.3 Implement embedding storage methods
    - Implement store_embedding for single vector storage
    - Implement store_embedding_with_named_vectors for multi-dimension storage
    - Implement store_batch for efficient batch storage
    - Store metadata alongside vectors (content_type, source_id, timestamp, dimension)
    - Return unique point IDs
    - _Requirements: 7.1-7.5, 10.6_

  - [x] 6.4 Implement vector search methods
    - Implement search with filters and score threshold
    - Implement search_with_named_vector for specific dimension search
    - Implement get_by_id for point retrieval
    - Implement delete_by_id for point deletion
    - _Requirements: 8.1-8.5, 9.1-9.5, 11.3_

  - [x] 6.5 Add error handling for Qdrant operations
    - Handle connection errors, collection not found, dimension mismatches
    - Provide descriptive error messages
    - _Requirements: 12.2_

  - [ ]* 6.6 Write unit tests for Vector_Store
    - Test collection initialization
    - Test single and batch storage operations
    - Test named vectors storage
    - Test search with various filters
    - Test error handling
    - Use Qdrant in-memory mode or mock for testing
    - _Requirements: 7.1-7.5, 9.1-9.5, 11.3, 12.2_

- [x] 7. Implement Search_Engine with cross-modal retrieval
  - [x] 7.1 Create SearchEngine class with service dependencies
    - Initialize with EmbeddingService and VectorStore instances
    - Set up configuration for default parameters
    - _Requirements: 8.1-8.5_

  - [x] 7.2 Implement single-stage search
    - Implement search method with query embedding
    - Apply modality filters using SearchFilters
    - Apply score threshold filtering
    - Return ranked results with metadata
    - _Requirements: 8.1-8.5, 9.1-9.5, 14.1-14.5_

  - [x] 7.3 Implement two-stage retrieval
    - Implement search_two_stage method
    - Stage 1: Search with lower dimension for candidate retrieval
    - Stage 2: Re-rank candidates with higher dimension
    - Return re-ranked results with two-stage metadata
    - _Requirements: 11.1-11.5_

  - [x] 7.4 Implement cross-modal search
    - Implement search_cross_modal method
    - Enable querying with one modality to retrieve any modality
    - Apply target modality filters
    - _Requirements: 8.1-8.5, 9.1-9.5_

  - [x] 7.5 Implement multilingual search support
    - Implement search_multilingual method
    - Support cross-lingual queries (query in one language, retrieve in others)
    - Apply language filters when specified
    - _Requirements: 1.3, 15.1-15.5_

  - [ ]* 7.6 Write unit tests for Search_Engine
    - Test single-stage search with various queries
    - Test two-stage retrieval flow
    - Test cross-modal search scenarios
    - Test multilingual search
    - Test filter application
    - Mock EmbeddingService and VectorStore for unit tests
    - _Requirements: 8.1-8.5, 9.1-9.5, 11.1-11.5, 14.1-14.5, 15.1-15.5_

- [x] 8. Checkpoint - Ensure search functionality works correctly
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement interleaved multimodal input support
  - [x] 9.1 Extend ContentItem model to support interleaved content
    - Add support for multiple modalities in single ContentItem
    - Define structure for interleaved content (e.g., list of modality-content pairs)
    - _Requirements: 16.1-16.5_

  - [x] 9.2 Extend Content_Processor validation for interleaved content
    - Add validate_interleaved method
    - Validate each modality component within interleaved content
    - _Requirements: 16.1-16.5_

  - [x] 9.3 Extend Embedding_Service for interleaved embedding
    - Implement embed_interleaved method
    - Call Vertex AI with interleaved multimodal content
    - Generate unified embedding capturing relationships between modalities
    - _Requirements: 16.1-16.5_

  - [x] 9.4 Update Search_Engine to handle interleaved queries
    - Enable search with interleaved multimodal queries
    - Ensure results match combined semantic meaning
    - _Requirements: 16.5_

  - [x] 9.5 Write unit tests for interleaved multimodal support
    - Test interleaved content validation
    - Test interleaved embedding generation
    - Test search with interleaved queries
    - _Requirements: 16.1-16.5_

- [x] 10. Implement high-level API endpoints
  - [x] 10.1 Create API wrapper class or module
    - Implement embed_content endpoint
    - Implement embed_batch endpoint
    - Implement search endpoint
    - Implement search_two_stage endpoint
    - Implement initialize_system endpoint
    - Wire together all components (ContentProcessor, EmbeddingService, VectorStore, SearchEngine)
    - _Requirements: All requirements_

  - [x] 10.2 Add configuration management
    - Implement VertexAIConfig and QdrantConfig classes
    - Load configuration from environment variables or config files
    - Validate configuration on initialization
    - _Requirements: 1.1, 10.1-10.6_

  - [x] 10.3 Add comprehensive error handling at API level
    - Catch and format all error types (ValidationError, EmbeddingError, StorageError, SearchError)
    - Return user-friendly error responses
    - _Requirements: 12.1-12.5_

  - [ ]* 10.4 Write integration tests for API endpoints
    - Test end-to-end embedding and storage flow
    - Test end-to-end search flow
    - Test two-stage retrieval flow
    - Test error handling across components
    - Use test Vertex AI project and local Qdrant instance
    - _Requirements: All requirements_

- [x] 11. Add example usage and documentation
  - [x] 11.1 Create example scripts
    - Write example for embedding different modalities
    - Write example for cross-modal search
    - Write example for two-stage retrieval
    - Write example for multilingual search
    - Write example for interleaved multimodal input
    - _Requirements: All requirements_

  - [x] 11.2 Add inline code documentation
    - Add docstrings to all classes and methods
    - Document parameters, return types, and exceptions
    - Add usage examples in docstrings
    - _Requirements: All requirements_

- [x] 12. Final checkpoint - Ensure all tests pass and system is functional
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The implementation uses Python as specified in the design document
- Vertex AI authentication should use Application Default Credentials or service account
- Qdrant can be run locally via Docker or used as a cloud service
- Two-stage retrieval requires storing embeddings at multiple dimensions using named vectors
- All modalities share a unified vector space, enabling true cross-modal search
- Matryoshka dimensions: 128, 256, 512, 756, 1024, 1536, 2048, 3072 (default: 3072, recommended for highest quality: 3072, 1536, 768)
- Recommended two-stage configuration: 256 (first stage) + 1024 (second stage)
- Content validation limits: text (≤8192 tokens), images (PNG/JPEG, max 6 per batch), audio (MP3/WAV, no specified limit), video (MP4/MOV, ≤120s), PDF (≤6 pages)
