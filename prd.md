# Product Requirements Document (PRD)
## University Notes RAG System

---

## 1. Executive Summary

### 1.1 Project Overview
A Retrieval-Augmented Generation (RAG) system that enables semantic search and question-answering over university lecture notes. The system will process PDF documents, create vector embeddings, and use a Large Language Model (LLM) to generate accurate, context-aware responses based solely on the provided university materials.

### 1.2 Objectives
- Enable fast, semantic search across 10,000–20,000 text chunks from university PDF notes
- Provide accurate, context-grounded answers using OpenAI's GPT models
- Deliver sub-2-second retrieval performance
- Create a modular, maintainable codebase for easy extension and debugging
- Leverage modern LangChain patterns (LCEL, Hub prompts, observability) for clean, production-ready code
- Provide a user-friendly web interface for querying the knowledge base

### 1.3 Key Innovations
This implementation incorporates best practices from university coursework and LangChain ecosystem:

1. **LangChain Expression Language (LCEL):** Declarative chain syntax for cleaner code
2. **LangChain Hub Integration:** Community-optimized prompts instead of ad-hoc prompt engineering
3. **High-Level Abstractions:** `Chroma.from_documents()` and `.as_retriever()` reduce boilerplate
4. **Built-in Evaluation:** Performance metrics (runtime, tokens) tracked from day one
5. **Observability:** LangSmith tracing enabled for debugging and optimization
6. **Separation of Concerns:** Chain initialization separated from execution for efficiency

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐
│   PDF Notes     │
│  (50-150 pages) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  loader.py      │  ← Text Extraction (PDFPlumber/PyMuPDF)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  chunker.py     │  ← Clean & Split (RecursiveCharacterTextSplitter)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  embedder.py    │  ← Generate Embeddings (sentence-transformers)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ vector_store.py │  ← Store & Retrieve (ChromaDB)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ rag_pipeline.py │  ← Query Processing & LLM Response (OpenAI GPT-4)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    app.py       │  ← User Interface (Streamlit/Gradio)
└─────────────────┘
```

### 2.2 Data Flow

**Indexing Phase:**
1. Load PDF → Extract raw text
2. Clean text → Remove headers/footers/page numbers
3. Chunk text → 200-500 word overlapping chunks
4. Generate embeddings → Vector representations
5. Store in ChromaDB → Persistent local storage

**Query Phase:**
1. User submits query via web interface
2. Query is embedded using same model
3. ChromaDB retrieves top 3-5 most similar chunks (cosine similarity)
4. Retrieved chunks + query sent to OpenAI GPT
5. LLM generates response grounded in retrieved context
6. Response displayed to user

### 2.3 Architecture Decisions

**Decision 1: Use LangChain Expression Language (LCEL)**
- **Rationale:** Inspired by university coursework patterns, LCEL provides cleaner, more maintainable code compared to procedural orchestration
- **Impact:** Declarative chain syntax with `|` operator improves readability and enables built-in tracing
- **Trade-off:** Adds LangChain dependency but gains composability and debugging tools

**Decision 2: Leverage LangChain Hub for Prompts**
- **Rationale:** Community-tested prompts (e.g., `rlm/rag-prompt`) are better optimized than custom prompts
- **Impact:** Faster development and improved response quality
- **Trade-off:** Requires network access for prompt loading; mitigated with local fallback

**Decision 3: Use High-Level ChromaDB Abstractions**
- **Rationale:** `Chroma.from_documents()` and `.as_retriever()` simplify embedding and retrieval logic
- **Impact:** Reduces boilerplate code and integrates seamlessly with LCEL chains
- **Trade-off:** Less control over low-level operations; acceptable for this use case

**Decision 4: Implement Evaluation Metrics from Day 1**
- **Rationale:** Inspired by university evaluation patterns, tracking runtime and token usage enables optimization
- **Impact:** Performance monitoring built into core pipeline
- **Trade-off:** Minimal overhead for significant debugging value

**Decision 5: Switch to OpenAI Embeddings (Production Optimization)**
- **Rationale:** Testing revealed local embeddings struggled with academic terminology and abbreviations (e.g., "IO" ≠ "Industrial Organization" semantically)
- **Impact:** 
  - Dramatically improved retrieval accuracy for domain-specific queries
  - Better understanding of abbreviations and technical terms
  - 1536 dimensions vs 384 provides richer semantic representation
- **Trade-off:** 
  - One-time reindexing cost: ~$0.50 for 5,247 chunks
  - Ongoing query cost: ~$0.0001 per query (negligible)
  - Slight API dependency (but acceptable for quality gain)
- **Result:** Essential for university notes with specialized vocabulary

**Decision 6: Increase TOP_K from 5 to 10**
- **Rationale:** Academic content requires higher recall to capture relevant context across multiple related concepts
- **Impact:** Better coverage of topic areas, handles synonym/abbreviation variations
- **Trade-off:** Slightly more context sent to LLM (minimal cost increase)

---

## 3. Technical Stack

### 3.1 Core Technologies

| Component | Technology | Version/Model | Purpose |
|-----------|-----------|---------------|---------|
| PDF Parsing | `PyMuPDF` (fitz) or `pdfplumber` | Latest | Extract text with structure preservation |
| Text Splitting | `langchain.text_splitter` | Latest | RecursiveCharacterTextSplitter for chunking |
| Embedding Model | OpenAI Embeddings | `text-embedding-3-small` | 1536-dim embeddings, superior semantic understanding |
| Vector Database | `ChromaDB` | Latest | Local persistent vector storage |
| LLM | OpenAI API | `gpt-4` or `gpt-3.5-turbo` | Response generation |
| Frontend | `Streamlit` or `Gradio` | Latest | Web interface |
| Programming Language | Python | 3.9+ | Core implementation |

### 3.2 Key Dependencies
```
langchain
langchain-openai
langchain-chroma
langchain-community
langchainhub
sentence-transformers
chromadb
openai
streamlit  # or gradio
PyMuPDF  # or pdfplumber
python-dotenv
```

### 3.3 LangChain Integration Patterns

**LCEL (LangChain Expression Language):**
- Use declarative chain syntax with `|` operator for cleaner, more maintainable code
- Leverage LangChain's built-in components (retrievers, output parsers, runnables)
- Enable automatic tracing and debugging with LangSmith

**Key Benefits:**
- **Composability:** Chain components together seamlessly
- **Observability:** Built-in tracing for debugging
- **Maintainability:** Declarative syntax is easier to understand and modify
- **Performance:** Optimized execution and parallel processing

---

## 4. Functional Requirements

### 4.1 PDF Text Extraction (`loader.py`)

**FR-1.1:** Extract text from PDF files while preserving document structure
- **Input:** Path to PDF file(s)
- **Output:** Raw text with structural markers (headings, lists, paragraphs)
- **Requirements:**
  - Support PDFs with 50-150 pages
  - Preserve formatting hints (bold headings, bullet points)
  - Handle multiple PDFs in batch mode
  - Detect and extract text from both text-based and OCR-needed PDFs

**FR-1.2:** Metadata extraction
- Extract document title, page numbers (for citation)
- Store source file name and page range for each extracted section

### 4.2 Text Preprocessing & Chunking (`chunker.py`)

**FR-2.1:** Text cleaning
- Remove page numbers (patterns: "Page 1", "1", "1/50")
- Remove headers and footers (repeating text across pages)
- Remove excess whitespace while preserving paragraph breaks
- Preserve bulleted/numbered lists structure

**FR-2.2:** Text chunking
- Use `RecursiveCharacterTextSplitter` from LangChain
- Chunk size: 200-500 words (~1000-2500 characters)
- Overlap: 50-100 words (~250-500 characters)
- Split on semantic boundaries (paragraphs, sentences, not mid-word)
- Preserve context by including section headings in chunks

**FR-2.3:** Chunk metadata
- Assign unique ID to each chunk
- Store source document name
- Store page number(s) covered
- Store chunk position in document

### 4.3 Embedding Generation (`embedder.py`)

**FR-3.1:** Generate embeddings for text chunks
- Use OpenAI `text-embedding-3-small` model for production (better accuracy on academic content)
- Fallback to `sentence-transformers/all-MiniLM-L6-v2` if needed (local, free)
- Batch processing for efficiency (32-64 chunks per batch)
- Output: 1536-dimensional dense vectors (OpenAI) or 384-dim (local)

**FR-3.2:** Query embedding
- Use same model for query embedding to ensure embedding space consistency
- Single query embedding generation (<100ms)

### 4.4 Vector Storage & Retrieval (`vector_store.py`)

**FR-4.1:** ChromaDB setup
- Initialize persistent local ChromaDB instance
- Create collection with cosine similarity metric
- Store embeddings with associated metadata

**FR-4.2:** Document indexing
- Batch insert embeddings with metadata
- Support incremental indexing (add new documents without reindexing all)
- Handle duplicate detection (skip already-indexed documents)

**FR-4.3:** Semantic search
- Accept query embedding as input
- Return top 10 most similar chunks based on cosine similarity (increased from 5 for better recall on academic queries)
- Include similarity scores and metadata in results
- Retrieval time: <2 seconds for 10,000-20,000 chunks
- Note: Higher TOP_K essential for handling abbreviations and domain-specific terminology

**FR-4.4:** Persistence
- Save database to disk automatically
- Load existing database on startup
- Support database reset/rebuild functionality

### 4.5 RAG Pipeline (`rag_pipeline.py`)

**FR-5.1:** LCEL Chain initialization
- Initialize RAG chain using LangChain Expression Language (LCEL)
- Configure retriever from vectorstore with `.as_retriever()` method
- Load prompt template from LangChain Hub or use custom prompt
- Chain components declaratively using `|` operator
- Support chain composition: `retriever | format_docs | prompt | llm | parser`

**FR-5.2:** Query processing
- Accept natural language query from user
- Pass query through LCEL chain using `.invoke(question)`
- Retriever automatically handles query embedding and similarity search
- Format retrieved documents using helper function (`format_docs`)

**FR-5.3:** LLM response generation
- Use OpenAI GPT-4 (or GPT-3.5-turbo) via `ChatOpenAI`
- Load system prompt from LangChain Hub (`rlm/rag-prompt`) as default
- Support custom prompts for university-specific requirements
- Include retrieved chunks in prompt via chain context
- Use `StrOutputParser()` for clean string output

**FR-5.4:** Response formatting and evaluation
- Return generated answer as string
- Include source citations (document name, page numbers) in response
- Implement `evaluate_rag()` function to track:
  - Runtime (seconds)
  - Estimated token usage
  - Output quality
- Handle cases where no relevant context is found

**FR-5.5:** Error handling and observability
- Handle API failures gracefully
- Validate API key on startup
- Implement retry logic for transient failures
- Enable LangSmith tracing for debugging (`LANGCHAIN_TRACING_V2=true`)
- Log chain execution steps for troubleshooting

### 4.6 Web Interface (`app.py`)

**FR-6.1:** User interface (Streamlit/Gradio)
- Text input box for queries
- Submit button to trigger search
- Display generated response with formatting
- Display source citations (clickable or expandable)
- Show retrieved chunks (optional, for debugging)

**FR-6.2:** Admin features
- Upload new PDF documents
- Trigger reindexing
- View indexing status (number of documents, chunks)
- Clear/reset database

---

## 5. Non-Functional Requirements

### 5.1 Performance

**NFR-1.1:** Retrieval speed
- Query processing (embedding + retrieval): <2 seconds
- LLM response generation: <10 seconds (OpenAI API dependent)
- Support for 10,000–20,000 chunks minimum

**NFR-1.2:** Scalability
- Handle up to 50 PDF documents (50-150 pages each)
- Efficient batch processing during indexing
- Memory-efficient embedding generation

### 5.2 Reliability

**NFR-2.1:** Data persistence
- ChromaDB must persist embeddings across sessions
- No data loss on application restart

**NFR-2.2:** Error handling
- Graceful degradation on API failures
- Clear error messages for users
- Logging for debugging

### 5.3 Maintainability

**NFR-3.1:** Code quality
- Modular architecture with clear separation of concerns
- Comprehensive docstrings for all functions/classes
- Type hints where applicable
- Clear comments for complex logic

**NFR-3.2:** Testability
- Unit tests for each module
- Integration test for full pipeline
- Performance benchmarking scripts

### 5.4 Usability

**NFR-4.1:** User experience
- Intuitive web interface
- Clear instructions and feedback
- Response time indicators
- Helpful error messages

**NFR-4.2:** Developer experience
- Easy setup with clear README
- Configuration via environment variables or config file
- Simple commands to run indexing and query server

---

## 6. Module Specifications

### 6.1 `loader.py`

**Purpose:** Extract text from PDF files

**Key Functions:**
- `load_pdf(pdf_path: str) -> List[Dict]`: Extract text with page numbers
- `load_multiple_pdfs(pdf_dir: str) -> List[Dict]`: Batch load PDFs from directory
- `preserve_structure(raw_text: str) -> str`: Maintain formatting hints

**Dependencies:** `PyMuPDF` or `pdfplumber`

### 6.2 `chunker.py`

**Purpose:** Clean and chunk text

**Key Functions:**
- `clean_text(text: str) -> str`: Remove headers, footers, page numbers
- `chunk_text(text: str, metadata: Dict) -> List[Dict]`: Split into chunks with metadata
- `create_text_splitter() -> RecursiveCharacterTextSplitter`: Configure splitter

**Dependencies:** `langchain`, `re`

### 6.3 `embedder.py`

**Purpose:** Generate embeddings

**Key Functions:**
- `load_embedding_model() -> SentenceTransformer`: Load model
- `embed_chunks(chunks: List[str]) -> np.ndarray`: Generate embeddings
- `embed_query(query: str) -> np.ndarray`: Embed single query

**Dependencies:** `sentence-transformers`

### 6.4 `vector_store.py`

**Purpose:** Vector database operations using LangChain abstractions

**Key Functions:**
- `initialize_vectorstore(persist_dir: str) -> Chroma`: Setup ChromaDB with LangChain wrapper
- `create_vectorstore_from_documents(documents: List[Document], embedding_function) -> Chroma`: High-level document indexing using `Chroma.from_documents()`
- `get_retriever(vectorstore: Chroma, search_kwargs: Dict) -> VectorStoreRetriever`: Return LangChain retriever interface
- `add_documents_incremental(vectorstore, documents)`: Add new documents to existing store
- `get_collection_stats(vectorstore) -> Dict`: Get DB statistics

**Implementation Notes:**
- Use `langchain_chroma.Chroma` wrapper instead of raw ChromaDB client
- Leverage `Chroma.from_documents()` for automatic embedding + insertion
- Return retriever via `.as_retriever()` for seamless LCEL integration
- Support both OpenAI embeddings and sentence-transformers

**Dependencies:** `langchain_chroma`, `chromadb`, `langchain_openai` or `langchain_community`

### 6.5 `rag_pipeline.py`

**Purpose:** End-to-end RAG inference using LCEL chains

**Key Functions:**
- `initialize_rag_chain(vectorstore: Chroma, llm_model: str = "gpt-4") -> RunnableSequence`: Setup LCEL chain
- `format_docs(docs: List[Document]) -> str`: Format retrieved documents for context (joins with `\n\n`)
- `query(rag_chain, question: str) -> str`: Execute RAG chain with question
- `evaluate_rag(rag_chain, question: str) -> Dict`: Query with performance metrics (runtime, token estimate)
- `get_prompt_template() -> ChatPromptTemplate`: Load prompt from hub or use custom

**LCEL Chain Architecture:**
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**Implementation Notes:**
- Use LangChain Hub (`hub.pull("rlm/rag-prompt")`) for battle-tested prompts
- Implement custom prompts as fallback for university-specific requirements
- Chain components using `|` operator for clean, declarative syntax
- Use `RunnablePassthrough()` to pass query through unchanged
- Include `evaluate_rag()` function for performance tracking (runtime, estimated tokens)

**Dependencies:** `langchain`, `langchain_openai`, `langchainhub`, all previous modules

### 6.6 `app.py`

**Purpose:** Web interface with RAG chain management

**Key Functions:**
- `initialize_rag_system() -> RunnableSequence`: Setup RAG chain on app startup (cached)
- `main()`: Streamlit app entry point
- `render_query_interface()`: Display query UI with input box and submit button
- `render_results(result: Dict)`: Display response with metrics (runtime, tokens)
- `render_admin_interface()`: Display admin UI for document management
- `handle_query(rag_chain, question: str)`: Process query using `evaluate_rag()`

**Implementation Notes:**
- Initialize RAG chain once on app startup, not per query
- Cache chain initialization using `@st.cache_resource` (Streamlit)
- Display performance metrics alongside results (runtime, estimated tokens)
- Separate chain initialization from query execution for efficiency
- Enable debug mode to show retrieved chunks and trace information

**Dependencies:** `streamlit` or `gradio`, `rag_pipeline`, `langchain`

---

## 7. Testing Requirements

### 7.1 Unit Tests

**Test Scripts:**
- `test_loader.py`: Verify PDF extraction works correctly
- `test_chunker.py`: Validate cleaning and chunking logic
- `test_embedder.py`: Check embedding generation
- `test_vector_store.py`: Test CRUD operations on ChromaDB
- `test_rag_pipeline.py`: End-to-end pipeline test

### 7.2 Integration Tests

**`test_indexing.py`:**
- Index a sample PDF (5-10 pages)
- Verify chunks are created correctly
- Verify embeddings are stored in ChromaDB
- Check metadata integrity

**`test_query.py`:**
- Submit test queries
- Verify retrieval returns relevant chunks
- Verify LLM response is generated
- Validate citations are included

### 7.3 Performance Tests

**`benchmark_retrieval.py`:**
- Measure retrieval time with 10k, 15k, 20k chunks
- Verify <2s retrieval time requirement
- Plot retrieval time vs. database size

**`benchmark_e2e.py`:**
- Measure end-to-end query processing time using `evaluate_rag()` function
- Track metrics:
  - Runtime (seconds)
  - Estimated token usage (`len(result) // 4`)
  - Retrieval accuracy
- Identify bottlenecks (embedding, retrieval, LLM)
- Generate performance report with statistics

**Evaluation Function Pattern:**
```python
def evaluate_rag(rag_chain, question: str) -> Dict:
    """Evaluate RAG chain with performance metrics."""
    start_time = time.time()
    result = rag_chain.invoke(question)
    end_time = time.time()
    
    return {
        "output": result,
        "estimated_tokens": len(result) // 4,
        "runtime": end_time - start_time
    }
```

---

## 8. Configuration

### 8.1 Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo

# LangChain Configuration (Optional but recommended)
LANGCHAIN_TRACING_V2=true  # Enable LangSmith tracing for debugging
LANGCHAIN_API_KEY=<your-langchain-api-key>  # Optional: for LangSmith
LANGCHAIN_PROJECT=university-rag  # Optional: project name in LangSmith

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Storage Configuration
CHROMA_PERSIST_DIR=./chroma_db
PDF_INPUT_DIR=./data/pdfs

# Chunking Configuration
CHUNK_SIZE=1000  # characters (~200 words)
CHUNK_OVERLAP=200  # characters (~50 words)

# Retrieval Configuration
TOP_K=5  # number of chunks to retrieve
```

### 8.2 Configuration File (`config.yaml`)

```yaml
data:
  pdf_directory: "./data/pdfs"
  persist_directory: "./chroma_db"

chunking:
  chunk_size: 1000
  chunk_overlap: 200
  separators: ["\n\n", "\n", ". ", " ", ""]

embedding:
  use_openai: true  # Use OpenAI embeddings for production accuracy
  model_name: "text-embedding-3-small"
  fallback_model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32

retrieval:
  top_k: 10  # Increased from 5 for better recall on academic content
  similarity_threshold: 0.7
  search_type: "similarity"  # or "mmr" for maximum marginal relevance

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.1
  max_tokens: 500
  
prompt:
  use_hub: true  # Use LangChain Hub prompts
  hub_prompt: "rlm/rag-prompt"  # Default hub prompt
  custom_system_prompt: "You are a helpful assistant that answers questions based solely on the provided context from university lecture notes. If the context does not contain the answer, say 'I cannot find this information in the provided notes.'"

langchain:
  tracing: true  # Enable LangSmith tracing
  project: "university-rag"  # LangSmith project name

frontend:
  framework: "streamlit"
  port: 8501
  title: "University Notes RAG System"
```

---

## 9. Success Metrics

### 9.1 Performance Metrics
- **Retrieval Speed:** <2s for databases with 10k-20k chunks
- **End-to-End Query Time:** <15s (including LLM generation)
- **Indexing Speed:** >100 chunks/second

### 9.2 Quality Metrics
- **Retrieval Accuracy:** Top-5 results contain answer >90% of the time (based on test queries)
- **Response Quality:** LLM responses are factually grounded in retrieved context
- **Citation Accuracy:** Sources cited are actually used in response generation

### 9.3 Usability Metrics
- **Setup Time:** <5 minutes to install and run
- **Query Success Rate:** >95% of queries return valid responses (not errors)

---

## 10. Assumptions & Constraints

### 10.1 Assumptions
- PDFs are text-based (not scanned images requiring OCR)
- Users have valid OpenAI API key with sufficient credits
- Local machine has sufficient storage (10k chunks ≈ 100MB ChromaDB)
- Python 3.9+ environment is available

### 10.2 Constraints
- System is for personal/educational use (not production-scale)
- Local deployment only (not cloud-hosted)
- English language only (model supports, but not optimized for multilingual)
- OpenAI API rate limits apply

### 10.3 Out of Scope (Future Enhancements)
- Multi-user authentication
- Cloud deployment
- Mobile app interface
- Real-time collaborative querying
- Advanced analytics dashboard
- Fine-tuning custom embedding models
- Support for non-PDF formats (Word, PPT, HTML)

---

## 11. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| OpenAI API costs exceed budget | High | Medium | Add cost tracking; use GPT-3.5-turbo; implement caching |
| Retrieval accuracy is poor | High | Medium | Tune chunk size; improve cleaning; experiment with embedding models |
| ChromaDB performance degrades at scale | Medium | Low | Monitor performance; consider FAISS if needed |
| PDF extraction fails on certain formats | Medium | Medium | Test multiple libraries; add manual preprocessing step |
| LLM hallucinates despite context | Medium | Low | Improve system prompt; implement confidence scoring |

---

## 12. Development Roadmap

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up project structure and dependencies
- [ ] Implement `loader.py` with PDF extraction
- [ ] Implement `chunker.py` with text cleaning
- [ ] Write unit tests for loader and chunker

### Phase 2: Vector Storage (Week 1-2)
- [ ] Implement `embedder.py` with sentence-transformers
- [ ] Implement `vector_store.py` with ChromaDB
- [ ] Test indexing pipeline with sample PDFs
- [ ] Benchmark retrieval performance

### Phase 3: RAG Pipeline (Week 2)
- [ ] Implement `rag_pipeline.py` with OpenAI integration
- [ ] Test end-to-end query flow
- [ ] Optimize prompt engineering for accurate responses
- [ ] Add citation generation

### Phase 4: Frontend (Week 2-3)
- [ ] Implement `app.py` with Streamlit
- [ ] Create query interface
- [ ] Add admin features (upload, reindex)
- [ ] Polish UI/UX

### Phase 5: Testing & Optimization (Week 3)
- [ ] Write comprehensive test suite
- [ ] Run performance benchmarks
- [ ] Optimize bottlenecks
- [ ] Write documentation (README, API docs)

### Phase 6: Deployment (Week 3-4)
- [ ] Deploy to personal server or local machine
- [ ] Integrate with website (API endpoint or iframe)
- [ ] Monitor usage and performance
- [ ] Gather user feedback and iterate

---

## 13. Deployment & Integration

### 13.1 Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=<your-key>

# Index PDFs
python scripts/index_pdfs.py --pdf-dir ./data/pdfs

# Run frontend
streamlit run app.py
```

### 13.2 Website Integration
- **Option A:** Run Streamlit app on local server, embed in iframe on website
- **Option B:** Create FastAPI REST API, call from website frontend
- **Option C:** Deploy to Streamlit Cloud, link from website

---

## 14. Documentation Requirements

### 14.1 README.md
- Project overview
- Installation instructions
- Quick start guide
- Configuration options
- Troubleshooting

### 14.2 API Documentation
- Docstrings for all public functions
- Usage examples
- Parameter descriptions
- Return value specifications

### 14.3 User Guide
- How to add new PDFs
- How to query the system
- Interpreting results
- Best practices for queries

---

## 15. Appendix

### 15.1 Example Queries
- "What is the definition of gradient descent?"
- "Explain the difference between supervised and unsupervised learning"
- "What are the key points from the lecture on neural networks?"
- "Summarize the chapter on linear algebra"

### 15.2 Example System Prompt
```
You are an intelligent assistant helping a university student understand their lecture notes.

You will be provided with relevant excerpts from the student's notes. Answer the question using ONLY the information provided in these excerpts. If the excerpts do not contain enough information to answer the question, clearly state that.

Always cite the source of your information by mentioning the document name and page number.

Be concise but thorough. Use bullet points for lists and structure your answer clearly.
```

### 15.3 LangChain Patterns Reference

**LCEL Chain Pattern (from university coursework):**
```python
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize components
llm = ChatOpenAI(model="gpt-4o-mini")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Create retriever and prompt
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

# Define format function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build LCEL chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Evaluate with metrics
def evaluate_rag(rag_chain, question):
    start_time = time.time()
    result = rag_chain.invoke(question)
    end_time = time.time()
    
    print(f"Output: {result}")
    print(f"Estimated token usage: {len(result) // 4}")
    print(f"Runtime: {end_time - start_time:.2f} seconds")

# Execute
evaluate_rag(rag_chain, "What is Task Decomposition?")
```

**Key Advantages of This Pattern:**
1. **Declarative Syntax:** Easy to read and understand data flow
2. **Composability:** Chain components can be reused and recombined
3. **Built-in Tracing:** Automatic debugging with LangSmith
4. **Performance Metrics:** Easy to track runtime and token usage
5. **Community Prompts:** Leverage optimized prompts from LangChain Hub

### 15.4 References
- [LangChain Documentation](https://docs.langchain.com/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [LangChain Hub](https://smith.langchain.com/hub)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangSmith Tracing](https://docs.smith.langchain.com/)

---

**Document Version:** 2.1  
**Last Updated:** October 19, 2025  
**Author:** Jonas  
**Status:** Production - Optimized

**Changelog:**
- **v2.1:** **PRODUCTION OPTIMIZATION**
  - Switched from local sentence-transformers to OpenAI embeddings (`text-embedding-3-small`) for significantly improved retrieval accuracy on academic content
  - Increased TOP_K from 5 to 10 for better recall on specialized queries (abbreviations, domain-specific terms)
  - Rationale: Local embeddings struggled with domain-specific terminology and abbreviations (e.g., "IO" vs "Industrial Organization"). OpenAI embeddings provide 1536 dimensions (vs 384) with better semantic understanding at minimal cost (~$0.50 one-time reindexing)
- **v2.0:** Integrated LangChain Expression Language (LCEL) patterns, LangChain Hub prompts, higher-level ChromaDB abstractions, evaluation metrics tracking, and LangSmith tracing capabilities
- **v1.0:** Initial PRD with modular architecture

