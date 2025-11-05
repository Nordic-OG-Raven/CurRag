# University Notes RAG System

A Retrieval-Augmented Generation (RAG) system for querying university lecture notes using LangChain, ChromaDB, and OpenAI.

## 🎯 Features

- **Semantic Search**: Query your notes using natural language
- **LangChain LCEL**: Modern, declarative chain composition
- **ChromaDB**: Local, persistent vector storage  
- **OpenAI Embeddings**: text-embedding-3-small (1536-dim, optimized for academic content)
- **OpenAI GPT-4**: High-quality response generation
- **Streamlit UI**: User-friendly web interface
- **Observability**: Built-in LangSmith tracing support

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key
- (Optional) LangChain API key for tracing

## 🚀 Quick Start

### Automated Setup (Recommended)

```bash
cd /Users/jonas/CurRag
./setup.sh
```

This will create the virtual environment, install dependencies, and set up the project structure.

### Manual Setup

#### 1. Installation

```bash
# Navigate to the project directory
cd /Users/jonas/CurRag

# Create virtual environment (use python3 on macOS)
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Always activate the virtual environment before running any scripts!

### 2. Configuration

Copy the example environment file and add your API keys:

```bash
cp env.example .env
```

Edit `.env` and add your keys:

```bash
OPENAI_API_KEY=your-openai-api-key-here
LANGCHAIN_TRACING_V2=true  # Optional, for debugging
LANGCHAIN_API_KEY=your-langchain-api-key  # Optional
```

### 3. Add Your PDFs

Place your university notes (PDF format) in the `data/pdfs` directory:

```bash
mkdir -p data/pdfs
cp ~/Downloads/lecture_notes.pdf data/pdfs/
```

### 4. Index Your Documents

Run the indexing script to process and embed your PDFs:

```bash
python scripts/index_documents.py --pdf-dir ./data/pdfs
```

This will:
- Load all PDFs from the directory
- Clean and chunk the text (1000 chars, 200 overlap)
- Generate embeddings (sentence-transformers/all-MiniLM-L6-v2)
- Store in ChromaDB (./chroma_db)

**To rebuild from scratch:**
```bash
python scripts/index_documents.py --pdf-dir ./data/pdfs --rebuild
```

### 5. Run the Web App

**Make sure your virtual environment is activated first!**

```bash
# Activate if not already active
source .venv/bin/activate

# Launch the Streamlit interface
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start querying!

## 📚 Usage Examples

### Web Interface (Recommended)

1. Run `streamlit run app.py`
2. Enter your question in the text box
3. View the answer with source citations
4. Check performance metrics (runtime, tokens)

### Command Line

**Activate virtual environment first:**
```bash
source .venv/bin/activate
```

Test a single query:

```bash
python rag_pipeline.py ./data/pdfs "What is gradient descent?"
```

Batch evaluation:

```bash
python rag_pipeline.py ./data/pdfs "Your question" --batch
```

### Python API

**In your Python script or notebook (with .venv activated):**

```python
from embedder import create_embedding_function
from vector_store import initialize_vectorstore
from rag_pipeline import initialize_rag_chain, evaluate_rag

# Initialize components
embedding_function = create_embedding_function()
vectorstore = initialize_vectorstore(
    persist_directory="./chroma_db",
    embedding_function=embedding_function
)

# Create RAG chain
rag_chain = initialize_rag_chain(
    vectorstore=vectorstore,
    llm_model="gpt-4"
)

# Query with metrics
result = evaluate_rag(rag_chain, "What is machine learning?")
print(result['output'])
print(f"Runtime: {result['runtime']:.2f}s")
```

## 🧪 Testing

### Test Queries

Run the test suite to validate system performance:

```bash
python tests/test_query.py
```

Test with custom questions:

```bash
python tests/test_query.py --questions "What is AI?" "Explain ML"
```

### Benchmark Retrieval

Measure retrieval performance:

```bash
python tests/benchmark_retrieval.py --queries 100
```

Generate performance plot:

```bash
python tests/benchmark_retrieval.py --queries 100 --plot
```

## 📁 Project Structure

```
CurRag/
├── README.md                 # This file
├── prd.md                    # Product Requirements Document
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration settings
├── env.example              # Environment variables template
│
├── loader.py                # PDF text extraction
├── chunker.py               # Text cleaning and chunking
├── embedder.py              # Embedding generation
├── vector_store.py          # ChromaDB operations
├── rag_pipeline.py          # LCEL RAG chain
├── app.py                   # Streamlit web interface
│
├── scripts/
│   └── index_documents.py   # Document indexing script
│
├── tests/
│   ├── test_query.py        # Query testing
│   └── benchmark_retrieval.py  # Performance benchmarking
│
├── data/
│   └── pdfs/                # Your PDF files go here
│
└── chroma_db/               # Vector database (generated)
```

## ⚙️ Configuration

Edit `config.yaml` to customize settings:

```yaml
data:
  pdf_directory: "./data/pdfs"
  persist_directory: "./chroma_db"

chunking:
  chunk_size: 1000           # Characters per chunk
  chunk_overlap: 200         # Overlap between chunks

retrieval:
  top_k: 5                   # Documents to retrieve
  search_type: "similarity"  # or "mmr"

llm:
  model: "gpt-4"             # or "gpt-3.5-turbo"
  temperature: 0.1
  max_tokens: 500

prompt:
  use_hub: true              # Use LangChain Hub prompts
  hub_prompt: "rlm/rag-prompt"
```

## 🎓 Key Patterns

This implementation uses modern LangChain patterns:

### LCEL Chain Composition

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Performance Evaluation

```python
def evaluate_rag(rag_chain, question):
    start_time = time.time()
    result = rag_chain.invoke(question)
    end_time = time.time()
    
    return {
        "output": result,
        "runtime": end_time - start_time,
        "estimated_tokens": len(result) // 4
    }
```

### High-Level Abstractions

```python
# Automatic embedding + insertion
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)

# Retriever interface for chains
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

## 🔍 Debugging

Enable LangSmith tracing for detailed debugging:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-langchain-key
export LANGCHAIN_PROJECT=university-rag
```

View traces at [smith.langchain.com](https://smith.langchain.com)

## 📊 Performance Metrics

Expected performance with 10k-20k chunks:

- **End-to-End Query**: ~15-30 seconds (including LLM)
- **Indexing Speed**: >100 chunks/second

Benchmark your system:

```bash
python tests/benchmark_retrieval.py --queries 100
```

## 🐛 Troubleshooting

### "Vector store not found"
Run indexing first: `python scripts/index_documents.py --pdf-dir ./data/pdfs`

### "OPENAI_API_KEY not set"
Add your API key to `.env` file or export it:
```bash
export OPENAI_API_KEY=your-key-here
```

### Slow retrieval (>2s)
- Check database size: may need optimization for very large datasets
- Reduce `top_k` in config.yaml
- Consider using smaller embedding model

### Poor answer quality
- Increase `top_k` to retrieve more context
- Adjust `chunk_size` and `chunk_overlap` in config.yaml
- Try different prompt templates
- Use GPT-4 instead of GPT-3.5-turbo

### Import errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 🌐 Website Integration

### Option 1: Iframe Embed

Run Streamlit on a server and embed:

```html
<iframe src="http://your-server:8501" width="100%" height="800px"></iframe>
```

### Option 2: FastAPI REST API

Create an API endpoint:

```python
from fastapi import FastAPI
from rag_pipeline import initialize_rag_chain, query

app = FastAPI()
rag_chain = initialize_rag_chain(vectorstore)

@app.post("/query")
async def query_notes(question: str):
    result = query(rag_chain, question)
    return {"answer": result}
```

### Option 3: Streamlit Cloud

Deploy to [share.streamlit.io](https://share.streamlit.io) and link from your site.

## 📚 Additional Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [LangChain Expression Language](https://python.langchain.com/docs/expression_language/)
- [LangChain Hub](https://smith.langchain.com/hub)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API](https://platform.openai.com/docs/)

## 📝 Development

To contribute or extend:

1. Read `prd.md` for architecture details
2. Each module has docstrings and examples
3. Run tests before committing
4. Follow the LCEL patterns for new chains

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Inspired by university coursework on RAG systems
- Built with LangChain, ChromaDB, and OpenAI
- Follows modern LCEL patterns for maintainability

---

**Version**: 1.0  
**Last Updated**: October 19, 2025  
**Author**: Jonas

For questions or issues, please refer to the documentation or create an issue.

