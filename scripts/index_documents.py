"""
Document Indexing Script

This script indexes PDF documents into the vector store.
Run this before using the RAG system to index your university notes.

Usage:
    python scripts/index_documents.py --pdf-dir ./data/pdfs
    python scripts/index_documents.py --pdf-dir ./data/pdfs --rebuild
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loader import load_multiple_pdfs, load_pdf, get_document_stats
from chunker import chunk_documents, get_chunk_stats
from embedder import create_embedding_function
from vector_store import (
    create_vectorstore_from_documents,
    initialize_vectorstore,
    add_documents_to_vectorstore,
    get_collection_stats
)


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Index PDF documents into vector store")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        help="Directory containing PDF files (overrides config)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector store from scratch (deletes existing)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    if config is None:
        print("❌ config.yaml not found. Using defaults.")
        config = {
            "data": {
                "pdf_directory": "./data/pdfs",
                "persist_directory": "./chroma_db"
            },
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "embedding": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
    
    # Override with command line args
    pdf_dir = args.pdf_dir or config.get("data", {}).get("pdf_directory", "./data/pdfs")
    persist_dir = config.get("data", {}).get("persist_directory", "./chroma_db")
    
    print("=" * 80)
    print("Document Indexing Pipeline")
    print("=" * 80)
    print(f"PDF Directory: {pdf_dir}")
    print(f"Persist Directory: {persist_dir}")
    print(f"Rebuild: {args.rebuild}")
    print("=" * 80 + "\n")
    
    # Check if PDF directory exists
    if not os.path.exists(pdf_dir):
        print(f"❌ PDF directory not found: {pdf_dir}")
        print("\nPlease create the directory and add your PDF files:")
        print(f"  mkdir -p {pdf_dir}")
        print(f"  cp your_notes.pdf {pdf_dir}/")
        sys.exit(1)
    
    # Handle rebuild
    if args.rebuild and os.path.exists(persist_dir):
        import shutil
        response = input(f"⚠️  Delete existing vector store at {persist_dir}? (yes/no): ")
        if response.lower() == "yes":
            shutil.rmtree(persist_dir)
            print(f"✓ Deleted existing vector store\n")
        else:
            print("✗ Rebuild cancelled")
            sys.exit(0)
    
    # Step 1: Load PDFs
    print("-" * 80)
    print("Step 1: Loading PDFs")
    print("-" * 80)
    
    documents = load_multiple_pdfs(pdf_dir)
    
    if not documents:
        print("❌ No documents loaded. Please add PDF files to the directory.")
        sys.exit(1)
    
    doc_stats = get_document_stats(documents)
    print(f"\nLoaded {doc_stats['total_pages']} pages from {doc_stats['unique_sources']} documents")
    
    # Step 2: Chunk documents
    print("\n" + "-" * 80)
    print("Step 2: Chunking Documents")
    print("-" * 80)
    
    chunk_size = config.get("chunking", {}).get("chunk_size", 1000)
    chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 200)
    
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        clean=True
    )
    
    chunk_stats = get_chunk_stats(chunks)
    print(f"\nChunk Statistics:")
    print(f"  Total chunks: {chunk_stats['total_chunks']}")
    print(f"  Avg chunk size: {chunk_stats['avg_chunk_size']} chars")
    print(f"  Min/Max: {chunk_stats['min_chunk_size']}/{chunk_stats['max_chunk_size']} chars")
    
    # Step 3: Create embeddings and index
    print("\n" + "-" * 80)
    print("Step 3: Creating Embeddings and Indexing")
    print("-" * 80)
    
    embedding_config = config.get("embedding", {})
    use_openai = embedding_config.get("use_openai", False)
    embedding_model = embedding_config.get("model_name", "text-embedding-3-small" if use_openai else "sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"\nEmbedding Configuration:")
    print(f"  Provider: {'OpenAI' if use_openai else 'Local (sentence-transformers)'}")
    print(f"  Model: {embedding_model}")
    
    embedding_function = create_embedding_function(model_name=embedding_model, use_openai=use_openai)
    
    # Check if vector store exists
    db_exists = os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))
    
    if db_exists and not args.rebuild:
        print("\n⚠️  Vector store already exists. Adding documents incrementally...")
        vectorstore = initialize_vectorstore(
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )
        add_documents_to_vectorstore(vectorstore, chunks)
    else:
        print("\nCreating new vector store...")
        vectorstore = create_vectorstore_from_documents(
            documents=chunks,
            embedding_function=embedding_function,
            persist_directory=persist_dir
        )
    
    # Step 4: Verify indexing
    print("\n" + "-" * 80)
    print("Step 4: Verification")
    print("-" * 80)
    
    stats = get_collection_stats(vectorstore)
    print(f"\nVector Store Statistics:")
    print(f"  Total documents: {stats.get('total_documents', 0)}")
    print(f"  Collection name: {stats.get('collection_name', 'unknown')}")
    print(f"  Persist directory: {stats.get('persist_directory', 'unknown')}")
    
    # Test retrieval
    test_query = "What are the main topics?"
    print(f"\nTesting retrieval with query: '{test_query}'")
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"✓ Retrieved {len(results)} documents")
    
    if results:
        print("\nTop result preview:")
        print(f"  Source: {results[0].metadata.get('source', 'unknown')}")
        print(f"  Content: {results[0].page_content[:150]}...")
    
    print("\n" + "=" * 80)
    print("✓ Indexing Complete!")
    print("=" * 80)
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")
    print("\nOr test queries from command line:")
    print("  python rag_pipeline.py ./data/pdfs 'Your question here'")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

