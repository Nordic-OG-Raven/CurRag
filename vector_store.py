"""
Vector Store Module

This module handles vector database operations using ChromaDB with LangChain abstractions.
Implements high-level patterns: Chroma.from_documents() and .as_retriever()
"""

import os
from typing import List, Optional, Dict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


def initialize_vectorstore(
    persist_directory: str = "./chroma_db",
    collection_name: str = "university_notes",
    embedding_function: Optional[Embeddings] = None
) -> Chroma:
    """
    Initialize or load an existing ChromaDB vector store.
    
    If the persist_directory exists, loads the existing database.
    Otherwise, creates a new one.
    
    Args:
        persist_directory (str): Directory to persist the vector store
        collection_name (str): Name of the collection
        embedding_function (Embeddings): Embedding function to use
        
    Returns:
        Chroma: Initialized vector store
    """
    if embedding_function is None:
        from embedder import create_embedding_function
        embedding_function = create_embedding_function()
    
    # Create persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Check if database already exists
    db_exists = os.path.exists(os.path.join(persist_directory, "chroma.sqlite3"))
    
    if db_exists:
        print(f"✓ Loading existing vector store from {persist_directory}")
    else:
        print(f"✓ Creating new vector store at {persist_directory}")
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    
    return vectorstore


def create_vectorstore_from_documents(
    documents: List[Document],
    embedding_function: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "university_notes"
) -> Chroma:
    """
    Create a new vector store from documents using high-level Chroma.from_documents().
    
    This method automatically:
    - Generates embeddings for all documents
    - Stores them in ChromaDB
    - Persists to disk
    
    Args:
        documents (List[Document]): Documents to index
        embedding_function (Embeddings): Embedding function
        persist_directory (str): Directory to persist the vector store
        collection_name (str): Name of the collection
        
    Returns:
        Chroma: Vector store with indexed documents
    """
    if not documents:
        raise ValueError("No documents provided to index")
    
    print(f"Creating vector store from {len(documents)} documents...")
    
    # Create persist directory
    os.makedirs(persist_directory, exist_ok=True)
    
    # Use Chroma.from_documents for automatic embedding + insertion
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print(f"✓ Vector store created with {len(documents)} documents")
    
    return vectorstore


def add_documents_to_vectorstore(
    vectorstore: Chroma,
    documents: List[Document]
) -> List[str]:
    """
    Add new documents to an existing vector store incrementally.
    
    Args:
        vectorstore (Chroma): Existing vector store
        documents (List[Document]): New documents to add
        
    Returns:
        List[str]: IDs of added documents
    """
    if not documents:
        print("⚠ No documents to add")
        return []
    
    print(f"Adding {len(documents)} documents to vector store...")
    
    # Add documents (embeddings generated automatically)
    ids = vectorstore.add_documents(documents)
    
    print(f"✓ Added {len(ids)} documents")
    
    return ids


def get_retriever(
    vectorstore: Chroma,
    search_type: str = "similarity",
    search_kwargs: Optional[Dict] = None
) -> VectorStoreRetriever:
    """
    Get a retriever from the vector store for use in LCEL chains.
    
    Args:
        vectorstore (Chroma): Vector store
        search_type (str): Type of search ("similarity" or "mmr")
        search_kwargs (Dict, optional): Search parameters
            - k: Number of documents to retrieve (default: 5)
            - score_threshold: Minimum similarity score (for similarity_score_threshold)
            
    Returns:
        VectorStoreRetriever: Retriever for use in chains
    """
    if search_kwargs is None:
        search_kwargs = {"k": 5}
    
    print(f"Creating retriever (type={search_type}, k={search_kwargs.get('k', 5)})")
    
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    return retriever


def hybrid_search(
    vectorstore: Chroma,
    query: str,
    k: int = 10
) -> List[Document]:
    """
    Perform hybrid search combining semantic similarity with keyword matching.
    
    This helps when embeddings miss exact terminology (e.g., "Industrial Organization"
    as a course name vs generic "industry" and "organization" terms).
    
    Args:
        vectorstore (Chroma): Vector store
        query (str): Query text
        k (int): Number of results to return
        
    Returns:
        List[Document]: Most relevant documents (hybrid scored)
    """
    # Get semantic results
    semantic_results = vectorstore.similarity_search_with_score(query, k=k*2)
    
    # Simple keyword boosting: if query terms appear in source filename, boost score
    query_terms = set(query.lower().split())
    
    rescored_results = []
    for doc, score in semantic_results:
        source = doc.metadata.get('source', '').lower()
        
        # Boost if source filename contains query terms
        keyword_boost = 0
        for term in query_terms:
            if term in source:
                keyword_boost += 0.5  # Significant boost for filename match
        
        # Boost if query terms appear in content
        content_lower = doc.page_content.lower()
        for term in query_terms:
            if term in content_lower:
                keyword_boost += 0.1
        
        # Lower score is better in ChromaDB (distance), so subtract boost
        adjusted_score = score - keyword_boost
        rescored_results.append((doc, adjusted_score))
    
    # Sort by adjusted score and return top k
    rescored_results.sort(key=lambda x: x[1])
    return [doc for doc, _ in rescored_results[:k]]


def similarity_search(
    vectorstore: Chroma,
    query: str,
    k: int = 5
) -> List[Document]:
    """
    Perform similarity search on the vector store.
    
    Args:
        vectorstore (Chroma): Vector store
        query (str): Query text
        k (int): Number of results to return
        
    Returns:
        List[Document]: Most similar documents
    """
    results = vectorstore.similarity_search(query, k=k)
    return results


def similarity_search_with_score(
    vectorstore: Chroma,
    query: str,
    k: int = 5
) -> List[tuple[Document, float]]:
    """
    Perform similarity search with relevance scores.
    
    Args:
        vectorstore (Chroma): Vector store
        query (str): Query text
        k (int): Number of results to return
        
    Returns:
        List[tuple[Document, float]]: Documents with similarity scores
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def get_collection_stats(vectorstore: Chroma) -> Dict:
    """
    Get statistics about the vector store collection.
    
    Args:
        vectorstore (Chroma): Vector store
        
    Returns:
        Dict: Statistics including document count, sources, etc.
    """
    try:
        # Get the underlying collection
        collection = vectorstore._collection
        count = collection.count()
        
        return {
            "total_documents": count,
            "collection_name": collection.name,
            "persist_directory": vectorstore._persist_directory
        }
    except Exception as e:
        return {
            "error": str(e),
            "total_documents": 0
        }


def delete_collection(
    persist_directory: str = "./chroma_db",
    collection_name: str = "university_notes"
):
    """
    Delete a collection from the vector store.
    
    Use with caution - this permanently removes all indexed documents.
    
    Args:
        persist_directory (str): Directory where vector store is persisted
        collection_name (str): Name of collection to delete
    """
    import shutil
    
    if os.path.exists(persist_directory):
        response = input(f"⚠ Delete vector store at {persist_directory}? (yes/no): ")
        if response.lower() == "yes":
            shutil.rmtree(persist_directory)
            print(f"✓ Deleted vector store at {persist_directory}")
        else:
            print("✗ Deletion cancelled")
    else:
        print(f"✗ Vector store not found at {persist_directory}")


# Example usage and testing
if __name__ == "__main__":
    import sys
    from loader import load_multiple_pdfs, load_pdf
    from chunker import chunk_documents
    from embedder import create_embedding_function
    
    print("=" * 80)
    print("Vector Store Test")
    print("=" * 80)
    
    if len(sys.argv) < 2:
        print("\nUsage: python vector_store.py <pdf_file_or_directory> [--rebuild]")
        print("\nOptions:")
        print("  --rebuild: Delete existing vector store and rebuild from scratch")
        sys.exit(1)
    
    path = sys.argv[1]
    rebuild = "--rebuild" in sys.argv
    
    persist_dir = "./test_chroma_db"
    
    # Delete existing if rebuild requested
    if rebuild and os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
        print(f"✓ Deleted existing vector store\n")
    
    # Create embedding function
    print("\n" + "-" * 80)
    embedding_function = create_embedding_function()
    
    # Check if vector store exists
    db_exists = os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))
    
    if db_exists and not rebuild:
        # Load existing vector store
        print("\n" + "-" * 80)
        vectorstore = initialize_vectorstore(
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )
    else:
        # Create new vector store from documents
        print("\n" + "-" * 80)
        
        # Load PDFs
        if os.path.isfile(path):
            documents = load_pdf(path)
        else:
            documents = load_multiple_pdfs(path)
        
        if not documents:
            print("✗ No documents loaded")
            sys.exit(1)
        
        # Chunk documents
        print("\n" + "-" * 80)
        chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
        
        # Create vector store
        print("\n" + "-" * 80)
        vectorstore = create_vectorstore_from_documents(
            documents=chunks,
            embedding_function=embedding_function,
            persist_directory=persist_dir
        )
    
    # Get statistics
    print("\n" + "-" * 80)
    stats = get_collection_stats(vectorstore)
    print("\nVector Store Statistics:")
    print(f"  Total documents: {stats.get('total_documents', 0)}")
    print(f"  Collection name: {stats.get('collection_name', 'unknown')}")
    print(f"  Persist directory: {stats.get('persist_directory', 'unknown')}")
    
    # Test similarity search
    print("\n" + "-" * 80)
    test_query = "What is machine learning?"
    print(f"\nTest query: '{test_query}'")
    print("Searching for similar documents...")
    
    results = similarity_search_with_score(vectorstore, test_query, k=3)
    
    print(f"\n✓ Found {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Source: {doc.metadata.get('source', 'unknown')}")
        print(f"   Page: {doc.metadata.get('page', 'unknown')}")
        print(f"   Content preview: {doc.page_content[:150]}...")
    
    # Test retriever
    print("\n" + "-" * 80)
    print("\nTesting retriever interface...")
    retriever = get_retriever(vectorstore, search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(test_query)
    print(f"✓ Retriever returned {len(retrieved_docs)} documents")
    
    print("\n" + "=" * 80)
    print("Vector store test complete!")
    print("=" * 80)

