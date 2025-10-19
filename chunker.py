"""
Text Chunking and Cleaning Module

This module handles text preprocessing, cleaning, and chunking of documents.
Uses LangChain's RecursiveCharacterTextSplitter for semantic chunking.
"""

import re
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Clean text by removing page numbers, excessive whitespace, and other artifacts.
    
    Preserves:
    - Paragraph breaks (double newlines)
    - List structures
    - Section headings
    
    Removes:
    - Page numbers (patterns like "Page 1", "1/50", standalone numbers)
    - Excessive whitespace
    - Header/footer artifacts (repeated text)
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove page number patterns
    # Matches: "Page 1", "1/50", "Page 1 of 50", etc.
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)
    text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    
    # Remove standalone page numbers at start/end of lines
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove multiple spaces (but preserve intentional indentation)
    text = re.sub(r' {3,}', '  ', text)
    
    # Remove multiple blank lines (keep max 2 newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing/leading whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove leading/trailing whitespace from entire text
    text = text.strip()
    
    return text


def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Clean a list of documents in place.
    
    Args:
        documents (List[Document]): Documents to clean
        
    Returns:
        List[Document]: Cleaned documents
    """
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    
    return documents


def create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a RecursiveCharacterTextSplitter configured for university notes.
    
    The splitter tries to split on semantic boundaries in this order:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences (periods)
    4. Spaces (words)
    5. Characters (last resort)
    
    Args:
        chunk_size (int): Target chunk size in characters (~200 words = 1000 chars)
        chunk_overlap (int): Overlap between chunks to preserve context
        separators (List[str], optional): Custom separators for splitting
        
    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter
    """
    if separators is None:
        # Default separators optimized for academic text
        separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            " ",     # Words
            ""       # Characters
        ]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        add_start_index=True  # Adds character index to metadata
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    clean: bool = True
) -> List[Document]:
    """
    Split documents into smaller chunks for efficient embedding and retrieval.
    
    This is the main function to use for chunking documents.
    
    Args:
        documents (List[Document]): Documents to chunk
        chunk_size (int): Target chunk size in characters
        chunk_overlap (int): Overlap between chunks
        clean (bool): Whether to clean text before chunking
        
    Returns:
        List[Document]: Chunked documents with preserved metadata
    """
    if not documents:
        return []
    
    # Clean documents if requested
    if clean:
        print("Cleaning documents...")
        documents = clean_documents(documents)
    
    # Create text splitter
    text_splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split documents
    print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    chunks = text_splitter.split_documents(documents)
    
    # Enhance metadata with chunk information
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_size'] = len(chunk.page_content)
    
    print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
    
    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Get statistics about chunks.
    
    Args:
        chunks (List[Document]): List of chunked documents
        
    Returns:
        dict: Statistics including count, avg size, min/max sizes
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0
        }
    
    sizes = [len(chunk.page_content) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(sizes) // len(sizes),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "sources": len(set(chunk.metadata.get("source", "unknown") for chunk in chunks))
    }


def preview_chunks(chunks: List[Document], n: int = 3):
    """
    Print preview of first n chunks.
    
    Args:
        chunks (List[Document]): Chunks to preview
        n (int): Number of chunks to show
    """
    print(f"\nPreview of first {min(n, len(chunks))} chunks:")
    print("=" * 80)
    
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i + 1}:")
        print(f"  Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"  Page: {chunk.metadata.get('page', 'unknown')}")
        print(f"  Size: {len(chunk.page_content)} chars")
        print(f"  Content preview: {chunk.page_content[:150]}...")
        print("-" * 80)


# Example usage and testing
if __name__ == "__main__":
    import sys
    from loader import load_multiple_pdfs, load_pdf
    import os
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        # Load documents
        if os.path.isfile(path):
            docs = load_pdf(path)
        else:
            docs = load_multiple_pdfs(path)
        
        if not docs:
            print("No documents loaded.")
            sys.exit(1)
        
        # Chunk documents
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
        
        # Print statistics
        stats = get_chunk_stats(chunks)
        print("\nChunk Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Average size: {stats['avg_chunk_size']} chars")
        print(f"  Min size: {stats['min_chunk_size']} chars")
        print(f"  Max size: {stats['max_chunk_size']} chars")
        print(f"  Unique sources: {stats['sources']}")
        
        # Preview chunks
        preview_chunks(chunks, n=3)
    else:
        print("Usage: python chunker.py <pdf_file_or_directory>")

