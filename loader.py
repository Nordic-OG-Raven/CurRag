"""
PDF Document Loader Module

This module handles loading PDF documents and converting them into LangChain Document objects.
Uses PyMuPDF (fitz) for robust PDF text extraction with metadata preservation.
"""

import os
from typing import List, Optional
from pathlib import Path
import fitz  # PyMuPDF
from langchain_core.documents import Document


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a single PDF file and convert to LangChain Document objects.
    
    Each page becomes a separate Document with metadata including:
    - source: filename
    - page: page number
    - total_pages: total pages in document
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[Document]: List of Document objects, one per page
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF cannot be read
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    documents = []
    filename = os.path.basename(pdf_path)
    
    try:
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            text = page.get_text()
            
            # Create Document with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_num + 1,  # 1-indexed for user-friendliness
                    "total_pages": total_pages,
                    "file_path": pdf_path
                }
            )
            documents.append(doc)
        
        pdf_document.close()
        print(f"✓ Loaded {filename}: {total_pages} pages")
        
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
    
    return documents


def load_multiple_pdfs(pdf_dir: str, pattern: str = "*.pdf") -> List[Document]:
    """
    Load all PDF files from a directory.
    
    Args:
        pdf_dir (str): Path to directory containing PDFs
        pattern (str): Glob pattern for PDF files (default: "*.pdf")
        
    Returns:
        List[Document]: Combined list of Documents from all PDFs
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    pdf_path = Path(pdf_dir)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")
    
    if not pdf_path.is_dir():
        raise ValueError(f"Path is not a directory: {pdf_dir}")
    
    # Find all PDF files
    pdf_files = list(pdf_path.glob(pattern))
    
    if not pdf_files:
        print(f"⚠ No PDF files found in {pdf_dir}")
        return []
    
    print(f"\nLoading {len(pdf_files)} PDF files from {pdf_dir}...")
    print("-" * 60)
    
    all_documents = []
    
    for pdf_file in sorted(pdf_files):
        try:
            docs = load_pdf(str(pdf_file))
            all_documents.extend(docs)
        except Exception as e:
            print(f"✗ Error loading {pdf_file.name}: {str(e)}")
            continue
    
    print("-" * 60)
    print(f"✓ Total: {len(all_documents)} pages loaded from {len(pdf_files)} PDFs\n")
    
    return all_documents


def get_document_stats(documents: List[Document]) -> dict:
    """
    Get statistics about loaded documents.
    
    Args:
        documents (List[Document]): List of Document objects
        
    Returns:
        dict: Statistics including total pages, unique sources, avg content length
    """
    if not documents:
        return {"total_pages": 0, "unique_sources": 0, "avg_content_length": 0}
    
    sources = set(doc.metadata.get("source", "unknown") for doc in documents)
    total_length = sum(len(doc.page_content) for doc in documents)
    
    return {
        "total_pages": len(documents),
        "unique_sources": len(sources),
        "avg_content_length": total_length // len(documents) if documents else 0,
        "sources": sorted(list(sources))
    }


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Load single PDF or directory
        path = sys.argv[1]
        
        if os.path.isfile(path):
            docs = load_pdf(path)
        else:
            docs = load_multiple_pdfs(path)
        
        # Print statistics
        stats = get_document_stats(docs)
        print("\nDocument Statistics:")
        print(f"  Total pages: {stats['total_pages']}")
        print(f"  Unique sources: {stats['unique_sources']}")
        print(f"  Avg content length: {stats['avg_content_length']} chars")
        
        # Show first document preview
        if docs:
            print("\nFirst document preview:")
            print(f"  Source: {docs[0].metadata['source']}")
            print(f"  Page: {docs[0].metadata['page']}")
            print(f"  Content preview: {docs[0].page_content[:200]}...")
    else:
        print("Usage: python loader.py <pdf_file_or_directory>")

