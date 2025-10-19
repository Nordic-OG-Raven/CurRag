"""
Embedding Generation Module

This module provides functions for generating embeddings from text using various models.
Supports both OpenAI embeddings and sentence-transformers models.
"""

from typing import List, Union
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


def create_embedding_function(
    model_name: str = "text-embedding-ada-002",
    use_openai: bool = True
) -> Embeddings:
    """
    Create an embedding function for generating vector embeddings.
    
    Supports two types of embeddings:
    1. OpenAI embeddings (text-embedding-ada-002) - requires API key
    2. Sentence Transformers (local, free) - runs locally
    
    Args:
        model_name (str): Model name for embeddings
            - For OpenAI: use "text-embedding-ada-002" or "text-embedding-3-small"
            - For local: use "sentence-transformers/all-MiniLM-L6-v2" (default, 384-dim)
        use_openai (bool): Whether to use OpenAI embeddings (requires API key)
        
    Returns:
        Embeddings: LangChain embeddings function
    """
    if use_openai:
        print(f"Using OpenAI embeddings: {model_name}")
        return OpenAIEmbeddings(model=model_name)
    else:
        print(f"Using local sentence-transformers: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
        )


def test_embedding_function(embedding_function: Embeddings):
    """
    Test an embedding function with sample text.
    
    Args:
        embedding_function (Embeddings): Embedding function to test
    """
    test_text = "This is a test sentence for embedding generation."
    
    print("\nTesting embedding function...")
    print(f"Test text: '{test_text}'")
    
    # Generate embedding
    embedding = embedding_function.embed_query(test_text)
    
    print(f"✓ Embedding generated successfully")
    print(f"  Dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Magnitude: {sum(x**2 for x in embedding)**0.5:.4f}")
    
    return embedding


def get_embedding_info(model_name: str, use_openai: bool = False) -> dict:
    """
    Get information about an embedding model.
    
    Args:
        model_name (str): Name of the embedding model
        use_openai (bool): Whether it's an OpenAI model
        
    Returns:
        dict: Model information including dimensions and characteristics
    """
    # Common embedding model dimensions
    model_info = {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dimensions": 384,
            "max_seq_length": 256,
            "description": "Fast, lightweight model for semantic similarity",
            "performance": "Good for most use cases"
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "dimensions": 768,
            "max_seq_length": 384,
            "description": "Higher quality embeddings, slower",
            "performance": "Best quality for sentence-transformers"
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_seq_length": 8191,
            "description": "OpenAI's embedding model (requires API)",
            "performance": "High quality, requires API costs"
        },
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_seq_length": 8191,
            "description": "OpenAI's efficient embedding model",
            "performance": "Good balance of cost and quality"
        }
    }
    
    return model_info.get(model_name, {
        "dimensions": "Unknown",
        "max_seq_length": "Unknown",
        "description": "Custom model",
        "performance": "Unknown"
    })


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    
    print("=" * 80)
    print("Embedding Function Test")
    print("=" * 80)
    
    # Determine which embedding to use
    use_openai = "--openai" in sys.argv
    
    if use_openai:
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set in environment")
            print("Please set it with: export OPENAI_API_KEY='your-key'")
            sys.exit(1)
        model_name = "text-embedding-3-small"
    else:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Get model info
    info = get_embedding_info(model_name, use_openai)
    print(f"\nModel: {model_name}")
    print(f"Dimensions: {info['dimensions']}")
    print(f"Max sequence length: {info['max_seq_length']}")
    print(f"Description: {info['description']}")
    print(f"Performance: {info['performance']}")
    
    # Create embedding function
    print("\n" + "-" * 80)
    embedding_function = create_embedding_function(model_name, use_openai)
    
    # Test it
    print("-" * 80)
    test_embedding_function(embedding_function)
    
    # Test batch embedding
    print("\n" + "-" * 80)
    print("Testing batch embedding...")
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "This sentence is completely different and talks about cooking."
    ]
    
    embeddings = embedding_function.embed_documents(test_texts)
    print(f"✓ Generated embeddings for {len(test_texts)} documents")
    print(f"  Shape: {len(embeddings)} x {len(embeddings[0])}")
    
    # Calculate similarity between first two (should be high) and first and third (should be lower)
    def cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a**2 for a in vec1)**0.5
        mag2 = sum(b**2 for b in vec2)**0.5
        return dot_product / (mag1 * mag2)
    
    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"\nSimilarity Analysis:")
    print(f"  Text 1 ↔ Text 2 (related): {sim_1_2:.4f}")
    print(f"  Text 1 ↔ Text 3 (unrelated): {sim_1_3:.4f}")
    print(f"  ✓ Related texts are more similar!" if sim_1_2 > sim_1_3 else "  ✗ Unexpected similarity pattern")
    
    print("\n" + "=" * 80)
    print("Embedding test complete!")
    print("=" * 80)

