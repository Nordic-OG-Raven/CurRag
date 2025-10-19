"""
Quick test to see what documents are being retrieved
"""

from dotenv import load_dotenv
load_dotenv()

from vector_store import initialize_vectorstore, hybrid_search, similarity_search_with_score
from embedder import create_embedding_function

print("=" * 80)
print("Retrieval Test: Industrial Organization Query")
print("=" * 80)

embedding_function = create_embedding_function(use_openai=True)
vectorstore = initialize_vectorstore('./chroma_db', embedding_function=embedding_function)

query = "industrial organization main topics"

print(f"\nQuery: '{query}'\n")

# Test regular semantic search
print("-" * 80)
print("SEMANTIC SEARCH ONLY (current):")
print("-" * 80)
semantic_results = similarity_search_with_score(vectorstore, query, k=10)

for i, (doc, score) in enumerate(semantic_results, 1):
    source = doc.metadata.get('source', 'unknown')
    page = doc.metadata.get('page', '?')
    is_io = '✅ IO!' if 'industrial organization' in source.lower() else ''
    print(f"{i}. Score: {score:.3f} | {source} (Page {page}) {is_io}")

# Test hybrid search
print("\n" + "-" * 80)
print("HYBRID SEARCH (semantic + keyword boosting):")
print("-" * 80)
hybrid_results = hybrid_search(vectorstore, query, k=10)

for i, doc in enumerate(hybrid_results, 1):
    source = doc.metadata.get('source', 'unknown')
    page = doc.metadata.get('page', '?')
    is_io = '✅ IO!' if 'industrial organization' in source.lower() else ''
    print(f"{i}. {source} (Page {page}) {is_io}")

print("\n" + "=" * 80)
print("Analysis:")
io_in_semantic = sum(1 for doc, _ in semantic_results if 'industrial organization' in doc.metadata.get('source', '').lower())
io_in_hybrid = sum(1 for doc in hybrid_results if 'industrial organization' in doc.metadata.get('source', '').lower())

print(f"  Semantic: {io_in_semantic}/10 results from Industrial Organization.pdf")
print(f"  Hybrid:   {io_in_hybrid}/10 results from Industrial Organization.pdf")
print("=" * 80)

