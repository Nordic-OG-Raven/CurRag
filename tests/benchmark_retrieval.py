"""
Benchmark Retrieval Performance

This script benchmarks retrieval performance to ensure <2s requirement is met.

Usage:
    python tests/benchmark_retrieval.py
    python tests/benchmark_retrieval.py --queries 100
"""

import argparse
import sys
from pathlib import Path
import os
import time
import random
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedder import create_embedding_function
from vector_store import initialize_vectorstore, get_collection_stats


def generate_test_queries(n: int = 50) -> List[str]:
    """Generate diverse test queries"""
    query_templates = [
        "What is {}?",
        "Explain {} in detail",
        "How does {} work?",
        "What are the key concepts of {}?",
        "Can you summarize {}?",
        "What is the definition of {}?",
        "Describe the process of {}",
        "What are the main features of {}?",
    ]
    
    topics = [
        "machine learning", "neural networks", "gradient descent", "backpropagation",
        "supervised learning", "unsupervised learning", "deep learning", "AI",
        "optimization", "training", "testing", "validation", "overfitting",
        "regularization", "cross-validation", "feature engineering", "data preprocessing"
    ]
    
    queries = []
    for i in range(n):
        template = random.choice(query_templates)
        topic = random.choice(topics)
        queries.append(template.format(topic))
    
    return queries


def benchmark_retrieval(vectorstore, queries: List[str], k: int = 5) -> dict:
    """
    Benchmark retrieval performance.
    
    Args:
        vectorstore: Vector store to benchmark
        queries (List[str]): Test queries
        k (int): Number of documents to retrieve
        
    Returns:
        dict: Benchmark results
    """
    print(f"\nBenchmarking retrieval with {len(queries)} queries (k={k})...")
    print("-" * 80)
    
    retrieval_times = []
    
    for i, query in enumerate(queries, 1):
        if i % 10 == 0 or i == 1:
            print(f"Progress: {i}/{len(queries)}", end="\r")
        
        start_time = time.time()
        results = vectorstore.similarity_search(query, k=k)
        end_time = time.time()
        
        retrieval_times.append(end_time - start_time)
    
    print(f"Progress: {len(queries)}/{len(queries)} ✓")
    
    # Calculate statistics
    avg_time = sum(retrieval_times) / len(retrieval_times)
    min_time = min(retrieval_times)
    max_time = max(retrieval_times)
    median_time = sorted(retrieval_times)[len(retrieval_times) // 2]
    
    # Calculate percentiles
    p95 = sorted(retrieval_times)[int(len(retrieval_times) * 0.95)]
    p99 = sorted(retrieval_times)[int(len(retrieval_times) * 0.99)]
    
    # Check if meets requirement
    meets_requirement = avg_time < 2.0 and p95 < 2.0
    
    return {
        "total_queries": len(queries),
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "median_time": median_time,
        "p95_time": p95,
        "p99_time": p99,
        "meets_requirement": meets_requirement,
        "times": retrieval_times
    }


def print_benchmark_results(results: dict):
    """Print formatted benchmark results"""
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    
    print(f"\nQueries tested: {results['total_queries']}")
    print(f"\nRetrieval Times:")
    print(f"  Average:  {results['avg_time']:.4f}s")
    print(f"  Median:   {results['median_time']:.4f}s")
    print(f"  Min:      {results['min_time']:.4f}s")
    print(f"  Max:      {results['max_time']:.4f}s")
    print(f"  95th %ile: {results['p95_time']:.4f}s")
    print(f"  99th %ile: {results['p99_time']:.4f}s")
    
    print(f"\nRequirement Check (<2s):")
    if results['meets_requirement']:
        print("  ✓ PASSED - Average and P95 under 2 seconds")
    else:
        print("  ✗ FAILED - Performance below requirement")
        if results['avg_time'] >= 2.0:
            print(f"    Average time {results['avg_time']:.2f}s exceeds 2s")
        if results['p95_time'] >= 2.0:
            print(f"    P95 time {results['p95_time']:.2f}s exceeds 2s")
    
    print("=" * 80 + "\n")


def plot_distribution(results: dict, output_file: str = "retrieval_times.png"):
    """Plot retrieval time distribution"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(results['times'], bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(results['avg_time'], color='red', linestyle='--', label=f'Average: {results["avg_time"]:.4f}s')
        plt.axvline(results['median_time'], color='green', linestyle='--', label=f'Median: {results["median_time"]:.4f}s')
        plt.axvline(2.0, color='orange', linestyle='--', label='Requirement: 2.0s')
        
        plt.xlabel('Retrieval Time (seconds)')
        plt.ylabel('Frequency')
        plt.title(f'Retrieval Time Distribution (n={results["total_queries"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {output_file}")
        
    except ImportError:
        print("⚠ matplotlib not installed. Skipping plot generation.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval performance")
    parser.add_argument(
        "--queries",
        type=int,
        default=50,
        help="Number of test queries (default: 50)"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_db",
        help="Vector store directory"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate distribution plot"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Retrieval Performance Benchmark")
    print("=" * 80 + "\n")
    
    # Check if vector store exists
    if not os.path.exists(args.persist_dir):
        print(f"❌ Vector store not found at {args.persist_dir}")
        print("\nPlease run indexing first:")
        print("  python scripts/index_documents.py --pdf-dir ./data/pdfs")
        sys.exit(1)
    
    # Initialize vector store
    print("-" * 80)
    print("Initializing Vector Store")
    print("-" * 80)
    
    embedding_function = create_embedding_function()
    vectorstore = initialize_vectorstore(
        persist_directory=args.persist_dir,
        embedding_function=embedding_function
    )
    
    # Get stats
    stats = get_collection_stats(vectorstore)
    print(f"\nVector Store Info:")
    print(f"  Documents: {stats.get('total_documents', 0)}")
    print(f"  Collection: {stats.get('collection_name', 'unknown')}")
    
    # Generate test queries
    print("\n" + "-" * 80)
    print(f"Generating {args.queries} test queries...")
    queries = generate_test_queries(args.queries)
    print(f"✓ Generated {len(queries)} queries")
    
    # Run benchmark
    print("-" * 80)
    results = benchmark_retrieval(vectorstore, queries, k=args.k)
    
    # Print results
    print_benchmark_results(results)
    
    # Generate plot if requested
    if args.plot:
        plot_distribution(results)
    
    # Exit with appropriate code
    sys.exit(0 if results['meets_requirement'] else 1)


if __name__ == "__main__":
    main()

