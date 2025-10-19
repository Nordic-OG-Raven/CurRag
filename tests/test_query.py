"""
Test Query Script

This script tests the RAG pipeline with sample queries and validates responses.

Usage:
    python tests/test_query.py
    python tests/test_query.py --questions "What is ML?" "Explain AI"
"""

import argparse
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedder import create_embedding_function
from vector_store import initialize_vectorstore, get_collection_stats
from rag_pipeline import initialize_rag_chain, evaluate_rag, batch_evaluate, print_evaluation_results


def get_test_questions():
    """Get default test questions"""
    return [
        "What are the main topics covered in these notes?",
        "Can you explain the key concepts?",
        "What examples are provided?",
        "Summarize the important points.",
        "What are the main formulas or equations?"
    ]


def validate_response(eval_result: dict) -> dict:
    """
    Validate a response for quality metrics.
    
    Args:
        eval_result (dict): Evaluation result from evaluate_rag()
        
    Returns:
        dict: Validation results
    """
    output = eval_result["output"]
    
    validation = {
        "has_content": len(output) > 50,
        "reasonable_length": 50 < len(output) < 2000,
        "fast_response": eval_result["runtime"] < 15.0,
        "not_empty": output.strip() != "",
        "no_error_messages": "error" not in output.lower() and "cannot" not in output.lower()
    }
    
    validation["all_passed"] = all(validation.values())
    
    return validation


def main():
    parser = argparse.ArgumentParser(description="Test RAG system with queries")
    parser.add_argument(
        "--questions",
        nargs="+",
        help="Custom questions to test"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_db",
        help="Vector store directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full responses"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RAG System Query Test")
    print("=" * 80 + "\n")
    
    # Check if vector store exists
    if not os.path.exists(args.persist_dir):
        print(f"❌ Vector store not found at {args.persist_dir}")
        print("\nPlease run indexing first:")
        print("  python scripts/index_documents.py --pdf-dir ./data/pdfs")
        sys.exit(1)
    
    # Initialize components
    print("-" * 80)
    print("Initializing RAG System")
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
    
    # Initialize RAG chain
    rag_chain = initialize_rag_chain(
        vectorstore=vectorstore,
        llm_model="gpt-4",
        search_kwargs={"k": 5}
    )
    
    # Get questions
    questions = args.questions if args.questions else get_test_questions()
    
    print(f"\nTesting with {len(questions)} questions...")
    print("=" * 80 + "\n")
    
    # Run tests
    results = []
    validations = []
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}/{len(questions)}: {question}")
        print("-" * 80)
        
        # Evaluate
        eval_result = evaluate_rag(rag_chain, question)
        
        # Validate
        validation = validate_response(eval_result)
        
        # Print results
        if args.verbose:
            print(f"\nAnswer:\n{eval_result['output']}\n")
        else:
            preview = eval_result['output'][:200] + "..." if len(eval_result['output']) > 200 else eval_result['output']
            print(f"Answer: {preview}\n")
        
        print(f"Metrics:")
        print(f"  Runtime: {eval_result['runtime']:.2f}s")
        print(f"  Tokens: {eval_result['estimated_tokens']}")
        print(f"  Validation: {'✓ PASSED' if validation['all_passed'] else '✗ FAILED'}")
        
        if not validation['all_passed']:
            failed_checks = [k for k, v in validation.items() if not v and k != 'all_passed']
            print(f"  Failed checks: {', '.join(failed_checks)}")
        
        print("\n" + "=" * 80 + "\n")
        
        results.append(eval_result)
        validations.append(validation)
    
    # Aggregate results
    print("=" * 80)
    print("Aggregate Test Results")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in validations if v['all_passed'])
    avg_runtime = sum(r['runtime'] for r in results) / len(results)
    avg_tokens = sum(r['estimated_tokens'] for r in results) / len(results)
    
    print(f"\nTests: {passed_tests}/{total_tests} passed ({100*passed_tests/total_tests:.1f}%)")
    print(f"Average runtime: {avg_runtime:.2f}s")
    print(f"Average tokens: {int(avg_tokens)}")
    
    # Performance requirements check
    print(f"\nPerformance Requirements:")
    print(f"  ✓ All queries < 15s: {all(r['runtime'] < 15 for r in results)}")
    print(f"  ✓ No empty responses: {all(len(r['output']) > 0 for r in results)}")
    
    # Validation breakdown
    print(f"\nValidation Breakdown:")
    for check in ['has_content', 'reasonable_length', 'fast_response', 'no_error_messages']:
        passed = sum(1 for v in validations if v.get(check, False))
        print(f"  {check}: {passed}/{total_tests} ({100*passed/total_tests:.1f}%)")
    
    print("\n" + "=" * 80)
    
    if passed_tests == total_tests:
        print("✓ All tests passed!")
    else:
        print(f"⚠ {total_tests - passed_tests} test(s) failed")
    
    print("=" * 80 + "\n")
    
    sys.exit(0 if passed_tests == total_tests else 1)


if __name__ == "__main__":
    main()

