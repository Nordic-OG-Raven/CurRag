"""
RAG Pipeline Module

This module implements the Retrieval-Augmented Generation pipeline using
LangChain Expression Language (LCEL) patterns inspired by university coursework.

Key pattern: declarative chain composition with | operator for clean, maintainable code.
"""

import time
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import langchainhub as hub
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single context string.
    
    This is a key helper function in the LCEL chain pattern.
    Joins document contents with double newlines for readability.
    
    Args:
        docs (List[Document]): Retrieved documents
        
    Returns:
        str: Formatted context string
    """
    return "\n\n".join(doc.page_content for doc in docs)


def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Format retrieved documents with source citations.
    
    Includes source file and page number for each chunk.
    
    Args:
        docs (List[Document]): Retrieved documents
        
    Returns:
        str: Formatted context string with sources
    """
    formatted_chunks = []
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        
        chunk_text = f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
        formatted_chunks.append(chunk_text)
    
    return "\n\n---\n\n".join(formatted_chunks)


def get_prompt_template(use_hub: bool = True, hub_prompt: str = "rlm/rag-prompt") -> ChatPromptTemplate:
    """
    Get prompt template from LangChain Hub or create custom one.
    
    LangChain Hub provides battle-tested, community-optimized prompts.
    Falls back to custom prompt if hub is unavailable.
    
    Args:
        use_hub (bool): Whether to use LangChain Hub
        hub_prompt (str): Hub prompt identifier
        
    Returns:
        ChatPromptTemplate: Prompt template for RAG
    """
    if use_hub:
        try:
            print(f"Loading prompt from LangChain Hub: {hub_prompt}")
            prompt = hub.pull(hub_prompt)
            return prompt
        except Exception as e:
            print(f"⚠ Could not load hub prompt: {e}")
            print("Falling back to custom prompt...")
    
    # Custom prompt for university notes
    template = """You are a helpful assistant that answers questions based solely on the provided context from university lecture notes.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information provided in the context above
- If the context does not contain enough information to answer the question, say "I cannot find this information in the provided notes."
- Always cite the source document and page number when providing information
- Be concise but thorough
- Use bullet points for lists and structure your answer clearly

Answer:"""
    
    return ChatPromptTemplate.from_template(template)


def initialize_rag_chain(
    vectorstore: Chroma,
    llm_model: str = "gpt-4",
    temperature: float = 0.1,
    use_hub_prompt: bool = True,
    search_kwargs: Optional[Dict] = None,
    use_hybrid_search: bool = True
) -> RunnableSequence:
    """
    Initialize RAG chain using LangChain Expression Language (LCEL).
    
    This follows the university class pattern:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    Args:
        vectorstore (Chroma): Vector store with indexed documents
        llm_model (str): OpenAI model to use
        temperature (float): LLM temperature (0.0-1.0)
        use_hub_prompt (bool): Whether to use LangChain Hub prompt
        search_kwargs (Dict, optional): Search parameters (e.g., {"k": 5})
        
    Returns:
        RunnableSequence: LCEL chain ready for .invoke()
    """
    print("\n" + "=" * 80)
    print("Initializing RAG Chain")
    print("=" * 80)
    
    # Set default search kwargs
    if search_kwargs is None:
        search_kwargs = {"k": 5}
    
    # Initialize LLM
    print(f"Initializing LLM: {llm_model} (temperature={temperature})")
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    
    # Create retriever from vectorstore
    k = search_kwargs.get('k', 5)
    
    if use_hybrid_search:
        print(f"Creating HYBRID retriever (semantic + keyword, k={k} documents)")
        # Create a custom retriever that uses hybrid search
        from langchain_core.runnables import RunnableLambda
        from vector_store import hybrid_search
        
        # Wrap hybrid search in a RunnableLambda to make it LCEL-compatible
        retriever = RunnableLambda(lambda query: hybrid_search(vectorstore, query, k=k))
    else:
        print(f"Creating retriever (semantic only, k={k} documents)")
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # Get prompt template
    prompt = get_prompt_template(use_hub=use_hub_prompt)
    print(f"✓ Prompt template loaded")
    
    # Build LCEL chain using declarative syntax
    print("\nBuilding LCEL chain...")
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✓ RAG chain initialized successfully")
    print("=" * 80 + "\n")
    
    return rag_chain


def query(rag_chain: RunnableSequence, question: str) -> str:
    """
    Execute RAG chain with a question.
    
    Simple wrapper around chain.invoke() for clarity.
    
    Args:
        rag_chain (RunnableSequence): Initialized RAG chain
        question (str): User question
        
    Returns:
        str: Generated answer
    """
    return rag_chain.invoke(question)


def evaluate_rag(rag_chain: RunnableSequence, question: str) -> Dict:
    """
    Execute RAG chain with performance metrics tracking.
    
    Inspired by university class evaluation pattern.
    Tracks runtime and estimated token usage.
    
    Args:
        rag_chain (RunnableSequence): Initialized RAG chain
        question (str): User question
        
    Returns:
        Dict: Results including output, runtime, and token estimate
    """
    start_time = time.time()
    result = rag_chain.invoke(question)
    end_time = time.time()
    
    runtime = end_time - start_time
    estimated_tokens = len(result) // 4  # Rough estimate: 1 token ≈ 4 chars
    
    return {
        "output": result,
        "runtime": runtime,
        "estimated_tokens": estimated_tokens,
        "question": question
    }


def print_evaluation_results(eval_results: Dict):
    """
    Print evaluation results in a formatted way.
    
    Args:
        eval_results (Dict): Results from evaluate_rag()
    """
    print("\n" + "=" * 80)
    print("RAG Evaluation Results")
    print("=" * 80)
    print(f"\nQuestion: {eval_results['question']}")
    print("\n" + "-" * 80)
    print("Answer:")
    print(eval_results['output'])
    print("-" * 80)
    print(f"\nMetrics:")
    print(f"  Runtime: {eval_results['runtime']:.2f} seconds")
    print(f"  Estimated tokens: {eval_results['estimated_tokens']}")
    print("=" * 80 + "\n")


def batch_evaluate(
    rag_chain: RunnableSequence,
    questions: List[str]
) -> List[Dict]:
    """
    Evaluate multiple questions and return aggregated results.
    
    Args:
        rag_chain (RunnableSequence): Initialized RAG chain
        questions (List[str]): List of questions
        
    Returns:
        List[Dict]: Results for each question
    """
    results = []
    
    print(f"\nEvaluating {len(questions)} questions...")
    print("-" * 80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        eval_result = evaluate_rag(rag_chain, question)
        results.append(eval_result)
        print(f"  ✓ Runtime: {eval_result['runtime']:.2f}s | Tokens: {eval_result['estimated_tokens']}")
    
    # Calculate aggregate statistics
    avg_runtime = sum(r['runtime'] for r in results) / len(results)
    avg_tokens = sum(r['estimated_tokens'] for r in results) / len(results)
    
    print("\n" + "=" * 80)
    print("Aggregate Statistics:")
    print(f"  Questions: {len(questions)}")
    print(f"  Avg runtime: {avg_runtime:.2f} seconds")
    print(f"  Avg tokens: {int(avg_tokens)}")
    print("=" * 80 + "\n")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    from loader import load_multiple_pdfs, load_pdf
    from chunker import chunk_documents
    from embedder import create_embedding_function
    from vector_store import initialize_vectorstore, create_vectorstore_from_documents
    
    print("=" * 80)
    print("RAG Pipeline Test")
    print("=" * 80)
    
    if len(sys.argv) < 2:
        print("\nUsage: python rag_pipeline.py <pdf_file_or_directory> [question]")
        print("\nExample:")
        print("  python rag_pipeline.py ./data/pdfs 'What is machine learning?'")
        sys.exit(1)
    
    path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "What are the main topics covered in these notes?"
    
    persist_dir = "./test_chroma_db"
    
    # Initialize components
    print("\n" + "-" * 80)
    embedding_function = create_embedding_function()
    
    # Check if vector store exists
    db_exists = os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))
    
    if db_exists:
        # Load existing vector store
        print("\n" + "-" * 80)
        vectorstore = initialize_vectorstore(
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )
    else:
        # Create new vector store
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
        chunks = chunk_documents(documents)
        
        # Create vector store
        print("\n" + "-" * 80)
        vectorstore = create_vectorstore_from_documents(
            documents=chunks,
            embedding_function=embedding_function,
            persist_directory=persist_dir
        )
    
    # Initialize RAG chain
    rag_chain = initialize_rag_chain(
        vectorstore=vectorstore,
        llm_model="gpt-4",
        use_hub_prompt=True,
        search_kwargs={"k": 5}
    )
    
    # Run evaluation
    eval_results = evaluate_rag(rag_chain, question)
    print_evaluation_results(eval_results)
    
    # Optional: batch evaluation with sample questions
    sample_questions = [
        "What are the key concepts?",
        "Can you summarize the main points?",
        "What examples are provided?"
    ]
    
    if "--batch" in sys.argv:
        print("\nRunning batch evaluation...")
        batch_results = batch_evaluate(rag_chain, sample_questions)

