"""
Streamlit Web Application for University Notes RAG System

This app provides a user-friendly interface for querying university notes using RAG.
Implements chain caching for efficiency (initialize once, query many times).
"""

import streamlit as st
import os
import sys
import time
from pathlib import Path
import yaml

# Import RAG components
from embedder import create_embedding_function
from vector_store import initialize_vectorstore, get_collection_stats
from rag_pipeline import initialize_rag_chain, evaluate_rag


# Page configuration
st.set_page_config(
    page_title="University Notes RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None


@st.cache_resource
def initialize_rag_system():
    """
    Initialize the RAG system (cached for performance).
    
    This function is called once and cached by Streamlit.
    Subsequent queries reuse the same chain, avoiding re-initialization.
    
    Returns:
        tuple: (rag_chain, vectorstore, config)
    """
    with st.spinner("Initializing RAG system..."):
        # Load config
        config = load_config()
        
        if config is None:
            st.error("config.yaml not found. Using default settings.")
            config = {
                "data": {"persist_directory": "./chroma_db"},
                "llm": {"model": "gpt-4", "temperature": 0.1},
                "retrieval": {"top_k": 5},
                "prompt": {"use_hub": True}
            }
        
        # Check environment variables
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OPENAI_API_KEY not set! Please set it in your environment.")
            st.stop()
        
        # Initialize embedding function
        embedding_function = create_embedding_function(
            model_name=config.get("embedding", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # Initialize vector store
        persist_dir = config.get("data", {}).get("persist_directory", "./chroma_db")
        
        if not os.path.exists(persist_dir):
            st.error(f"❌ Vector store not found at {persist_dir}")
            st.info("Please run indexing first: `python scripts/index_documents.py`")
            st.stop()
        
        vectorstore = initialize_vectorstore(
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )
        
        # Initialize RAG chain
        rag_chain = initialize_rag_chain(
            vectorstore=vectorstore,
            llm_model=config.get("llm", {}).get("model", "gpt-4"),
            temperature=config.get("llm", {}).get("temperature", 0.1),
            use_hub_prompt=config.get("prompt", {}).get("use_hub", True),
            search_kwargs={"k": config.get("retrieval", {}).get("top_k", 5)},
            use_hybrid_search=config.get("retrieval", {}).get("use_hybrid", True)
        )
        
        return rag_chain, vectorstore, config


def render_header():
    """Render the app header"""
    st.title("📚 University Notes RAG System")
    st.markdown("*Ask questions about your university lecture notes*")
    
    # Info banner for free tier hosting
    st.info("ℹ️ **Demo Notice:** This is hosted on a free tier. First load may take up to 60 seconds if the app has been inactive. Subsequent queries are fast!")
    
    st.markdown("---")


def render_sidebar(vectorstore, config):
    """Render the sidebar with system information"""
    st.sidebar.title("⚙️ System Info")
    
    # Vector store statistics
    stats = get_collection_stats(vectorstore)
    st.sidebar.metric("Total Documents", stats.get("total_documents", 0))
    st.sidebar.metric("Collection", stats.get("collection_name", "unknown"))
    
    st.sidebar.markdown("---")
    
    # Configuration
    st.sidebar.subheader("Configuration")
    st.sidebar.text(f"Model: {config.get('llm', {}).get('model', 'gpt-4')}")
    st.sidebar.text(f"Temperature: {config.get('llm', {}).get('temperature', 0.1)}")
    st.sidebar.text(f"Top-K: {config.get('retrieval', {}).get('top_k', 5)}")
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.subheader("Display Settings")
    show_metadata = st.sidebar.checkbox("Show metadata", value=False)
    show_metrics = st.sidebar.checkbox("Show metrics", value=True)
    
    return show_metadata, show_metrics


def render_query_interface(rag_chain, show_metadata, show_metrics):
    """Render the main query interface"""
    
    # Query input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g., What is gradient descent?",
            key="question_input"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        submit = st.button("🔍 Search", type="primary")
    
    # Example questions
    with st.expander("💡 Example questions"):
        example_questions = [
            "What are the main concepts covered?",
            "Explain the key principles.",
            "What examples are provided?",
            "Summarize the main points.",
            "What are the important formulas?"
        ]
        for eq in example_questions:
            if st.button(eq, key=f"example_{eq}"):
                st.session_state.question_input = eq
                st.rerun()
    
    st.markdown("---")
    
    # Process query
    if submit and question:
        process_query(rag_chain, question, show_metadata, show_metrics)
    
    # Display query history
    if "query_history" in st.session_state and st.session_state.query_history:
        render_history()


def process_query(rag_chain, question, show_metadata, show_metrics):
    """Process a query and display results"""
    
    with st.spinner("🔍 Searching and generating answer..."):
        # Evaluate query with metrics
        eval_results = evaluate_rag(rag_chain, question)
    
    # Display results
    st.subheader("📝 Answer")
    st.markdown(eval_results["output"])
    
    # Display metrics
    if show_metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("⏱️ Runtime", f"{eval_results['runtime']:.2f}s")
        with col2:
            st.metric("🔢 Est. Tokens", eval_results['estimated_tokens'])
    
    # Display metadata if requested
    if show_metadata:
        with st.expander("🔍 Debug Info"):
            st.json(eval_results)
    
    # Save to history
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    st.session_state.query_history.append({
        "question": question,
        "answer": eval_results["output"],
        "runtime": eval_results["runtime"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Limit history to last 10 queries
    if len(st.session_state.query_history) > 10:
        st.session_state.query_history = st.session_state.query_history[-10:]


def render_history():
    """Render query history"""
    with st.expander("📜 Query History"):
        for i, entry in enumerate(reversed(st.session_state.query_history), 1):
            st.markdown(f"**{i}. {entry['question']}** *({entry['timestamp']})*")
            st.markdown(f"> {entry['answer'][:200]}...")
            st.markdown(f"*Runtime: {entry['runtime']:.2f}s*")
            st.markdown("---")


def render_admin_interface():
    """Render admin interface for document management"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Admin")
    
    if st.sidebar.button("🔄 Reload System"):
        st.cache_resource.clear()
        st.success("System reloaded! Refresh the page.")
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear History"):
        if "query_history" in st.session_state:
            st.session_state.query_history = []
            st.success("History cleared!")


def main():
    """Main application entry point"""
    
    # Initialize RAG system (cached)
    try:
        rag_chain, vectorstore, config = initialize_rag_system()
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        st.info("Please check your configuration and environment variables.")
        st.stop()
    
    # Render UI
    render_header()
    show_metadata, show_metrics = render_sidebar(vectorstore, config)
    render_query_interface(rag_chain, show_metadata, show_metrics)
    render_admin_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "University Notes RAG System | Powered by LangChain & OpenAI"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

