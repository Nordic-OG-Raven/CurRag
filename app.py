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
    initial_sidebar_state="collapsed"
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


def inject_custom_css():
    """Inject custom CSS to match website design system"""
    # Inject favicon via JavaScript (deferred to avoid blocking)
    try:
        st.markdown("""
        <script>
        (function() {
            try {
                // Remove existing favicon
                const existingFavicon = document.querySelector("link[rel='icon']");
                if (existingFavicon) existingFavicon.remove();
                
                // Add new favicon
                const link = document.createElement('link');
                link.rel = 'icon';
                link.type = 'image/jpeg';
                link.href = '/favicon.jpg';
                document.head.appendChild(link);
                
                // Also set apple-touch-icon
                const appleLink = document.createElement('link');
                appleLink.rel = 'apple-touch-icon';
                appleLink.href = '/favicon.jpg';
                document.head.appendChild(appleLink);
            } catch(e) {
                console.error('Favicon injection error:', e);
            }
        })();
        </script>
        """, unsafe_allow_html=True)
    except Exception as e:
        pass  # Don't crash if JS injection fails
    
    st.markdown("""
    <style>
    /* Hide sidebar completely */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stSidebar {display: none !important;}
    
    /* Animated background - CSS particles */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: #0f172a;
        z-index: -2;
    }
    
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(124, 58, 237, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(124, 58, 237, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(124, 58, 237, 0.06) 0%, transparent 50%);
        background-size: 100% 100%;
        animation: pulse 8s ease-in-out infinite;
        z-index: -1;
        pointer-events: none;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
    }
    
    /* Animated particles */
    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(30px, -30px) rotate(120deg); }
        66% { transform: translate(-20px, 20px) rotate(240deg); }
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
        position: relative;
        z-index: 1;
    }
    
    /* Background color */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
        position: relative;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    p, div, span, label {
        color: #f1f5f9 !important;
    }
    
    /* Remove default Streamlit styling */
    .stMarkdown {
        color: #f1f5f9;
    }
    
    /* Title styling */
    h1 {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #f1f5f9 !important;
    }
    
    /* Subtitle */
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Description text */
    .description {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background-color: #1e293b;
        border-radius: 12px;
        border: 1px solid #1e293b;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #7c3aed;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #6d28d9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        color: #f1f5f9;
        border-radius: 8px;
    }
    
    .streamlit-expanderContent {
        background-color: #1e293b;
        color: #f1f5f9;
    }
    
    /* Answer section */
    .answer-section {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #1e293b;
        margin-top: 1.5rem;
    }
    
    /* Metrics styling */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    [data-testid="stMetricValue"] {
        color: #7c3aed;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }
    
    /* Divider */
    hr {
        border-color: #1e293b;
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.875rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #1e293b;
    }
    
    /* Configuration info (subtle) */
    .config-info {
        color: #64748b;
        font-size: 0.75rem;
        text-align: center;
        margin-top: 1rem;
    }
    
    /* Course catalog styling */
    .dataframe {
        background-color: #1e293b;
        border: 1px solid #1e293b;
        border-radius: 12px;
    }
    
    /* Logo container */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 2rem;
    }
    
    .logo-container img {
        max-width: 300px;
        width: auto;
        height: auto;
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
        image-rendering: high-quality;
    }
    
    /* Link styling */
    a {
        color: #7c3aed;
        text-decoration: none;
    }
    
    a:hover {
        color: #6d28d9;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #7c3aed;
    }
    
    /* Error messages */
    .stAlert {
        background-color: #1e293b;
        border: 1px solid #1e293b;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header(config):
    """Render the app header"""
    # Logo - use use_column_width=False to preserve quality
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    try:
        st.image("au_logo.png", use_column_width=False, width=300)
    except:
        pass
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Title
    st.title("University Notes RAG System")
    
    # Subtitle
    st.markdown('<p class="subtitle">Ask questions about your university lecture notes</p>', unsafe_allow_html=True)
    
    # Description (styled, not blue info box)
    st.markdown("""
    <div class="description">
        <strong>Retrieval-Augmented Generation (RAG)</strong> — an AI system that searches through my personal knowledge base 
        to answer your questions. Trained on <strong>2,652 pages</strong> from 26 university courses covering Business, Economics, 
        Data Science, and AI. Every answer is grounded in lecture notes written by me, not generic internet knowledge.
    </div>
    """, unsafe_allow_html=True)


def render_query_interface(rag_chain, config):
    """Render the main query interface"""
    
    # Query input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g., What is gradient descent?",
            key="question_input",
            label_visibility="visible"
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
        process_query(rag_chain, question, config)
    
    # Display query history
    if "query_history" in st.session_state and st.session_state.query_history:
        render_history()


def process_query(rag_chain, question, config):
    """Process a query and display results"""
    
    with st.spinner("🔍 Searching and generating answer..."):
        # Evaluate query with metrics
        eval_results = evaluate_rag(rag_chain, question)
    
    # Display results
    st.markdown('<div class="answer-section">', unsafe_allow_html=True)
    st.subheader("📝 Answer")
    st.markdown(eval_results["output"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⏱️ Runtime", f"{eval_results['runtime']:.2f}s")
    with col2:
        st.metric("🔢 Est. Tokens", eval_results['estimated_tokens'])
    
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


def render_course_catalog():
    """Render catalog of all courses in the knowledge base"""
    st.markdown("---")
    st.markdown("### 📚 Course Catalog")
    
    # Intro text
    st.markdown(
        "All answers are based on lecture notes from my **Bachelor's in Economics and Business Administration** "
        "and **Master's in Business Intelligence** at Aarhus University. "
        "Interested in learning more? View the full curricula: "
        "[Bachelor Programme](https://bachelor.au.dk/en/economics-and-business-administration) | "
        "[Master Programme](https://masters.au.dk/businessintelligence)"
    )
    
    st.markdown("")
    
    # Bachelor courses table
    st.markdown("#### 🎓 Bachelor's Degree - Economics and Business Administration")
    bachelor_courses = {
        "Course": [
            "Organizational Behaviour",
            "Strategy",
            "Marketing Management",
            "Operations Management",
            "Finance",
            "Financial Accounting",
            "Management Accounting",
            "Microeconomics",
            "Macroeconomics",
            "Descriptive Economics",
            "Industrial Organization",
            "Industrial Organization (Laws)",
            "International Finance",
            "Math Formulas"
        ],
        "Description": [
            "Study of human behavior in organizational settings, leadership, and team dynamics",
            "Corporate strategy formulation, competitive analysis, and strategic decision-making",
            "Market analysis, consumer behavior, branding, and marketing strategy development",
            "Production planning, supply chain management, and process optimization",
            "Corporate finance, investment analysis, capital structure, and financial markets",
            "External financial reporting, IFRS standards, and financial statement analysis",
            "Internal accounting for planning, control, and managerial decision-making",
            "Individual economic behavior, market mechanisms, and resource allocation",
            "National economic performance, monetary policy, and business cycles",
            "Empirical economic analysis and statistical methods in economics",
            "Market structures, competition policy, and strategic firm behavior",
            "Competition law, antitrust regulations, and legal frameworks for business",
            "Foreign exchange markets, international monetary systems, and global finance",
            "Mathematical foundations for business and economic analysis"
        ]
    }
    st.dataframe(bachelor_courses, use_container_width=True, hide_index=True)
    
    st.markdown("")
    
    # Master courses table
    st.markdown("#### 🎓 Master's Degree - Business Intelligence")
    master_courses = {
        "Course": [
            "Business Intelligence (BI)",
            "BI Front End",
            "Business Development with IS",
            "Business Forecasting",
            "Customer Analytics",
            "Data Science",
            "Data Mining (DM)",
            "ESG Analytics",
            "Generative AI with LLMs",
            "High Frequency Trading",
            "Machine Learning (ML)",
            "Qualitative Research Methods",
            "Quantitative Research Methods"
        ],
        "Description": [
            "Data warehousing, OLAP, dimensional modeling, and enterprise analytics systems",
            "Dashboard design, data visualization, and user interface for BI systems",
            "IT-enabled innovation, digital transformation, and information systems strategy",
            "Time series analysis, forecasting models, and predictive analytics for business planning",
            "Customer segmentation, CLV analysis, and data-driven marketing optimization",
            "End-to-end data science process from problem definition to model deployment",
            "Pattern discovery, clustering, classification, and association rule mining",
            "Environmental, Social, and Governance metrics analysis for sustainable business",
            "Large language models, prompt engineering, and AI applications in business",
            "Algorithmic trading strategies, market microstructure, and quantitative finance",
            "Supervised and unsupervised learning, model evaluation, and ML deployment",
            "Interview methods, case studies, ethnography, and qualitative data analysis",
            "Statistical inference, hypothesis testing, regression, and experimental design"
        ]
    }
    st.dataframe(master_courses, use_container_width=True, hide_index=True)


def render_footer(config):
    """Render footer with configuration info"""
    st.markdown("---")
    
    # Configuration info (subtle)
    model = config.get('llm', {}).get('model', 'gpt-4')
    temperature = config.get('llm', {}).get('temperature', 0.1)
    top_k = config.get('retrieval', {}).get('top_k', 5)
    
    st.markdown(
        f'<div class="config-info">Model: {model} • Temperature: {temperature} • Top-K: {top_k}</div>',
        unsafe_allow_html=True
    )
    
    # Footer
    st.markdown(
        '<div class="footer">University Notes RAG System | Powered by LangChain & OpenAI</div>',
        unsafe_allow_html=True
    )


def main():
    """Main application entry point"""
    
    # Show something IMMEDIATELY - before any other code
    st.title("University Notes RAG System")
    st.write("Loading...")
    
    # Inject favicon and custom CSS AFTER showing content
    try:
        inject_custom_css()
    except Exception as e:
        st.warning(f"CSS injection failed: {e}")
    
    # Initialize RAG system (cached)
    rag_chain = None
    config = None
    try:
        rag_chain, vectorstore, config = initialize_rag_system()
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        st.info("Please check your configuration and environment variables.")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        # Don't stop - show what we can
        config = {
            "llm": {"model": "gpt-4", "temperature": 0.1},
            "retrieval": {"top_k": 5}
        }
    
    # Render UI (even if RAG failed)
    if rag_chain is not None and config is not None:
        # Clear the loading message and render full UI
        st.empty()
        render_header(config)
        render_query_interface(rag_chain, config)
    else:
        st.warning("⚠️ RAG system unavailable. Please check the error above.")
        if config:
            render_header(config)
    render_course_catalog()
    if config:
        render_footer(config)


if __name__ == "__main__":
    main()
