"""
RoboPaper - arXiv RAG Demo
A professional Streamlit interface for research paper Q&A.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="RoboPaper - Research Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Academic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Playfair+Display:wght@700&display=swap');

    /* Main background - Clean Minimalist */
    .stApp {
        background-color: #FFFFFF;
        color: #1E293B;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers - Serif for Academic feel */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Minimalist Buttons */
    .stButton>button {
        background-color: #1E293B;
        color: white;
        border-radius: 2px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-size: 0.85rem;
    }
    
    .stButton>button:hover {
        background-color: #334155;
    }
    
    /* Results Container */
    .answer-container {
        border-left: 3px solid #1E293B;
        padding: 24px;
        background-color: #F8FAFC;
        line-height: 1.7;
        font-size: 1.05rem;
        font-weight: 300;
    }
    
    /* Source Cards */
    .stExpander {
        border: none !important;
        border-bottom: 1px solid #E2E8F0 !important;
        border-radius: 0px !important;
    }
    
    /* Home Page Cards */
    .info-card {
        padding: 20px;
        height: 100%;
        border-left: 1px solid #E2E8F0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 0px;
        color: #64748B;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #0F172A;
        border-bottom: 2px solid #0F172A;
    }
</style>
""", unsafe_allow_html=True)

# Lazy load RAG components
@st.cache_resource
def get_rag_engine():
    """Build the retrieval core."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from huggingface_hub import InferenceClient
    
    # 1. Local Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Vector Store
    if not os.path.exists("./chroma_db"):
        return None, None, None
        
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 3. LLM Client
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        st.error("HUGGINGFACEHUB_API_TOKEN missing in .env")
        st.stop()
        
    client = InferenceClient(token=hf_token)
    
    return vectorstore, client, embeddings

# Sidebar info
with st.sidebar:
    st.markdown("### Research Assistant")
    st.markdown("---")
    st.markdown("### Methodology")
    st.caption("Retrieval-Augmented Generation (RAG) using Vector Embeddings.")
    
    st.markdown("### Knowledge Base")
    st.caption("arXiv Research Abstracts (Robotics & ML)")
    
    st.markdown("### Scope")
    st.caption("Strictly limited to the application of Machine Learning in Robotics.")
    
    st.markdown("---")
    st.markdown("**Author**: Anas")
    
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

# Check for database
vectorstore, hf_client, _ = get_rag_engine()

# Tabs for Home vs Application
# Using invisible characters for cleaner tab look if desired, or simple text
tab_home, tab_app = st.tabs(["Home", "Consult Assistant"])

# --- HOME PAGE ---
with tab_home:
    st.title("RoboPaper")
    st.markdown("### *Specialized Intelligence for Robotics & Machine Learning*")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            "This system provides a targeted interface for querying scientific literature. "
            "It ingests a dynamic corpus of **75 latest research papers** directly from **arXiv**, focusing exclusively on the intersection of **Robotics** and **Machine Learning**."
        )
        st.markdown(
            "**System Metrics**: The current knowledge base consists of a **curated snapshot of 75 research papers** on Robot Reinforcement Learning. "
            "Upon querying, the system searches *within this fixed snapshot* to retrieve the **top 10 most relevant text chunks**."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        # Call to Action
        if st.button("Begin Research Query →"):
            # This is a hack to switch tabs - simpler is to just guide them
            st.info("Please click the 'Consult Assistant' tab above to begin.")

    with col2:
        st.caption("SYSTEM ARCHITECTURE")
        st.code("""
SOURCE: arXiv API
   ↓
VECTOR STORE: ChromaDB
   ↓
RETRIEVAL: Semantic Search
   ↓
GENERATION: LLM (Qwen-72B)
        """, language="text")

# --- APP PAGE ---
with tab_app:
    if vectorstore is None:
        st.warning("Knowledge base not initialized.")
        st.info("Please index documents to provide context for the assistant.")
        st.stop()

    st.markdown("### Consult Knowledge Base")
    st.caption("Scope: Applications of Machine Learning in Robotics")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("query", placeholder="e.g. How is reinforcement learning applied in soft robotics?", label_visibility="collapsed")
    with col2:
        search_clicked = st.button("Analyze", use_container_width=True)

    if search_clicked or question:
        if question:
            st.markdown("---")
            
            # Step 1: Retrieval
            with st.status("Retrieving context...", expanded=False) as status:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
                docs = retriever.invoke(question)
                status.update(label=f"Identified {len(docs)} references", state="complete")
            
            # Step 2: Generation
            with st.spinner("Synthesizing response..."):
                try:
                    context = "\n\n".join(doc.page_content for doc in docs)
                    prompt = f"""Use the following context to answer the question. 
This system is specialized for questions about Machine Learning in Robotics.
If the question is unrelated to this scope, politely inform the user about the scope limitation.
If you don't know the answer based on the context, say "I don't know" - don't make things up.

Context:
{context}

Question: {question}

Answer:"""
                    
                    response = hf_client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model="Qwen/Qwen2.5-72B-Instruct",
                        max_tokens=600,
                    )
                    answer = response.choices[0].message.content
                    
                    # Display Answer
                    st.markdown("#### Synthesis")
                    st.markdown(f'<div class="answer-container">{answer}</div>', unsafe_allow_html=True)
                    
                    # Display Sources
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.caption("REFERENCE LITERATURE")
                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"[{i}] {doc.metadata.get('Title', 'Untitled')}"):
                            st.markdown(f"**Authors**: {doc.metadata.get('Authors', 'N/A')}")
                            st.markdown(f"**Source**: {doc.metadata.get('URL', '#')}")
                            st.caption(doc.page_content)
                            
                except Exception as e:
                    st.error(f"System Error: {e}")
        else:
            st.warning("Please enter a valid research query.")

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption("RoboPaper v1.0 • Built by Anas")
