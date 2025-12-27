"""
RoboPaper - arXiv RAG Demo
A Streamlit app for asking questions about arXiv papers using RAG.
Deploy this to HuggingFace Spaces for free hosting.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="RoboPaper - arXiv RAG",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RoboPaper")
st.markdown("Ask questions about arXiv papers using **Retrieval-Augmented Generation (RAG)**")

# Check for HuggingFace token
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    st.error("⚠️ `HUGGINGFACEHUB_API_TOKEN` not found. Please set it in your environment or Spaces secrets.")
    st.stop()

# Lazy load heavy imports
@st.cache_resource
def load_rag_components():
    """Load embeddings and vector store (cached)."""
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
    from langchain_chroma import Chroma
    from langchain.chains import RetrievalQA
    
    # Embeddings (local, free)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector store
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # LLM (HuggingFace Inference API, free)
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.1,
        max_new_tokens=512,
    )
    
    # RAG Chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    
    return qa_chain

# Load components with spinner
with st.spinner("Loading RAG components..."):
    try:
        qa_chain = load_rag_components()
    except Exception as e:
        st.error(f"❌ Failed to load RAG components: {e}")
        st.info("Make sure you have indexed documents first by running `python vector_rag/scripts/01_index.py`")
        st.stop()

# Question input
question = st.text_input("🔍 Ask a question about the papers:", placeholder="What are the latest trends in robotics?")

if st.button("Get Answer", type="primary") or question:
    if question:
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke({"query": question})
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.write(response["result"])
                
                # Display sources
                if response.get("source_documents"):
                    st.markdown("### 📄 Sources")
                    for i, doc in enumerate(response["source_documents"], 1):
                        with st.expander(f"Source {i}: {doc.metadata.get('Title', 'Unknown')}"):
                            st.write(doc.page_content[:500] + "...")
                            if doc.metadata:
                                st.caption(f"Authors: {doc.metadata.get('Authors', 'N/A')}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("Built with LangChain, ChromaDB, and HuggingFace 🤗")
