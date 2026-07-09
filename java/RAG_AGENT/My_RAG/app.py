import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


import time
import shutil
import streamlit as st
import pandas as pd
from pathlib import Path

# Set Page Config
st.set_page_config(
    page_title="UniTutor AI - College Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif;
    background-color: #0b0f19;
    color: #f3f4f6;
}
[data-testid="stHeader"] {
    background-color: rgba(11, 15, 25, 0.8);
    backdrop-filter: blur(10px);
}
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    text-align: center;
}
.subtitle {
    font-size: 1.1rem;
    color: #9ca3af;
    text-align: center;
    margin-bottom: 2rem;
}
.glass-card {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-bottom: 20px;
}
.metric-container {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    margin-top: 15px;
    margin-bottom: 15px;
}
.metric-box {
    flex: 1;
    background: rgba(31, 41, 55, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-box:hover {
    transform: translateY(-2px);
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-label {
    font-size: 0.85rem;
    color: #9ca3af;
    margin-top: 5px;
}
.success-banner {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 12px;
    padding: 15px;
    color: #34d399;
    font-weight: 600;
    text-align: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Add Imports from local modules
try:
    from My_RAG.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, PINECONE_API_KEY, PINECONE_INDEX_NAME, GOOGLE_API_KEY
    from My_RAG.document_loader import load_document, split_documents
    from My_RAG.embeddings import EmbeddingManager
    from My_RAG.vectorstore import VectorStoreManager
    from My_RAG.agent import run_study_agent
except ImportError:
    # Fallback to local import if path resolution is direct
    from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, PINECONE_API_KEY, PINECONE_INDEX_NAME, GOOGLE_API_KEY
    from document_loader import load_document, split_documents
    from embeddings import EmbeddingManager
    from vectorstore import VectorStoreManager
    from agent import run_study_agent

# Sidebar Setup
with st.sidebar:
    st.markdown("### ⚙️ Ingestion Parameters")
    c_size = st.number_input("Chunk Size", value=CHUNK_SIZE, min_value=100, max_value=5000, step=100)
    c_overlap = st.number_input("Chunk Overlap", value=CHUNK_OVERLAP, min_value=0, max_value=2000, step=50)
    
    st.markdown("---")
    st.markdown("### 📁 Ingested Documents")
    
    # List files from DATA_DIR
    if os.path.exists(DATA_DIR):
        files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
        if files:
            for file in files:
                col_file, col_del = st.columns([5, 1])
                file_ext = os.path.splitext(file)[1].lower()
                icon = "📓" if file_ext in [".ipynb", ".pynb"] else "📄"
                
                with col_file:
                    st.markdown(f"{icon} **{file}**")
                with col_del:
                    if st.button("🗑️", key=f"del_{file}", help=f"Delete {file} from database"):
                        # 1. Delete local file
                        file_path = os.path.join(DATA_DIR, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            
                        # 2. Delete vectors from Pinecone
                        try:
                            vs_manager = VectorStoreManager()
                            vs_manager.delete_document(file)
                            st.toast(f"✅ Deleted {file} successfully!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("No documents uploaded yet.")
    else:
        st.info("No documents uploaded yet.")
    st.markdown("---")
    st.markdown("### 💬 Discussed Questions")
    if "discussed_questions" not in st.session_state:
        st.session_state.discussed_questions = []
        
    if st.session_state.discussed_questions:
        for idx, q in enumerate(st.session_state.discussed_questions):
            # Format display label
            display_label = q
            if q.startswith("Please search the document and generate a 5-question"):
                display_label = "📝 Generated Quiz"
            elif q.startswith("Please search the document and generate 5 multiple-choice"):
                display_label = "❓ Generated MCQs"
            elif q.startswith("Please search the document and extract the most important definitions"):
                display_label = "⭐ Generated Exam Key Points"
            elif len(display_label) > 28:
                display_label = display_label[:25] + "..."
                
            if st.button(display_label, key=f"q_{idx}", use_container_width=True, help=q):
                st.session_state.temp_query = q
                st.rerun()
    else:
        st.info("No questions discussed yet.")


# Main Page Layout
st.markdown('<div class="main-title">🎓 UniTutor AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your ultimate college exam preparation companion, powered by Google Gemini</div>', unsafe_allow_html=True)

# Create two tabs
tab1, tab2 = st.tabs(["📤 Ingestion Dashboard", "💬 Study & Exam Board"])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📤 Document Upload")
    
    uploaded_file = st.file_uploader("Choose a document", type=["pdf", "ipynb", "pynb", "txt"])
    
    if uploaded_file is not None:
        # File details
        file_name = uploaded_file.name
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        
        st.info(f"📁 Selected: **{file_name}** ({file_size_kb:.2f} KB)")
        
        # Ingest Button
        ingest_button = st.button("🚀 Ingest Document", use_container_width=True)
        
        if ingest_button:
            # Create temporary storage path
            os.makedirs(DATA_DIR, exist_ok=True)
            dest_path = os.path.join(DATA_DIR, file_name)
            
            # Save file to disk
            with open(dest_path, "wb") as f:
                f.write(uploaded_file.getvalue())
                
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            # Determine file type
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext == ".pdf":
                file_type_label = "PDF pages"
                file_type = "pdf"
                unit_label = "Pages"
            elif file_ext in [".ipynb", ".pynb"]:
                file_type_label = "Jupyter Notebook cells"
                file_type = "notebook"
                unit_label = "Cells"
            else:
                file_type_label = "text document"
                file_type = "text"
                unit_label = "Documents"
                
            # Step 1: Loading
            status_container.markdown(f"⏳ **Step 1/4:** Loading {file_type_label}...")
            progress_bar.progress(25)
            start_time = time.time()
            
            try:
                docs = load_document(dest_path)
                
                # Enrich metadata
                for doc in docs:
                    doc.metadata['source_file'] = file_name
                    doc.metadata['file_type'] = file_type
                    
                num_loaded = len(docs)
                time.sleep(0.5)  # Visual padding
                
                # Step 2: Chunking
                status_container.markdown("✂️ **Step 2/4:** Splitting documents into chunks...")
                progress_bar.progress(50)
                
                chunks = split_documents(docs, chunk_size=c_size, chunk_overlap=c_overlap)
                num_chunks = len(chunks)
                time.sleep(0.5)
                
                # Step 3: Embeddings
                status_container.markdown("🧠 **Step 3/4:** Generating embeddings (this may take a few seconds)...")
                progress_bar.progress(75)
                
                emb_manager = EmbeddingManager()
                embeddings = emb_manager.get_embeddings()
                
                # Step 4: Save Pinecone
                status_container.markdown("💾 **Step 4/4:** Uploading to Pinecone Vector DB index...")
                progress_bar.progress(90)
                
                vs_manager = VectorStoreManager()
                vs_manager.create_and_save(chunks, embeddings)
                
                # Finish
                progress_bar.progress(100)
                duration = time.time() - start_time
                status_container.empty()
                
                # Success banner
                st.markdown('<div class="success-banner">✨ Data Ingestion Completed Successfully!</div>', unsafe_allow_html=True)
                
                # Stats Metrics
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-box">
                        <div class="metric-value">{num_loaded}</div>
                        <div class="metric-label">{unit_label} Loaded</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{num_chunks}</div>
                        <div class="metric-label">Chunks Generated</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{duration:.2f}s</div>
                        <div class="metric-label">Ingestion Time</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show chunks preview
                if chunks:
                    with st.expander("🔍 View Ingested Chunks Preview"):
                        preview_data = []
                        for i, chunk in enumerate(chunks[:10]):  # Show up to 10 chunks
                            preview_data.append({
                                "Chunk #": i + 1,
                                "Source": chunk.metadata.get("source_file", file_name),
                                "Preview": chunk.page_content[:150] + "..."
                            })
                        st.table(pd.DataFrame(preview_data))
                        if len(chunks) > 10:
                            st.caption(f"Showing first 10 of {len(chunks)} total chunks.")
                            
            except Exception as e:
                progress_bar.empty()
                status_container.empty()
                st.error(f"❌ Error during ingestion: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💬 Interactive Study Board")
    
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        st.warning("⚠️ **Gemini API Key Not Set!** Please configure the `GOOGLE_API_KEY` in your `.env` file to use the interactive study board.")
    else:
        # Prepare chat markdown download string
        chat_markdown = "# 📓 AI Study Tutor - Study Session Log\n\n"
        if "messages" in st.session_state and st.session_state.messages:
            for msg in st.session_state.messages:
                role = "Student" if msg["role"] == "user" else "AI Study Tutor"
                chat_markdown += f"### 👤 {role}\n{msg['content']}\n\n---\n\n"
        else:
            chat_markdown += "*No chat messages in this session yet.*"

        col1, col2, col3, col4, col5 = st.columns(5)
        
        quick_action_prompt = None
        with col1:
            if st.button("📝 Generate Quiz", use_container_width=True):
                quick_action_prompt = "Please search the document and generate a 5-question fill-in-the-blank or short-answer quiz based on the key topics. Put the answers at the very bottom."
        with col2:
            if st.button("❓ Generate MCQs", use_container_width=True):
                quick_action_prompt = "Please search the document and generate 5 multiple-choice questions (MCQs) with options A, B, C, D. Clearly list the correct answers at the bottom."
        with col3:
            if st.button("⭐ Exam Key Points", use_container_width=True):
                quick_action_prompt = "Please search the document and extract the most important definitions, theories, and key points to keep in mind for an exam."
        with col4:
            st.download_button(
                label="📥 Export Chat",
                data=chat_markdown,
                file_name="study_session.md",
                mime="text/markdown",
                use_container_width=True,
                help="Download your full study session chat log as a Markdown file"
            )
        with col5:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.discussed_questions = []
                st.session_state.thread_id = f"rag-study-thread-{int(time.time())}"
                st.rerun()
                
        st.markdown("---")
        
        # Chat memory logic using st.session_state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = f"rag-study-thread-{int(time.time())}"
            
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Handle chat input, quick action, or clicking a discussed question
        user_input = st.chat_input("Ask a question about your uploaded document...")
        if quick_action_prompt:
            user_input = quick_action_prompt
        elif "temp_query" in st.session_state and st.session_state.temp_query:
            user_input = st.session_state.temp_query
            st.session_state.temp_query = None  # Clear it after consumption
            
        if user_input:
            # Add to discussed questions history if not already present
            if "discussed_questions" not in st.session_state:
                st.session_state.discussed_questions = []
            if user_input not in st.session_state.discussed_questions:
                st.session_state.discussed_questions.append(user_input)
                
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Use the dynamic thread_id for conversation history memory
                        response = run_study_agent(user_input, thread_id=st.session_state.thread_id)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
    st.markdown('</div>', unsafe_allow_html=True)

