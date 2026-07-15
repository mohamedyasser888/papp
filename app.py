import streamlit as st
import os
import tempfile
import uuid
import hashlib
import logging
from rag_pipeline import RAGPipeline

# Configure logging for Streamlit Cloud debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Application starting...")


# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF - RAG Application",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
logger.info("Page configuration complete")

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: #f8fafc;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Glass card ── */
.main-container {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 1.25rem;
    padding: 2.25rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

/* ── Gradient headings ── */
h1, h2, h3 {
    background: linear-gradient(to right, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
    background-color: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 1rem !important;
    padding: 1.25rem !important;
    margin-bottom: 1.25rem !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stChatMessage"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* ── Source citation cards ── */
.source-card {
    background: rgba(245, 158, 11, 0.08) !important;
    border-left: 4px solid #f59e0b !important;
    border-right: 1px solid rgba(245, 158, 11, 0.15) !important;
    border-top: 1px solid rgba(245, 158, 11, 0.15) !important;
    border-bottom: 1px solid rgba(245, 158, 11, 0.15) !important;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-top: 0.75rem;
    color: #fef3c7;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* ── Document badge ── */
.doc-badge {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.5rem !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.45) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-container'>
    <h1 style='text-align: center; margin-bottom: 0.5rem;'>📄 Chat with PDF</h1>
    <p style='text-align: center; color: #94a3b8; margin-top: 0;'>
        Upload PDF documents and ask questions about their content using a
        high-performance RAG pipeline powered by Groq · Llama 3.3 · HuggingFace Embeddings.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Session state initialisation ───────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("documents", []),
    ("processed_file_hashes", set()),
]:
    if key not in st.session_state:
        st.session_state[key] = default
logger.info("Session state initialized")


# ── Helper: resolve an API key (secrets → env → sidebar input) ────────────
def _resolve_key(secret_name: str) -> str | None:
    """Return key from Streamlit secrets or .env, or None."""
    try:
        val = st.secrets.get(secret_name, "")
        if val and val.strip() not in ("", f"your_{secret_name.lower()}_here"):
            return val.strip()
    except Exception:
        pass
    val = os.getenv(secret_name, "")
    if val and val.strip():
        return val.strip()
    return None


# ── Sidebar — API keys ─────────────────────────────────────────────────────
groq_key = _resolve_key("GROQ_API_KEY")

with st.sidebar:
    st.markdown("### 🔑 API Configuration")

    # ── Groq key ──
    if not groq_key:
        groq_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your key from the Groq console",
        )
        st.markdown(
            "[👉 **Get a free Groq API Key**](https://console.groq.com/keys)",
            unsafe_allow_html=True,
        )
        if groq_input and groq_input.strip().startswith("gsk_"):
            groq_key = groq_input.strip()
        elif groq_input:
            st.warning("⚠️ Groq keys start with `gsk_`.")
    else:
        st.success("✅ Groq API Key loaded")

# ── Halt if keys are missing ───────────────────────────────────────────────
if not groq_key:
    st.warning(
        "⚠️ **Groq API Key required.** "
        "Please enter it in the sidebar to start."
    )
    st.info("💡 Get a **free** Groq key → [console.groq.com/keys](https://console.groq.com/keys)")
    st.stop()

# Expose keys to environment so libraries can pick them up

# ── Initialise / refresh RAG pipeline ─────────────────────────────────────
if "rag_pipeline" not in st.session_state:
    try:
        logger.info("Initializing RAG pipeline...")
        st.session_state.rag_pipeline = RAGPipeline(
            groq_api_key=groq_key,
        )
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        st.sidebar.error(f"❌ Failed to initialise RAG pipeline: {e}")
        st.stop()

# ── Sidebar — Document management ─────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyse",
    )

    if uploaded_files:
        new_files = []
        for file in uploaded_files:
            file_bytes = file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            if file_hash not in st.session_state.processed_file_hashes:
                new_files.append((file, file_hash, file_bytes))

        if new_files:
            is_first = len(st.session_state.processed_file_hashes) == 0
            spinner_msg = (
                "⏳ Loading embeddings & processing PDF..."
                if is_first else
                "📄 Processing PDF..."
            )
            with st.spinner(spinner_msg):
                for file, file_hash, file_bytes in new_files:
                    temp_path = None
                    try:
                        unique_id = uuid.uuid4().hex
                        temp_path = os.path.join(
                            tempfile.gettempdir(), f"temp_{unique_id}.pdf"
                        )
                        with open(temp_path, "wb") as f:
                            f.write(file_bytes)

                        doc_id = st.session_state.rag_pipeline.process_pdf(
                            temp_path, original_filename=file.name
                        )
                        st.session_state.documents.append({
                            "filename": file.name,
                            "doc_id": doc_id,
                        })
                        st.session_state.processed_file_hashes.add(file_hash)
                        st.success(f"✅ {file.name} processed")
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {e}")
                    finally:
                        if temp_path and os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except Exception:
                                pass

    # Uploaded documents list
    if st.session_state.documents:
        st.markdown("### 📚 Uploaded Documents")
        for doc in st.session_state.documents:
            st.markdown(f"""
                <div class="doc-badge">
                    <span>📄</span>
                    <span style="font-size:0.9rem;font-weight:500;
                                 overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                        {doc['filename']}
                    </span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown(f"**Total:** {len(st.session_state.documents)} document(s)")

    # Clear button
    st.markdown("---")
    if st.button("🗑️ Clear All Documents", use_container_width=True, type="secondary"):
        st.session_state.rag_pipeline.clear()
        st.session_state.documents = []
        st.session_state.messages = []
        st.session_state.processed_file_hashes.clear()
        st.session_state.pop("session_id", None)
        st.success("All documents cleared!")
        st.rerun()

    # Session stats
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    st.metric("Messages", len(st.session_state.messages))

# ── Main chat area ─────────────────────────────────────────────────────────
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("### 💬 Chat")
st.markdown("---")

if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"📚 View {len(message['sources'])} Source(s)", expanded=False):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                            <div class="source-card">
                                <strong>📄 Source {idx}: {source['source']} — Page {source['page']}</strong><br>
                                <small style='opacity:0.95;'>{source['content']}</small>
                            </div>
                        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='text-align:center;padding:3rem;color:#64748b;'>
        <h3>👋 Welcome!</h3>
        <p>Upload one or more PDF documents in the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("💬 Ask a question about your documents...", key="chat_input"):
    if not st.session_state.rag_pipeline.has_documents():
        st.error("⚠️ Please upload a PDF document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_pipeline.query(
                        prompt, st.session_state.get("session_id")
                    )
                    st.session_state.session_id = result["session_id"]
                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander(
                            f"📚 View {len(result['sources'])} Source(s)", expanded=False
                        ):
                            for idx, source in enumerate(result["sources"], 1):
                                st.markdown(f"""
                                    <div class="source-card">
                                        <strong>📄 Source {idx}: {source['source']} — Page {source['page']}</strong><br>
                                        <small style='opacity:0.95;'>{source['content']}</small>
                                    </div>
                                """, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {e}",
                    })

        st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='
    text-align:center;
    padding:2rem 1rem 1.5rem;
    margin-top:1rem;
    border-top:1px solid rgba(255,255,255,0.06);
'>
    <small style='color:#475569;'>
        ⚡ Built with Streamlit · LangChain · FAISS · Groq API · HuggingFace Embeddings · Llama 3.3 70B
    </small>
    <br><br>
    <span style='
        display:inline-block;
        background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(168,85,247,0.15));
        border:1px solid rgba(99,102,241,0.3);
        border-radius:2rem;
        padding:0.4rem 1.25rem;
        font-size:0.82rem;
        color:#a5b4fc;
        letter-spacing:0.03em;
    '>
        ✦ Made with ❤️ by <strong style="color:#c4b5fd;">Mohamed Yasser</strong>
    </span>
</div>
""", unsafe_allow_html=True)