# Chat with PDF - RAG Application

A production-ready Streamlit web application that allows you to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG) with Groq API.

## Features

- **PDF Upload**: Upload one or multiple PDF files via sidebar
- **Text Extraction**: Uses PDFLoader to extract text from PDFs
- **Text Chunking**: Employs RecursiveCharacterTextSplitter for optimal chunking
- **Embeddings**: Generates embeddings using HuggingFace sentence-transformers (lazy-loaded)
- **Vector Database**: Stores and retrieves chunks using FAISS
- **RAG with Groq**: Uses Groq's Llama 3.3 70B model for intelligent question answering
- **Source Citations**: Provides citations with source document and page numbers
- **Conversation Memory**: Maintains conversation context across multiple questions
- **Modern UI**: Clean Streamlit interface with glass-morphism design
- **Production-Ready**: Optimized for Streamlit Community Cloud deployment

## Tech Stack

- **Framework**: Streamlit >= 1.40.0
- **LLM**: Groq API (Llama 3.3 70B Versatile)
- **LangChain**: >= 0.3.0 (modern LCEL chains)
- **Embeddings**: langchain-huggingface >= 0.1.0 (sentence-transformers all-MiniLM-L6-v2)
- **Vector Store**: FAISS >= 1.8.0
- **PDF Processing**: PyPDF >= 5.0.0
- **Python**: 3.12+

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Groq API key in the `.env` file:
```
GROQ_API_KEY=gsk_your_api_key_here
```

## Running the Application

Run the Streamlit application:
```bash
streamlit run app.py
```

Or using Python module:
```bash
python -m streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

1. Enter your Groq API key in the sidebar (or set it via environment variable)
2. Upload PDF files using the sidebar uploader
3. Wait for the documents to be processed (embeddings load on first upload)
4. Type your question in the chat input at the bottom
5. View the answer with source citations displayed below

## Streamlit Community Cloud Deployment

### Prerequisites

1. **GitHub Repository**: Push your code to a GitHub repository
2. **Groq API Key**: Add your API key to Streamlit secrets

### Deployment Steps

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app" and connect your GitHub repository
3. Configure your app:
   - **Repository**: Select your repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app.py`
4. In the "Secrets" section, add:
   ```
   GROQ_API_KEY=gsk_your_api_key_here
   ```
5. Click "Deploy"

### Important Notes for Cloud Deployment

- **Lazy Loading**: Embeddings are loaded on first PDF upload, not at startup
- **Cached Resources**: HuggingFace embeddings are cached using `@st.cache_resource` for efficiency
- **Logging**: Detailed logging is enabled for debugging deployment issues
- **No Blocking Operations**: UI appears immediately; heavy resources load only when needed

### Troubleshooting Cloud Deployment

If the app shows a blank screen on Streamlit Cloud:

1. Check the deployment logs for error messages
2. Verify your `GROQ_API_KEY` is set in secrets
3. Ensure all dependencies in `requirements.txt` are compatible
4. Look for log messages indicating where initialization stopped

## Migration to LangChain 0.3+

This project has been upgraded to use the latest LangChain 0.3+ APIs and Python 3.12 ecosystem.

### Key Changes:

**Imports Updated:**
- `langchain.text_splitter` → `langchain_text_splitters`
- `langchain.prompts` → `langchain_core.prompts`
- `langchain.memory` → Manual message history management
- `langchain.chains.ConversationalRetrievalChain` → LCEL chains
- `langchain_community.embeddings.HuggingFaceEmbeddings` → `langchain_huggingface.HuggingFaceEmbeddings`

**Chain Architecture:**
- Replaced deprecated `ConversationalRetrievalChain` with modern LCEL (LangChain Expression Language) chains
- Uses manual chain composition with `ChatPromptTemplate`, `StrOutputParser`
- Manual conversation history management using `HumanMessage` and `AIMessage` from `langchain_core.messages`
- Lazy loading of embeddings to prevent startup blocking

**Performance Optimizations:**
- Embeddings cached with `@st.cache_resource` decorator
- Embeddings load only on first PDF upload, not at startup
- LLM initialization is lightweight and non-blocking
- Detailed logging for debugging deployment issues

**Dependencies:**
- Added `langchain-core>=0.3.0` for core primitives
- Added `langchain-text-splitters>=0.3.0` for text splitting utilities
- Added `langchain-huggingface>=0.1.0` for modern embeddings
- Removed `langchain-cohere` (not needed, using local embeddings)
- Added `sentence-transformers>=3.0.0` and `torch>=2.0.0` for embeddings
- All packages updated to latest stable versions compatible with Python 3.12

### Files Modified:
- `rag_pipeline.py` - Complete rewrite using modern LangChain APIs, lazy loading, and logging
- `app.py` - Updated API key handling, removed Cohere dependencies, added logging
- `requirements.txt` - Updated all package versions for Python 3.12 compatibility

### No Deprecated APIs:
All imports and APIs are now using the modern LangChain 0.3+ ecosystem. No deprecated patterns remain.

## Project Structure

```
.
├── app.py                  # Streamlit application (production-ready)
├── rag_pipeline.py         # RAG pipeline implementation (modernized)
├── requirements.txt        # Python dependencies (updated for cloud)
├── .env                    # Environment variables (local development)
└── README.md              # This file
```

## Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required)
  - Get it from: https://console.groq.com/keys
  - Can be set in `.env` file locally or in Streamlit secrets for cloud deployment
