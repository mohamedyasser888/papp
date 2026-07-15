"""
RAG Pipeline Module
Provides a high-performance Retrieval-Augmented Generation pipeline using 
LangChain, Groq, and Cohere.
"""

import os
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata for a processed document."""
    doc_id: str
    filename: str
    num_chunks: int
    page_count: int

@dataclass
class SourceCitation:
    """Source citation for a generated answer."""
    source: str
    page: int
    content: str
    doc_id: str

@dataclass
class QueryResult:
    """Result of a RAG query."""
    answer: str
    sources: List[SourceCitation]
    session_id: str

@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and cache HuggingFace embeddings.
    Shared across all sessions to avoid redundant downloads.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

class RAGPipeline:
    """
    Main RAG Pipeline handler.
    Handles document processing, indexing, and querying.
    """

    def __init__(self, groq_api_key: str):
        """
        Initialize the RAG pipeline with lazy loading.
        
        Args:
            groq_api_key: API key for Groq LLM.
        """
        self.groq_api_key = groq_api_key
        logger.info("RAG Pipeline initialized with API key")

        # Embeddings are lazy-loaded via cached property
        self._embeddings = None
        
        # Vector store and retriever (initialized on first PDF upload)
        self.vector_store: Optional[FAISS] = None
        self.retriever: Optional[Any] = None

        # LLM initialization (lightweight, no blocking)
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_retries=2,
        )
        logger.info("Groq LLM initialized")

        # Document tracking
        self.processed_documents: List[DocumentMetadata] = []
        
        # In-memory history (can be replaced by external DB if needed)
        self.history: Dict[str, List[BaseMessage]] = {}

        # Compile chains (lightweight, no blocking)
        self._init_chains()
        logger.info("RAG Pipeline chains compiled")

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy load and cache embeddings using Streamlit's resource cache."""
        if self._embeddings is None:
            logger.info("Loading HuggingFace embeddings (first time only)...")
            self._embeddings = get_embeddings()
            logger.info("HuggingFace embeddings loaded successfully")
        return self._embeddings

    def _init_chains(self) -> None:
        """Initialize LangChain Expression Language (LCEL) chains."""
        
        # 1. Contextualization Chain
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone question "
            "which can be understood without the chat history. "
            "Do NOT answer the question — just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        self.contextualize_chain = contextualize_q_prompt | self.llm | StrOutputParser()

        # 2. QA Chain
        qa_system_prompt = """You are an expert document analyst. Use the provided context to answer the user's question.

RULES:
1. Use ONLY the provided context to answer. If the answer isn't there, say you don't know.
2. Structure your answer with clear Markdown headers (##, ###), bullet points, or numbered lists.
3. Keep a professional, analytical tone.
4. For EVERY claim, cite the source: [Source: filename, Page: X].
5. If multiple sources are used, cite all of them.

CONTEXT:
{context}"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        self.qa_chain = qa_prompt | self.llm | StrOutputParser()

    def process_pdf(self, pdf_path: str, original_filename: str) -> str:
        """
        Process a PDF: Load, Split, Embed, and Index.
        
        Args:
            pdf_path: Path to the local PDF file.
            original_filename: Original name of the file for metadata.
            
        Returns:
            doc_id: Unique identifier for the processed document.
        """
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                raise ValueError(f"Empty or unreadable PDF: {original_filename}")

            doc_id = str(uuid.uuid4())
            
            # Metadata attachment
            for page in pages:
                page.metadata["doc_id"] = doc_id
                page.metadata["source"] = original_filename

            # Chunking strategy
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(pages)
            
            if not chunks:
                raise ValueError(f"No text content could be extracted from {original_filename}")

            # Incremental indexing
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vector_store.add_documents(chunks)

            # Update retriever with MMR for diversity
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
            )

            # Track metadata
            metadata = DocumentMetadata(
                doc_id=doc_id,
                filename=original_filename,
                num_chunks=len(chunks),
                page_count=len(pages)
            )
            self.processed_documents.append(metadata)
            
            logger.info(f"Successfully processed {original_filename} ({len(chunks)} chunks)")
            return doc_id

        except Exception as e:
            logger.error(f"Error processing PDF {original_filename}: {str(e)}")
            raise

    def query(self, question: str, session_id: Optional[str] = None) -> QueryResult:
        """
        Execute a RAG query.
        
        Args:
            question: The user's natural language question.
            session_id: Optional session identifier for history tracking.
            
        Returns:
            QueryResult containing answer and citations.
        """
        if not self.retriever:
            raise ValueError("No documents indexed. Please upload a PDF first.")

        session_id = session_id or str(uuid.uuid4())
        
        # Manage history (last 10 messages)
        chat_history = self.history.get(session_id, [])[-10:]
        
        # 1. Contextualize question
        if chat_history:
            standalone_question = self.contextualize_chain.invoke({
                "input": question,
                "chat_history": chat_history
            })
        else:
            standalone_question = question

        # 2. Retrieve relevant context
        docs = self.retriever.invoke(standalone_question)
        
        # 3. Format context string
        context_parts = []
        for d in docs:
            source = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", 0) + 1
            context_parts.append(f"[Source: {source}, Page: {page}]\n{d.page_content}")
        
        formatted_context = "\n\n---\n\n".join(context_parts)

        # 4. Generate answer
        answer = self.qa_chain.invoke({
            "context": formatted_context,
            "input": question,
            "chat_history": chat_history
        })

        # 5. Extract citations
        sources = [
            SourceCitation(
                source=d.metadata.get("source", "Unknown"),
                page=d.metadata.get("page", 0) + 1,
                content=d.page_content,
                doc_id=d.metadata.get("doc_id", "")
            )
            for d in docs
        ]

        # 6. Update history
        chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
        self.history[session_id] = chat_history

        return QueryResult(
            answer=answer,
            sources=sources,
            session_id=session_id
        )

    def clear(self) -> None:
        """Clear all state."""
        self.vector_store = None
        self.retriever = None
        self.processed_documents = []
        self.history = {}
        # We don't clear embeddings as they are reusable and cached at the class instance level
        logger.info("RAG Pipeline state cleared.")

    def has_documents(self) -> bool:
        """Check if any documents are loaded."""
        return len(self.processed_documents) > 0
