import re
import tempfile
import pathlib
from typing import List, Dict, Tuple

import streamlit as st
import requests
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Download stopwords quietly and load env vars/API key
nltk.download("stopwords", quiet=True)
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Stroke RAG Agent",
    page_icon="🧠",
    layout="wide"
)

# Initialize OpenAI client
if "client" not in st.session_state:
    st.session_state.client = OpenAI()

# Configuration
PDF_URLS = [
    "https://dnvp9c1uo2095.cloudfront.net/cms-content/030-046l_S2e_Akuttherapie-des-ischaemischen-Schlaganfalls_2022-11-verlaengert_1718363551944.pdf",
    "https://dnvp9c1uo2095.cloudfront.net/cms-content/030133_LL_Sekunda%CC%88rprophylaxe_Teil_1_2022_final_korr_1739803472035.pdf",
    "https://dnvp9c1uo2095.cloudfront.net/cms-content/030143_LL_Sekunda%CC%88rprophylaxe_Teil2_2022_V1.1_1670949892924.pdf",
    "https://tempis.de/download/tempis-sop-2025/?tmstv=1765451597",
]

START_TITLES = ["Was gibt es Neues?", "Die wichtigsten Empfehlungen auf einen Blick"]
EXCLUDE_HEADINGS = ["Literatur", "Referenzen", "References", "Bibliografie"]
CHUNK_SIZE = 950
CHUNK_OVERLAP = 180
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"
MAX_MEMORY = 10

# Persona/system prompt
SYSTEM_PERSONA = (
    "You are a highly experienced neurologist specialized in stroke medicine. "
    "When talking to clinicians, use precise medical terminology and cite key recommendations succinctly. "
    "When talking to patients or families, simplify explanations, avoid jargon, and be empathetic. "
    "Always base answers on retrieved context; if insufficient, say so and request clarification."
)


@st.cache_data
def download_pdfs(urls: List[str], dest_dir: pathlib.Path) -> List[pathlib.Path]:
    """Download all PDFs to a temp folder and return their paths."""
    files = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(urls, 1):
        fname = dest_dir / f"doc_{i}.pdf"
        if not fname.exists():
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            fname.write_bytes(resp.content)
        files.append(fname)
    return files


def page_contains_start(text: str) -> bool:
    """Detect first relevant section titles to start extraction."""
    return any(title.lower() in text.lower() for title in START_TITLES)


def heading_is_excluded(text: str) -> bool:
    """Check if a heading should be skipped (references)."""
    return any(h.lower() in text.lower() for h in EXCLUDE_HEADINGS)


def looks_like_reference_block(text: str) -> bool:
    """Heuristic to drop pages/blocks that are mostly references."""
    ref_patterns = [r"\[[0-9]{1,3}\]", r"\([0-9]{1,3}\)", r"\d{4}\."]
    hits = sum(len(re.findall(p, text)) for p in ref_patterns)
    density = hits / max(1, len(text.split()))
    return density > 0.08 or heading_is_excluded(text[:80])


def clean_text(text: str) -> str:
    """Light cleaning: drop URLs, citations, page numbers, extra whitespace."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b\d+\s*/\s*\d+\b", " ", text)  # page numbers like 12/34
    text = re.sub(r"\s+\d+\s+", " ", text)  # lone numbers
    text = re.sub(r"\[[0-9]+\]|\([0-9]+\)|\^{[0-9]+}", " ", text)  # citation markers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_relevant_text(pdf_path: pathlib.Path) -> str:
    """Extract text starting at first target section; skip reference-like pages."""
    doc = fitz.open(pdf_path)
    collecting = False
    blocks = []
    for page in doc:
        txt = page.get_text("text") or ""
        if not collecting and page_contains_start(txt):
            collecting = True
        if not collecting:
            continue  # skip intro pages
        if looks_like_reference_block(txt):
            continue  # drop reference pages/sections
        cleaned = clean_text(txt)
        if cleaned:
            blocks.append(cleaned)
    doc.close()
    return "\n".join(blocks)


@st.cache_data
def process_pdfs(pdf_paths: List[pathlib.Path]) -> Tuple[List[str], List[Dict]]:
    """Extract and chunk text from PDFs."""
    stop_words = set(stopwords.words("german")) | set(stopwords.words("english"))
    
    documents = []
    for p in pdf_paths:
        extracted = extract_relevant_text(p)
        documents.append(extracted)
    
    # Combine docs and split into overlapping chunks for retrieval
    all_text = "\n".join(documents)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    chunks = splitter.split_text(all_text)
    
    # Simple metadata mapping per chunk
    metadatas = [{"source": f"chunk_{i}"} for i in range(len(chunks))]
    
    return chunks, metadatas


@st.cache_resource
def build_vector_store(chunks: List[str], metadatas: List[Dict]) -> Tuple[faiss.Index, Dict[int, str], Dict[int, Dict]]:
    """Build FAISS index from chunks."""
    def embed_texts(texts: List[str]) -> np.ndarray:
        """Batch-embed all chunks using OpenAI embeddings."""
        embeddings = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i+64]
            resp = st.session_state.client.embeddings.create(model=EMBED_MODEL, input=batch)
            embeddings.extend([item.embedding for item in resp.data])
        return np.array(embeddings, dtype="float32")
    
    # Compute embeddings and build FAISS index
    emb_matrix = embed_texts(chunks)
    index = faiss.IndexFlatL2(emb_matrix.shape[1])
    index.add(emb_matrix)
    
    # Store mappings for retrieval
    id2text = {i: t for i, t in enumerate(chunks)}
    id2meta = {i: m for i, m in enumerate(metadatas)}
    
    return index, id2text, id2meta


def retrieve(query: str, index: faiss.Index, id2text: Dict, id2meta: Dict, k: int = 5) -> List[Tuple[str, Dict]]:
    """Vector search top-k chunks for a query."""
    q_emb = np.array(
        st.session_state.client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding,
        dtype="float32"
    )
    q_emb = np.expand_dims(q_emb, axis=0)
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append((id2text[idx], id2meta[idx]))
    return results


def build_context(chunks: List[Tuple[str, Dict]]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for i, (text, meta) in enumerate(chunks, 1):
        parts.append(f"[Chunk {i}] {text}")
    return "\n\n".join(parts)


def agent_answer(query: str, context: str) -> str:
    """Generate a draft answer using persona + retrieved context."""
    msgs = [
        {"role": "system", "content": SYSTEM_PERSONA},
        {"role": "system", "content": "Use the provided context. If context is thin, say so and request clarification."},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user", "content": query},
    ]
    resp = st.session_state.client.chat.completions.create(model=CHAT_MODEL, messages=msgs)
    return resp.choices[0].message.content.strip()


def reflect_and_improve(response: str, query: str, context: str) -> str:
    """Check medical quality; if feedback not OK, regenerate with fixes."""
    reflection = st.session_state.client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are checking whether the response is medically accurate, complete, safe, and well-structured. Respond with 'OK' or provide concrete corrections and missing points.",
            },
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}\n\nDraft response:\n{response}"},
        ],
    )
    feedback = reflection.choices[0].message.content.strip()
    if feedback.upper() == "OK":
        return response
    # Regenerate with feedback injected
    regen_msgs = [
        {"role": "system", "content": SYSTEM_PERSONA},
        {"role": "system", "content": "Use the provided context. If context is thin, say so and ask for clarification."},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "system", "content": f"Please fix issues noted: {feedback}"},
        {"role": "user", "content": query},
    ]
    regen = st.session_state.client.chat.completions.create(model=CHAT_MODEL, messages=regen_msgs)
    return regen.choices[0].message.content.strip()


def build_memory_context() -> str:
    """Build context from conversation memory."""
    if "memory_buffer" not in st.session_state:
        return ""
    if not st.session_state.memory_buffer:
        return ""
    formatted = []
    for i, (q, a) in enumerate(st.session_state.memory_buffer, 1):
        formatted.append(f"[Memory {i}] Q: {q}\nA: {a}")
    return "\n\n".join(formatted)


def add_to_memory(question: str, answer: str) -> None:
    """Add Q&A pair to memory buffer."""
    if "memory_buffer" not in st.session_state:
        st.session_state.memory_buffer = []
    st.session_state.memory_buffer.append((question, answer))
    if len(st.session_state.memory_buffer) > MAX_MEMORY:
        st.session_state.memory_buffer.pop(0)


def workflow(query: str, index: faiss.Index, id2text: Dict, id2meta: Dict, k: int = 5) -> str:
    """Full RAG pipeline: retrieve → draft → reflect → final (with memory)."""
    retrieved = retrieve(query, index, id2text, id2meta, k=k)
    retrieval_context = build_context(retrieved)
    memory_context = build_memory_context()

    # Combine memory + retrieval context for richer answers
    if memory_context:
        combined_context = f"{memory_context}\n\n{retrieval_context}"
    else:
        combined_context = retrieval_context

    draft = agent_answer(query, combined_context)
    final = reflect_and_improve(draft, query, combined_context)

    # Persist this turn in memory
    add_to_memory(query, final)
    return final


def initialize_system():
    """Initialize the RAG system (download PDFs, process, and build vector store)."""
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False
    
    if not st.session_state.vector_store_ready:
        with st.spinner("Initializing system... This may take a few minutes."):
            # Create temp directory
            if "temp_dir" not in st.session_state:
                st.session_state.temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="rag_pdfs_"))
            
            # Download PDFs
            with st.spinner("Downloading PDFs..."):
                pdf_paths = download_pdfs(PDF_URLS, st.session_state.temp_dir)
            
            # Process PDFs
            with st.spinner("Processing PDFs and extracting text..."):
                chunks, metadatas = process_pdfs(pdf_paths)
            
            # Build vector store
            with st.spinner("Building vector store and computing embeddings..."):
                index, id2text, id2meta = build_vector_store(chunks, metadatas)
                st.session_state.index = index
                st.session_state.id2text = id2text
                st.session_state.id2meta = id2meta
                st.session_state.vector_store_ready = True
                st.success(f"System ready! Loaded {len(chunks)} chunks from {len(pdf_paths)} PDFs.")


def main():
    """Main Streamlit app."""
    st.title("🧠 Stroke RAG Agent")
    st.markdown("**An openai-based agent to interact with the German Stroke Guidelines**")
    st.markdown("---")
    
    # Initialize system
    initialize_system()
    
    if not st.session_state.vector_store_ready:
        st.warning("System is still initializing. Please wait...")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        if st.button("Clear Conversation Memory", use_container_width=True):
            if "memory_buffer" in st.session_state:
                st.session_state.memory_buffer = []
            st.success("Memory cleared!")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This agent is powered by:
        - **RAG (Retrieval-Augmented Generation)** for accurate guideline retrieval
        - **Reflection mechanism** for quality assurance
        - **Conversation memory** for context awareness
        
        The agent adapts its responses based on whether you're a clinician or patient.
        """)
    
    # Chat interface
    st.header("💬 Chat with the Stroke Guidelines Agent")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about stroke guidelines..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Checking the guidelines..."):
                answer = workflow(
                    prompt,
                    st.session_state.index,
                    st.session_state.id2text,
                    st.session_state.id2meta
                )
                st.markdown(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

