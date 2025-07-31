import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from services.gemini_llm import query_gemini  # ✅ GPU-based LLM

UPLOAD_FOLDER = "assets/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        pdf_reader = PdfReader(uploaded_file)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)

@st.cache_resource(show_spinner="🔎 Processing file and creating vector index...")
def process_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore, chunks

def ask_ai_groq(vectorstore, chunks, query):
    top_k_docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in top_k_docs)
    
    prompt = f"""
You are an expert legal assistant. Based on the case notes below, answer the user's question clearly and concisely.

### Case Notes:
{context}

### Question:
{query}

### Answer:
"""
    return query_gemini(prompt)

def handle_input(vectorstore, chunks):
    query = st.session_state.chat_input
    if query:
        with st.spinner("🤖 Thinking..."):
            response = ask_ai_groq(vectorstore, chunks, query)
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("ai", response))
        st.session_state.chat_input = ""  # Clear input box safely

def display_case_analyzer():
    st.markdown("""
        <h2 style='text-align: center;'>📂 AI-Powered Case Note Analyzer</h2>
        <p style='text-align: center; color: gray;'>Upload your legal case files and interact with them like a conversation.</p>
        <hr style='margin-top: 10px;'>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("📘 How it works")
        st.markdown("- Upload a PDF or TXT case file\n"
                    "- Ask questions like:\n"
                    "  - What are the key facts?\n"
                    "  - Any legal risks involved?\n"
                    "- AI answers using your uploaded content.")

    uploaded_file = st.file_uploader("📎 Upload a case file (.pdf or .txt)", type=["pdf", "txt"])

    if uploaded_file:
        if "vectorstore" not in st.session_state or "chunks" not in st.session_state:
            with st.spinner("🧠 Reading and indexing document..."):
                text = extract_text_from_file(uploaded_file)
                vectorstore, chunks = process_text(text)
                st.session_state.vectorstore = vectorstore
                st.session_state.chunks = chunks
                st.session_state.chat_history = []

        st.success("✅ File processed! You can now ask your questions.")
        st.markdown("---")
        st.markdown("### 💬 Ask a question about the case")

        if "chat_input" not in st.session_state:
            st.session_state.chat_input = ""

        st.text_input(
            "", 
            key="chat_input", 
            placeholder="Type your question here...",
            on_change=handle_input, 
            args=(st.session_state.vectorstore, st.session_state.chunks)
        )

        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.markdown("### 🧵 Conversation Thread")
            for speaker, msg in reversed(st.session_state.chat_history):
                if speaker == "user":
                    with st.chat_message("user"):
                        st.markdown(f"**🧑‍💼 You:** {msg}")
                else:
                    with st.chat_message("assistant"):
                        st.markdown(f"<div style='display: flex; align-items: center;'><span style='font-size: 24px;'>💭</span> <span style='margin-left: 8px;'>**AI:** {msg}</span></div>", unsafe_allow_html=True)
