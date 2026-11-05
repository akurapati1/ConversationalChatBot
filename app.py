"""Conversational RAG over uploaded PDFs with per-session chat history (Streamlit + LangChain + Groq)."""
import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question which might reference context in the "
            "chat history, formulate a standalone question which can be understood without the chat "
            "history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved "
            "context to answer the question. If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_pdfs(files) -> list:
    """Read uploaded PDFs through private temp files (no shared/leaked temp.pdf on disk)."""
    documents = []
    for uploaded in files:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            path = tmp.name
        try:
            docs = PyPDFLoader(path).load()
            for d in docs:
                d.metadata["source"] = uploaded.name
            documents.extend(docs)
        finally:
            os.remove(path)
    return documents


def build_retriever(files):
    """Index the PDFs once; rebuild only when the set of uploaded files changes."""
    digest = hashlib.sha256(b"".join(f.getvalue() for f in files)).hexdigest()
    if st.session_state.get("index_digest") != digest:
        with st.spinner("Indexing documents..."):
            splits = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500).split_documents(load_pdfs(files))
            st.session_state.retriever = Chroma.from_documents(splits, get_embeddings()).as_retriever()
            st.session_state.index_digest = digest
    return st.session_state.retriever


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = st.session_state.setdefault("store", {})
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDFs and chat with their content.")

api_key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""))
if not api_key:
    st.warning("Please enter your Groq API key.")
    st.stop()

session_id = st.text_input("Session ID", value="default_session")
files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)
if not files:
    st.info("Upload at least one PDF to start chatting.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name=GROQ_MODEL)
retriever = build_retriever(files)
rag_chain = create_retrieval_chain(
    create_history_aware_retriever(llm, retriever, CONTEXTUALIZE_Q_PROMPT),
    create_stuff_documents_chain(llm, QA_PROMPT),
)
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

history = get_session_history(session_id)
for message in history.messages:
    st.chat_message("user" if message.type == "human" else "assistant").write(message.content)

if question := st.chat_input("Ask a question about your documents"):
    st.chat_message("user").write(question)
    try:
        response = conversational_rag_chain.invoke(
            {"input": question}, config={"configurable": {"session_id": session_id}}
        )
        st.chat_message("assistant").write(response["answer"])
        with st.expander("Sources"):
            for doc in response.get("context", []):
                st.caption(f"{doc.metadata.get('source', 'unknown')} - page {doc.metadata.get('page', '?')}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")
