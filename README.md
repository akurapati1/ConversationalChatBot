# ConversationalChatBot

A **conversational RAG** (retrieval-augmented generation) chatbot: upload one or more PDFs and ask follow-up questions in natural language. Built with **Streamlit**, **LangChain**, **ChromaDB**, **HuggingFace embeddings** and **Groq** LLMs.

## How it works
1. PDFs are read with `PyPDFLoader` and split into overlapping chunks.
2. Chunks are embedded (`all-MiniLM-L6-v2`) and indexed in Chroma — only re-indexed when the uploaded files change.
3. A **history-aware retriever** rewrites each follow-up question into a standalone query using the chat history.
4. The retrieved chunks are stuffed into a QA prompt and answered by a Groq-hosted model; sources are shown under each answer.
5. History is kept per **Session ID**, so you can run several independent conversations.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # optionally pre-fill GROQ_API_KEY (https://console.groq.com)
streamlit run app.py
```
You can also paste the Groq key into the UI instead of using `.env`.

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | – | Groq key (or enter in the UI) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Chat model |

## Tech stack
Streamlit · LangChain · ChromaDB · sentence-transformers · Groq
