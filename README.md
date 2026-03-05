# RAG: Document Question Answering Pipeline

A end-to-end **Retrieval-Augmented Generation (RAG)** pipeline implemented in a Google Colab notebook using LangChain. The notebook covers every stage of the pipeline — from loading and chunking documents to multi-turn conversational Q&A — and is designed to run entirely for free.

---

## Overview

RAG improves on standalone language models by grounding responses in a retrieved knowledge base rather than relying solely on parametric memory. This reduces hallucinations and allows the model to answer questions about documents it was never trained on.

```
Documents -> Split -> Embed -> VectorStore
                                    |
User Query -> Retrieve -> LLM -> Answer
```

---

## Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Document loading | `WebBaseLoader`, `PyPDFLoader` | URLs and PDFs |
| Text splitting | `RecursiveCharacterTextSplitter` | Configurable chunk size and overlap |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) | Runs locally on CPU, no API key |
| Vector store | ChromaDB (Parts 1–6), FAISS (Part 7) | ChromaDB persists to disk |
| LLM | Groq free tier | No credit card required |
| Framework | LangChain | `langchain`, `langchain-community`, `langchain-classic` |

---

## Requirements

**API key:** A Groq API key is required. Create one for free (no credit card) at [console.groq.com/keys](https://console.groq.com/keys).

**Runtime:** Google Colab CPU is sufficient. No GPU is needed.

**Python packages** (installed automatically in the first notebook cell):

```
langchain
langchain-community
langchain-groq
langchain-huggingface
langchain-classic
chromadb
sentence-transformers
pypdf
beautifulsoup4
lark
scikit-learn
docarray
faiss-cpu
einops
```

---

## Notebook Structure

### Setup
- Installs all dependencies
- Configures the Groq API key
- Initialises the embedding model (`all-MiniLM-L6-v2`) and LLM
- Defines `safe_invoke()`, a wrapper that handles Groq rate-limit errors (HTTP 429) with automatic retry and backoff

### Part 1 — Document Loading
Loads two knowledge sources:
- Four Wikipedia articles: Machine Learning, Deep Learning, Natural Language Processing, Large Language Models
- The "Attention Is All You Need" paper (arXiv PDF, downloaded at runtime)

### Part 2 — Document Splitting
Demonstrates three splitting strategies:
- `RecursiveCharacterTextSplitter` — splits on paragraph, sentence, then word boundaries
- `CharacterTextSplitter` — splits on a single separator character
- `MarkdownHeaderTextSplitter` — preserves Markdown header hierarchy as chunk metadata

All loaded documents are split into 1000-character chunks with 150-character overlap for indexing.

### Part 3 — Vector Stores and Embeddings
- Explains cosine similarity with a worked example
- Builds a persistent ChromaDB vector store from all document chunks
- Demonstrates standard similarity search
- Demonstrates Maximum Marginal Relevance (MMR) search to return diverse results

### Part 4 — Advanced Retrieval
Three retrieval techniques:

| Technique | Description |
|-----------|-------------|
| Metadata filtering | Restricts search to a specific source (e.g. PDF only) |
| LLM-guided source routing | LLM classifies the query and selects the appropriate source to search |
| Contextual compression | Retrieved chunks are compressed to only the relevant excerpt before being passed to the LLM |

### Part 5 — Question Answering
- Basic `RetrievalQA` chain
- Custom prompt template with source attribution
- Comparison of `stuff`, `map_reduce`, and `refine` chain types
- Demonstration of the stateless limitation of `RetrievalQA`

### Part 6 — Conversational Chat with Memory
- `ConversationalRetrievalChain` with `ConversationBufferMemory`
- Follow-up questions are rephrased in context before retrieval
- Configurable chatbot with a custom prompt and MMR retrieval
- `chat_with_your_pdf()` function for uploading and querying an arbitrary PDF

### Part 7 — Alternative Stack (FAISS + Nomic Embeddings)
An alternative pipeline for querying an uploaded PDF:
- `nomic-embed-text-v1.5` embeddings (higher retrieval accuracy than `all-MiniLM-L6-v2`)
- FAISS in-memory vector store (no persistence, faster setup)
- Same Groq LLM reused from setup

---

## Configuration

### Switching models

The LLM is configured via a single variable at the top of the setup cell:

```python
GROQ_MODEL = "llama-3.1-8b-instant"  # change this to switch models
```

Available models on the Groq free tier:

| Model | Speed | Quality | Context window |
|-------|-------|---------|----------------|
| `llama-3.1-8b-instant` | Fastest | Good | 128k |
| `llama-3.3-70b-versatile` | Medium | Best | 128k |
| `mixtral-8x7b-32768` | Medium | Very good | 32k |
| `gemma2-9b-it` | Fast | Good | 8k |

### Rate limiting

The Groq free tier allows approximately 30 requests per minute (varies by model). The notebook handles this in two ways:

1. `safe_invoke()` retries automatically on HTTP 429 errors with exponential backoff
2. `time.sleep(4)` calls between sequential LLM requests keep throughput within the free tier limit

If a daily quota is reached, switch to a different model in the `GROQ_MODEL` variable.

---

## Knowledge Base

The default knowledge base (Parts 1–6) consists of:

- [Machine Learning — Wikipedia](https://en.wikipedia.org/wiki/Machine_learning)
- [Deep Learning — Wikipedia](https://en.wikipedia.org/wiki/Deep_learning)
- [Natural Language Processing — Wikipedia](https://en.wikipedia.org/wiki/Natural_language_processing)
- [Large Language Model — Wikipedia](https://en.wikipedia.org/wiki/Large_language_model)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

To use a different knowledge base, replace the URLs in Part 1 or modify `chat_with_your_pdf()` in Part 6.

---

## Known Issues

- `ContextualCompressionRetriever` and `LLMChainExtractor` must be imported from `langchain_classic`, not `langchain` or `langchain_community`, depending on the installed package versions. The notebook uses the correct import path.
- `get_relevant_documents()` is deprecated in recent LangChain versions. The notebook uses `.invoke()` throughout.
- `PromptTemplate` must be imported from `langchain_core.prompts`, not `langchain.prompts`.

---

## Possible Extensions

- Load additional source types: local files, databases, Notion, Google Drive
- Add a retrieval re-ranking step with `CrossEncoderReranker` for improved precision
- Replace ChromaDB with a persistent cloud vector store (Pinecone, Weaviate, Qdrant)
- Build a web interface with [Gradio](https://gradio.app/) or [Streamlit](https://streamlit.io/)
- Upgrade to `llama-3.3-70b-versatile` for higher quality answers on complex queries
