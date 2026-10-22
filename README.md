# Multiple-PDF-Chatbot
An AI-powered application that allows users to chat with multiple PDF documents and receive context-aware answers using Retrieval-Augmented Generation (RAG), LangChain, and free LLMs and embedding models.

# Overview

This project enables users to upload multiple PDF files, ask questions about their content, and receive answers based on relevant information retrieved from the documents.

Instead of sending entire documents to an LLM, the application splits the text into smaller chunks, converts them into embeddings, and retrieves the most relevant chunks for each question.


# Architecture
<img width="2048" height="1035" alt="image" src="https://github.com/user-attachments/assets/08719f97-640a-4683-8f7d-3cfb3912ebd2" />

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

1. PDF Loading: Extract text from uploaded PDF documents.

2. Text Chunking: Split extracted text into smaller, manageable chunks.

3. Embedding Generation: Convert text chunks into vector representations using an embedding model.

4. Vector Storage: Store embeddings in a vector database for efficient similarity search.

5. Question Embedding: Convert the user's question into an embedding using the same embedding model.

6. Semantic Search: Retrieve the document chunks most relevant to the question.

7. Response Generation: Provide the retrieved context and user's question to an LLM to generate a relevant answer.


# Tech Stack

- Python — Core programming language

- LangChain — LLM application development and pipeline orchestration

- Streamlit — Interactive web interface

- PyPDF2 — PDF text extraction

- FAISS — Vector storage and similarity search

- Hugging Face — Access to embedding models and compatible language models

- LLMs — Context-aware answer generation

- RAG — Retrieval-Augmented Generation

- The exact models and libraries depend on the implementation and configuration.


 # Features

- Upload and process multiple PDF documents

- Ask natural-language questions about document content

- Retrieve relevant information through semantic search

- Generate context-aware answers using an LLM

- Build a document-based conversational assistant
