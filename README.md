
# Offline AI Agent

An offline AI Agent built using Llama 3.1, Ollama, ChromaDB and Streamlit.

## Features

- Local LLM using Ollama
- Tool Calling
- Retrieval-Augmented Generation (RAG)
- PDF Upload
- ChromaDB Vector Database
- Calculator Tool
- Dataset Inspector
- Chat Memory
- Streamlit UI

## Architecture

(image)

## Technologies

- Python
- Ollama
- Llama 3.1
- Streamlit
- ChromaDB
- PyPDF
- Nomic Embeddings

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/offline-ai-agent.git
```

### 2. Go to the project folder

```bash
cd offline-ai-agent
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama

```bash
ollama serve
```

### 5. Pull the required models

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 6. Run the application

```bash
streamlit run app.py
```

...
