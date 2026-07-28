import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from pypdf import PdfReader
import os

print("Setting up ChromaDB and Ollama Embeddings...")

# 1. Setup Ollama Embeddings (Runs locally!)
ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

# 2. Setup ChromaDB (Saves locally in a folder)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents", 
    embedding_function=ollama_ef
)

# 3. Read the PDF
pdf_path = "data/sample.pdf"
if not os.path.exists(pdf_path):
    print(f"Error: Please put a file named 'sample.pdf' in the data folder!")
    exit()

print(f"Reading {pdf_path}...")
reader = PdfReader(pdf_path)

# 4. Chunk and Store with Page Citations
print("Chunking and embedding (this might take a minute)...")
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        # Simple chunking: split every 1000 characters with 200 overlap
        chunks = [text[j:j+1000] for j in range(0, len(text), 800)]
        for k, chunk in enumerate(chunks):
            # Create a unique ID for each chunk
            doc_id = f"page{i}_chunk{k}"
            
            # Add to ChromaDB with Page Number as metadata
            collection.add(
                documents=[chunk],
                metadatas=[{"source": pdf_path, "page": i + 1}],
                ids=[doc_id]
            )

print(f"✅ Success! Ingested {len(reader.pages)} pages into offline ChromaDB.")