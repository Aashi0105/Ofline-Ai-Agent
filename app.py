import streamlit as st
import ollama
import ast
import operator
import logging
import os
import io
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from pypdf import PdfReader

# ==========================================
# 1. PAGE CONFIG & LOGGING
# ==========================================
st.set_page_config(page_title="Offline AI Agent", page_icon="🤖", layout="wide")

if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('data'):
    os.makedirs('data')
logging.basicConfig(filename='logs/agent.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ==========================================
# 2. CHROMADB & TOOLS SETUP
# ==========================================
@st.cache_resource
def get_chroma_collection():
    ollama_ef = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name="nomic-embed-text")
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="documents", embedding_function=ollama_ef)

collection = get_chroma_collection()

# Safe Calculator
ALLOWED_OPERATORS = { ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg }
def safe_calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode='eval')
        def _eval(node):
            if isinstance(node, ast.Constant): return node.value
            elif isinstance(node, ast.BinOp): return ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp): return ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
            else: raise ValueError("Unsupported")
        return str(_eval(tree.body))
    except Exception as e: return f"Error calculating: {e}"

# RAG Search with Citations
def search_documents(query: str) -> str:
    try:
        results = collection.query(query_texts=[query], n_results=2)
        if not results['documents'][0]: return "No relevant documents found."
        output = ""
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            page_num = meta.get('page', 'Unknown')
            output += f"Content: {doc}\n(Source: Page {page_num})\n\n"
        return output.strip()
    except Exception as e:
        return f"Error searching documents: {e}"

# Dataset Inspector
def inspect_dataset(csv_path: str) -> str:
    try:
        df = pd.read_csv(csv_path)
        buffer = io.StringIO()
        df.info(buf=buffer)
        return f"Dataset Info:\n{buffer.getvalue()}\n\nFirst 3 rows:\n{df.head(3)}\n\nStatistics:\n{df.describe()}"
    except Exception as e: return f"Error reading CSV: {e}"

# Tool Schema
tools_schema = [
    {"type": "function", "function": {"name": "safe_calculator", "description": "Evaluates mathematical expressions with numbers ONLY.", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. '25 * 4'"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search_documents", "description": "Searches local offline database for information. Returns text and page numbers.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "inspect_dataset", "description": "Reads a CSV file and returns its columns and statistics.", "parameters": {"type": "object", "properties": {"csv_path": {"type": "string", "description": "Path to CSV file, e.g. 'data/data.csv'"}}, "required": ["csv_path"]}}}
]

# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful offline AI. Use tools when necessary. Cite page numbers if searching documents. For CSVs, use path 'data/data.csv'."}]
if "uploaded_pdf_name" not in st.session_state:
    st.session_state.uploaded_pdf_name = None
if "uploaded_csv_name" not in st.session_state:
    st.session_state.uploaded_csv_name = None

# ==========================================
# 4. STREAMLIT SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Agent Controls")
    
    # Knowledge Base Counter
    db_count = collection.count()
    st.metric("📄 Knowledge Base Chunks", db_count)
    
    st.divider()

    # PDF Uploader (FIXED: Clears old DB and saves file properly)
    st.subheader("Upload Data")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")
    if uploaded_pdf and uploaded_pdf.name != st.session_state.uploaded_pdf_name:
        with st.spinner("Ingesting new PDF... (Clearing old memory first)"):
            try:
                # 1. Delete old documents so it ONLY knows the new PDF
                collection.delete(where={"source": {"$ne": ""}}) # Deletes everything
                
                # 2. Save uploaded file to disk so PyPDF can read it reliably
                save_path = os.path.join("data", uploaded_pdf.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                
                # 3. Read and Ingest
                reader = PdfReader(save_path)
                all_chunks, all_metadatas, all_ids = [], [], []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        chunks = [text[j:j+1000] for j in range(0, len(text), 800)]
                        for k, chunk in enumerate(chunks):
                            all_chunks.append(chunk)
                            all_metadatas.append({"source": uploaded_pdf.name, "page": i + 1})
                            all_ids.append(f"{uploaded_pdf.name}_p{i}_c{k}")
                
                if all_chunks:
                    collection.add(documents=all_chunks, metadatas=all_metadatas, ids=all_ids)
                
                st.session_state.uploaded_pdf_name = uploaded_pdf.name
                st.success(f"✅ Added {len(reader.pages)} pages!")
            except Exception as e:
                st.error(f"Error ingesting PDF: {e}")

    # CSV Uploader
    uploaded_csv = st.file_uploader("Upload CSV Dataset", type="csv", key="csv_uploader")
    if uploaded_csv and uploaded_csv.name != st.session_state.uploaded_csv_name:
        try:
            csv_save_path = os.path.join("data", "data.csv")
            with open(csv_save_path, "wb") as f:
                f.write(uploaded_csv.getbuffer())
            st.session_state.uploaded_csv_name = uploaded_csv.name
            st.success("✅ CSV saved to memory!")
        except Exception as e:
            st.error(f"Error saving CSV: {e}")

    st.divider()

    # Clear Chat Button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [{"role": "system", "content": "You are a helpful offline AI. Use tools when necessary. Cite page numbers if searching documents. For CSVs, use path 'data/data.csv'."}]
        st.rerun()

# ==========================================
# 5. STREAMLIT MAIN CHAT UI
# ==========================================
st.title("🤖 Offline AI Agent")
st.caption("100% Local • Custom Agentic Loop • No APIs • Built from Scratch")

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant" and "content" in msg and msg["content"]:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask your agent anything..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Agent Loop
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Agent is thinking... (Running locally on your CPU)"):
            while True:
                try:
                    response = ollama.chat(model='llama3.2:3b', messages=st.session_state.messages, tools=tools_schema)
                    assistant_message = response['message']
                    st.session_state.messages.append(assistant_message)

                    if 'tool_calls' in assistant_message and assistant_message['tool_calls']:
                        for tool in assistant_message['tool_calls']:
                            tool_name = tool['function']['name']
                            tool_args = tool['function']['arguments']
                            
                            # Show Agent Thoughts
                            with st.status(f"🧠 Using tool: **{tool_name}**", expanded=True):
                                st.write(f"**Arguments:** {tool_args}")
                                logging.info(f"Tool Used: {tool_name} | Args: {tool_args}")
                                
                                if tool_name == "safe_calculator": tool_result = safe_calculator(tool_args['expression'])
                                elif tool_name == "search_documents": tool_result = search_documents(tool_args['query'])
                                elif tool_name == "inspect_dataset": tool_result = inspect_dataset(tool_args['csv_path'])
                                else: tool_result = "Error: Unknown tool"
                                
                                st.write(f"**Observation:** {tool_result[:500]}...")
                                logging.info(f"Tool Result: {tool_result}")
                            
                            st.session_state.messages.append({"role": "tool", "name": tool_name, "content": tool_result})
                    
                    elif 'content' in assistant_message and assistant_message['content']:
                        full_response = assistant_message['content']
                        response_placeholder.markdown(full_response)
                        logging.info(f"Final Answer: {full_response}")
                        break
                    else:
                        break
                except Exception as e:
                    full_response = f"An error occurred: {e}"
                    response_placeholder.error(full_response)
                    logging.error(f"Agent Loop Error: {e}")
                    break