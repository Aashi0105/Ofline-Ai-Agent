import ollama
import ast
import operator
import logging
import os
import io
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ==========================================
# 1. LOGGING SETUP
# ==========================================
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/agent.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================================
# 2. CHROMADB SETUP (For RAG Tool)
# ==========================================
print("Loading Offline Vector Database...")
ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents", embedding_function=ollama_ef)

# ==========================================
# 3. THE TOOLS
# ==========================================

# Tool A: Safe Calculator
ALLOWED_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg
}

def safe_calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode='eval')
        def _eval(node):
            if isinstance(node, ast.Constant): return node.value
            elif isinstance(node, ast.BinOp): return ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp): return ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
            else: raise ValueError("Unsupported")
        return str(_eval(tree.body))
    except Exception as e:
        return f"Error calculating: {e}"

# Tool B: RAG Search with Citations
def search_documents(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=2)
    
    if not results['documents'][0]:
        return "No relevant documents found."
    
    output = ""
    # Loop through the results and attach the page metadata
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        page_num = meta.get('page', 'Unknown')
        output += f"Content: {doc}\n(Source: Page {page_num})\n\n"
    
    return output.strip()

# Tool C: Dataset Inspector
def inspect_dataset(csv_path: str) -> str:
    try:
        df = pd.read_csv(csv_path)
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        describe_str = str(df.describe())
        head_str = str(df.head(3))
        
        return f"Dataset Info:\n{info_str}\n\nFirst 3 rows:\n{head_str}\n\nStatistics:\n{describe_str}"
    except Exception as e:
        return f"Error reading CSV: {e}"

# ==========================================
# 4. TOOL SCHEMAS
# ==========================================
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "safe_calculator",
            "description": "Evaluates a mathematical expression. Use ONLY for math with numbers (like +, -, *, /). Do NOT use this tool for general knowledge questions or words.",
            "parameters": {
                "type": "object",
                "properties": { "expression": { "type": "string", "description": "e.g. '25 * 4'" } },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Searches the local offline database for information related to the query. Returns relevant text and page numbers.",
            "parameters": {
                "type": "object",
                "properties": { "query": { "type": "string", "description": "The search query" } },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset",
            "description": "Reads a CSV file and returns its columns, data types, and statistics. Use when the user asks about a dataset or CSV file.",
            "parameters": {
                "type": "object",
                "properties": { "csv_path": { "type": "string", "description": "Path to the CSV file, e.g. 'data/data.csv'" } },
                "required": ["csv_path"]
            }
        }
    }
]

# ==========================================
# 5. THE AGENT CONTROLLER
# ==========================================
def run_agent(user_query: str):
    print(f"\n🧑 User: {user_query}")
    logging.info(f"User Query: {user_query}")
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful offline AI assistant. ONLY use tools when necessary: if you need to do math, search documents, or inspect a dataset. For general knowledge or conversational questions, answer directly WITHOUT using any tools. Always cite page numbers if provided by the search tool."
        },
        { "role": "user", "content": user_query }
    ]
    
    while True:
        response = ollama.chat(model='llama3.2:3b', messages=messages, tools=tools_schema)
        assistant_message = response['message']
        messages.append(assistant_message)
        
        if 'tool_calls' in assistant_message and assistant_message['tool_calls']:
            for tool in assistant_message['tool_calls']:
                tool_name = tool['function']['name']
                tool_args = tool['function']['arguments']
                
                print(f"🔧 Agent using tool: {tool_name} | Args: {tool_args}")
                logging.info(f"Tool Used: {tool_name} | Args: {tool_args}")
                
                # ROUTING
                if tool_name == "safe_calculator":
                    tool_result = safe_calculator(tool_args['expression'])
                elif tool_name == "search_documents":
                    tool_result = search_documents(tool_args['query'])
                elif tool_name == "inspect_dataset":
                    tool_result = inspect_dataset(tool_args['csv_path'])
                else:
                    tool_result = "Error: Unknown tool"
                
                print(f"📄 Tool Result: {tool_result[:200]}...") # Truncate printing for terminal cleanliness
                logging.info(f"Tool Result: {tool_result}")
                
                messages.append({ "role": "tool", "name": tool_name, "content": tool_result })
        
        elif 'content' in assistant_message and assistant_message['content']:
            print(f"🤖 Agent: {assistant_message['content']}\n")
            logging.info(f"Final Answer: {assistant_message['content']}")
            break
        else:
            break

# ==========================================
# 6. TEST IT
# ==========================================
if __name__ == "__main__":
    # Test 1: General Knowledge (Should NOT use tools)
    run_agent("What is the capital of France?")
    
    # Test 2: Calculator
    run_agent("What is 100 divided by 3.5?")
    
    # Test 3: RAG (You MUST run ingest.py first and have a sample.pdf!)
    # Uncomment the line below to test RAG after ingesting your PDF
    run_agent("What is the main topic of the document?")
    
    # Test 4: CSV Inspector (Make sure data/data.csv exists!)
    # Uncomment the line below to test CSV after putting a csv in data/
    run_agent("Can you inspect the dataset at data/data.csv and tell me what columns it has?")