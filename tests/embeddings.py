from typing_extensions import Annotated, List, TypedDict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import START, StateGraph

load_dotenv()

# ================================
# 1. Contoh 4 Kalimat
# ================================
dummy_docs = [
    Document(page_content="Large language models are increasingly used in various NLP tasks."),
    Document(page_content="Prompt engineering techniques help improve model performance."),
    Document(page_content="Vector databases store embeddings for efficient retrieval."),
    Document(page_content="LangChain simplifies building applications with LLMs."),
]

# ================================
# 2. Split dokumen
# ================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(dummy_docs)
print(f"Total chunks: {len(all_splits)}")

# ================================
# 3. Embedding + Vector Store
# ================================
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

print(f"Load embeddings selesai")

vector_store = Chroma(
    collection_name="pertor_collection",
    embedding_function=embeddings,
    persist_directory="./pertor_db",
)

# tambahkan split dummy
vector_store.add_documents(all_splits)

print(f"Add document ke db selesai")

# ================================
# 4. Retrieval Only
# ================================
class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run."]

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]

def analyze_query(state: State):
    return {"query": {"query": state["question"]}}

def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(query["query"], k=2)
    return {"context": retrieved_docs}

# ================================
# 5. Graph untuk Retrieval saja
# ================================
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

print(f"Graph selesai")

# ================================
# 6. Contoh Query Retrieval
# ================================
question = "Apa fungsi prompt engineering?"
result = graph.invoke({"question": question})

print("Pertanyaan:", question)
print("\nHasil retrieval (5 dokumen teratas):")
for i, doc in enumerate(result["context"], 1):
    print(f"\n--- Dokumen {i} ---\n{doc.page_content[:500]}...")
