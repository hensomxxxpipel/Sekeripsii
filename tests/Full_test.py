from typing import Literal
from typing_extensions import Annotated, List, TypedDict

import os
from dotenv import load_dotenv
from langchain import hub
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import START, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI   # bisa diganti dengan model lain
from langchain_core.prompts import PromptTemplate

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ================================
# 1. Load PDF
# ================================
file_path = r"D:\Kuliah\Semester 7\Skripsi\RAG_LLM\document\Per-55-2023-Penyelenggaraan-Pendidikan-Universitas-Brawijaya-Tahun-Akademik-20232024.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()

# ================================
# 2. Split dokumen
# ================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(pages)
print(f"Total chunks: {len(all_splits)}")

# ================================
# 3. Embedding + Vector Store
# ================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/Qwen/Qwen3-Embedding-8B")

vector_store = Chroma(
    collection_name="pertor_collection",
    embedding_function=embeddings,
    persist_directory="./pertor_db",
)

vector_store.add_documents(pages)

# ================================
# 4. Retrieval & Generation
# ================================
# Definisikan schema query
class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run."]

# Prompt dasar dari LangChain Hub (opsional bisa buat manual)
# prompt = hub.pull("rlm/rag-prompt")

prompt = PromptTemplate.from_template(
    "Gunakan konteks berikut untuk menjawab pertanyaan.\n"
    "Jika jawabannya ada dalam konteks, berikan jawaban detail dan lengkap.\n"
    "Jika tidak ada, katakan 'tidak ditemukan'.\n\n"
    "Konteks:\n{context}\n\n"
    "Pertanyaan: {question}\nJawaban:"
)

# LLM untuk generation
llm = ChatGoogleGenerativeAI(
        api_key=GEMINI_API_KEY,
        model="gemini-2.0-flash",
        temperature=0.7
        )  # ganti dengan model lain jika perlu

# Definisikan state
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

def analyze_query(state: State):
    return {"query": {"query": state["question"]}}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(query["query"], k=1000)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# ================================
# 5. RAG Pipeline Graph
# ================================
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# ================================
# 6. Contoh QnA
# ================================
question = "Apa syarat umum pendaftaran program pascasarjana?"
result = graph.invoke({"question": question})
print("Pertanyaan:", question)
print("Jawaban:", result["answer"])
