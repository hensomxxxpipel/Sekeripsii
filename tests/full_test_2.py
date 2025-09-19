# rag_hf_calme.py

import os
from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import START, StateGraph

from transformers import pipeline

load_dotenv()

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

# ================================
# 3. Embedding + Vector Store
# ================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="pertor_collection",
    embedding_function=embeddings,
    persist_directory="./pertor_db",
)

vector_store.add_documents(all_splits)

# ================================
# 4. LLM: HuggingFace pipeline
# ================================
hf_pipeline = pipeline(
    "text-generation",
    model="MaziyarPanahi/calme-3.2-instruct-78b",
    torch_dtype="auto",
    device_map="auto",
    max_new_tokens=512,
    temperature=0.7,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ================================
# 5. Prompt
# ================================
prompt = PromptTemplate.from_template(
    "Gunakan konteks berikut untuk menjawab pertanyaan.\n"
    "Jika jawabannya ada dalam konteks, berikan jawaban detail.\n"
    "Jika tidak ada, katakan 'tidak ditemukan'.\n\n"
    "Konteks:\n{context}\n\n"
    "Pertanyaan: {question}\nJawaban:"
)

# ================================
# 6. State + Functions
# ================================
class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run."]

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

def analyze_query(state: State):
    return {"query": {"query": state["question"]}}

def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(query["query"], k=10)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt_text = prompt.format(
        question=state["question"],
        context=docs_content
    )
    response = llm.invoke(prompt_text)
    return {"answer": response}

# ================================
# 7. RAG Pipeline Graph
# ================================
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# ================================
# 8. Contoh QnA
# ================================
if __name__ == "__main__":
    question = "Apa ketentuan pendaftaran mahasiswa baru?"
    result = graph.invoke({"question": question})
    print("Pertanyaan:", question)
    print("Jawaban:", result["answer"])
