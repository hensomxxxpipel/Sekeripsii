import os
import time
import random
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



load_dotenv()

LANGSMITH_TRACING="true"
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

tracer = LangChainTracer(project_name = LANGSMITH_PROJECT)
callback_manager = CallbackManager([tracer])

# Inisialisasi LLM dengan model berbeda
llm_models = {
    "llama4": ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        streaming=True,
        temperature=0.7,
        callback_manager=callback_manager,
    ),
    
    "deepseek": ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="deepseek-r1-distill-llama-70b",
        streaming=True,
        temperature=0.7,
        callback_manager=callback_manager,
    ),
    
    "llama3": ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        streaming=True,
        temperature=0.7,
        callback_manager=callback_manager,
    ),

    "gemini.flash": ChatGoogleGenerativeAI(
        api_key=GEMINI_API_KEY,
        model="gemini-2.5-pro-exp-03-25",
        temperature=0.7,
        streaming=True,
        callback_manager=callback_manager,
    ),
}

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
