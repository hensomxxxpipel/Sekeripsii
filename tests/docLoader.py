from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

file_path = r"D:\Kuliah\Semester 7\Skripsi\RAG_LLM\document\Per-55-2023-Penyelenggaraan-Pendidikan-Universitas-Brawijaya-Tahun-Akademik-20232024.pdf"

loader = PyPDFLoader(file_path)
pages = loader.load()

# print(pages)

print("====pages====")
print(f"{pages[39].metadata}\n")
print(pages[39].page_content)
print("=============")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(pages)

print(f"Split blog post into {len(all_splits)} sub-documents.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="pertor_collection",
    embedding_function=embeddings,
    persist_directory="./pertor_db",  # Where to save data locally, remove if not necessary
)

print("berhasil")

vector_store.add_documents(documents=all_splits, id=1)

data = vector_store.get()
print(data.keys())      # ['ids', 'embeddings', 'documents', 'metadatas']

print("===========Jumlah dokuman==========")
print(len(data["ids"])) # jumlah dokumen

print("===========2 dokumen pertama==========")
print(data["documents"][:2])  # 2 dokumen pertama

print("===========2 metadada dokumen pertama==========")
print(data["metadatas"][:2])  # metadata 2 dokumen pertama



