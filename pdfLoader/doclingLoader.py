# from docling.document_converter import DocumentConverter

# source = r"D:\Kuliah\Semester 7\Skripsi\RAG_LLM\document\Per-55-2023-Penyelenggaraan-Pendidikan-Universitas-Brawijaya-Tahun-Akademik-20232024.pdf"
# converter = DocumentConverter()
# doc = converter.convert(source).document

# print(doc.export_to_markdown())  # output: "### Docling Technical Report[...]"

from langchain_docling import DoclingLoader

FILE_PATH = r"D:\Kuliah\Semester 7\Skripsi\RAG_LLM\document\Per-55-2023-Penyelenggaraan-Pendidikan-Universitas-Brawijaya-Tahun-Akademik-20232024.pdf"

loader = DoclingLoader(file_path=FILE_PATH)