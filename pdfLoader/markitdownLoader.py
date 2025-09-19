from markitdown import MarkItDown

file_path = r"D:\Kuliah\Semester 7\Skripsi\RAG_LLM\document\Per-55-2023-Penyelenggaraan-Pendidikan-Universitas-Brawijaya-Tahun-Akademik-20232024.pdf"


md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
result = md.convert(file_path)
print(result.text_content)