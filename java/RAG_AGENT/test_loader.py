import sys
import io
sys.path.append('.')

# Reconfigure stdout/stderr to use UTF-8 with character replacement on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from My_RAG.document_loader import load_directory, split_documents

print("Scanning and loading documents...")
docs = load_directory("My_RAG")
if docs:
    print(f"Loaded {len(docs)} documents.")
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)
    print(f"Split into {len(chunks)} chunks.")
    if chunks:
        print("\n--- Example Chunk ---")
        print(chunks[0].page_content)
        print("---------------------")
else:
    print("No documents loaded.")
