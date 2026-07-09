import sys
import io
sys.path.append('.')

# Set up encoding to prevent Windows console encoding crashes
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from My_RAG.document_loader import load_directory, split_documents
from My_RAG.embeddings import EmbeddingManager
from My_RAG.vectorstore import VectorStoreManager

def main():
    print("==================================================")
    print("🚀 STARTING END-TO-END RAG PIPELINE WORKFLOW TEST")
    print("==================================================")
    
    # 1. Load and parse the notebook
    print("\nStep 1: Scanning directory for Jupyter Notebooks...")
    docs = load_directory("My_RAG")
    if not docs:
        print("❌ No notebook documents found in My_RAG directory.")
        return
    print(f"✅ Loaded {len(docs)} cells/documents.")
    
    # 2. Split the documents
    print("\nStep 2: Splitting documents into chunks...")
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    print(f"✅ Generated {len(chunks)} chunks.")
    
    # 3. Initialize embeddings
    print("\nStep 3: Initializing Embedding Model (all-MiniLM-L6-v2)...")
    emb_manager = EmbeddingManager()
    embeddings = emb_manager.get_embeddings()
    print("✅ Embedding model initialized successfully.")
    
    # 4. Connect and Upsert to Pinecone
    print("\nStep 4: Connecting and Upserting to Pinecone...")
    try:
        vs_manager = VectorStoreManager()
        db = vs_manager.create_and_save(chunks, embeddings)
        print("✅ Document chunks successfully stored in Pinecone.")
    except Exception as e:
        print(f"❌ Error during Pinecone storage: {e}")
        return
        
    # 5. Verify Ingestion with a Query
    print("\nStep 5: Testing Vector Retrieval Query...")
    query = "What does the PDF Document Loader do?"
    print(f"Running similarity search query: '{query}'")
    try:
        results = db.similarity_search(query, k=2)
        print(f"✅ Retrieval successful. Found {len(results)} matches:")
        for idx, result in enumerate(results):
            print(f"\n--- Match {idx + 1} ---")
            print(f"Source: {result.metadata.get('source_file')}")
            print(f"Content Preview: {result.page_content[:200]}...")
    except Exception as e:
        print(f"❌ Error during similarity search query: {e}")
        
    print("\n==================================================")
    print("🎉 WORKFLOW TEST COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
