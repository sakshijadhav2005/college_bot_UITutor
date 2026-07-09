import os
from langchain_pinecone import PineconeVectorStore
try:
    from My_RAG.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
except ImportError:
    from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


class VectorStoreManager:
    def __init__(self, index_name: str = PINECONE_INDEX_NAME, api_key: str = PINECONE_API_KEY):
        self.index_name = index_name
        self.api_key = api_key
        print(f"VectorStoreManager initialized with index: {index_name}")
        
        # Ensure API key is set in environment so Pinecone SDK can pick it up
        if api_key:
            os.environ["PINECONE_API_KEY"] = api_key

    def create_and_save(self, chunks, embeddings):
        """Upload document chunks and embeddings to Pinecone index."""
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME environment variable is not set. Please set it in your .env file.")
        
        print(f"Uploading {len(chunks)} chunks to Pinecone index '{self.index_name}'...")
        db = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=self.index_name,
            pinecone_api_key=self.api_key
        )
        print(f"Document chunks uploaded successfully to Pinecone index: {self.index_name}")
        return db

    def load(self, embeddings):
        """Connect to the existing Pinecone index."""
        if not self.index_name:
            print("No Pinecone index name specified.")
            return None
            
        print(f"Connecting to Pinecone index '{self.index_name}'...")
        try:
            db = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=embeddings,
                pinecone_api_key=self.api_key
            )
            print("Successfully connected to Pinecone index.")
            return db
        except Exception as e:
            print(f"Failed to connect to Pinecone index: {e}")
            return None

    def delete_document(self, filename: str):
        """Delete all vectors matching the source_file metadata from the Pinecone index."""
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME is not set.")
            
        from pinecone import Pinecone
        pc = Pinecone(api_key=self.api_key)
        index = pc.Index(self.index_name)
        print(f"Deleting vectors for document '{filename}' from Pinecone index '{self.index_name}'...")
        index.delete(filter={"source_file": {"$eq": filename}})
        print(f"Vectors for document '{filename}' deleted successfully.")


