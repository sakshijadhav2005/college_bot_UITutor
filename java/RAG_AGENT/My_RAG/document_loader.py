import os
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader, NotebookLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(source: str) -> List[Any]:
    """Load content from Web URL, PDF, TXT, or Jupyter Notebook (.ipynb/.pynb) file."""
    if source.startswith("http"): 
        loader = WebBaseLoader(source)
    elif source.endswith(".pdf"):
        loader = PyPDFLoader(source)
    elif source.endswith(".ipynb") or source.endswith(".pynb"):
        loader = NotebookLoader(source, include_outputs=True)
    else:
        loader = TextLoader(source, encoding="utf-8")
    return loader.load()

def load_directory(directory: str) -> List[Any]:
    """Process all Jupyter Notebook (.ipynb/.pynb) files in a directory recursively."""
    all_documents = []
    dir_path = Path(directory)
    
    # Find all notebook files recursively
    notebook_files = list(dir_path.glob("**/*.ipynb")) + list(dir_path.glob("**/*.pynb"))
    
    print(f"Found {len(notebook_files)} notebook files to process")
    
    for notebook_file in notebook_files:
        print(f"\nProcessing: {notebook_file.name}")
        try:
            loader = NotebookLoader(str(notebook_file), include_outputs=True)
            documents = loader.load()
            
            # Add custom metadata fields to each document cell
            for doc in documents:
                doc.metadata['source_file'] = notebook_file.name
                doc.metadata['file_type'] = 'notebook'
            
            all_documents.extend(documents)
            print(f"  [SUCCESS] Loaded {len(documents)} notebook cells/documents")
            
        except Exception as e:
            print(f"  [ERROR] Error loading {notebook_file.name}: {e}")

            
    print(f"\nTotal notebook documents loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents: List[Any], chunk_size=1000, chunk_overlap=200) -> List[Any]:
    """Split loaded document pages or notebook cells into smaller chunks for optimal vector search."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

