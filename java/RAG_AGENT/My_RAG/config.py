import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Chunking Config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding Config
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Pinecone Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Gemini LLM Config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

