from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from My_RAG.config import EMBEDDING_MODEL_NAME
except ImportError:
    from config import EMBEDDING_MODEL_NAME



class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Initializing EmbeddingManager with model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("EmbeddingManager initialized successfully.")

    def get_embeddings(self):
        return self.embeddings
