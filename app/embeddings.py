import os
from langchain_openai import OpenAIEmbeddings


class EmbeddingManager:
    def __init__(self):
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    def get_embedding_fn(self):
        return OpenAIEmbeddings(model=self.model)
