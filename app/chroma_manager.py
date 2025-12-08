import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma


class ChromaDBManager:
    def __init__(self, persist_directory: str = None, collection_name: str = "books"):
        self.persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
        self.collection_name = collection_name

    def build_or_load(self, docs: List[Document], embeddings):
        if os.path.exists(self.persist_dir) and any(os.scandir(self.persist_dir)):
            print("📂 Loading existing Chroma DB...")
            return Chroma(
                persist_directory=self.persist_dir,
                collection_name=self.collection_name,
                embedding_function=embeddings,
            )

        print("⚙️ Creating new Chroma DB...")
        vectordb = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )
        return vectordb
