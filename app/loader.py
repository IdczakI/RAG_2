import os
from typing import List
from langchain_core.documents import Document


class TextLoader:

    def __init__(self, docs_dir: str = None):
        self.docs_dir = docs_dir or os.getenv("DOCS_DIR", "docs")

    def load_all_txt(self) -> List[Document]:
        if not os.path.exists(self.docs_dir):
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")

        documents: List[Document] = []
        for filename in sorted(os.listdir(self.docs_dir)):
            if not filename.lower().endswith(".txt"):
                continue

            path = os.path.join(self.docs_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            metadata = {
                "source": filename,
            }
            documents.append(Document(page_content=text, metadata=metadata))

        return documents
