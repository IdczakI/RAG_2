import os
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document


class PDFLoader:

    def __init__(self, docs_dir: str = None):
        self.docs_dir = docs_dir or os.getenv("DOCS_DIR", "docs")

    def load_all_pdfs(self) -> List[Document]:
        if not os.path.exists(self.docs_dir):
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")

        documents: List[Document] = []
        for filename in sorted(os.listdir(self.docs_dir)):
            if not filename.lower().endswith(".pdf"):
                continue
            path = os.path.join(self.docs_dir, filename)
            reader = PdfReader(path)
            for page_index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                metadata = {
                    "source": filename,
                    "page_number": page_index,
                }
                documents.append(
                    Document(page_content=text, metadata=metadata)
                )

        return documents
