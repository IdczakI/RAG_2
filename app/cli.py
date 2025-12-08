import os
import argparse
from .loader import PDFLoader
from .splitter import SectionSplitter
from .embeddings import EmbeddingManager
from .chroma_manager import ChromaDBManager
from .retriever import RetrieverWrapper
from .llm_client import OpenAIClient
from .orchestrator import QAOrchestrator


def build_or_load_index(force_rebuild=False):
    pdf_loader = PDFLoader()
    splitter = SectionSplitter()
    emb_mgr = EmbeddingManager()
    chroma_mgr = ChromaDBManager()

    if force_rebuild and os.path.exists(chroma_mgr.persist_dir):
        import shutil
        shutil.rmtree(chroma_mgr.persist_dir)

    if os.path.exists(chroma_mgr.persist_dir) and any(os.scandir(chroma_mgr.persist_dir)):
        vectordb = chroma_mgr.build_or_load([], emb_mgr.get_embedding_fn())
    else:
        pages = pdf_loader.load_all_pdfs()
        chunks = splitter.split_documents(pages)
        vectordb = chroma_mgr.build_or_load(chunks, emb_mgr.get_embedding_fn())

    return vectordb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    vectordb = build_or_load_index(force_rebuild=args.force_rebuild)
    retriever = RetrieverWrapper(vectordb).as_retriever()
    llm = OpenAIClient()
    orchestrator = QAOrchestrator(retriever, llm)

    print("System ready. Type questions or 'exit'.")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break

        res = orchestrator.answer(q)
        print("\n--- ANSWER ---")
        print(res["answer"])
        print("\n--- SOURCES ---")
        for d in res["source_documents"]:
            print(f"- {d.metadata.get('source')} | {d.page_content[:200].replace('\n',' ')}...")
        print()


if __name__ == "__main__":
    main()
